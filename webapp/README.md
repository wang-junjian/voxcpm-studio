# VoxCPM2 Studio — Web 语音合成应用

基于本地部署的 VoxCPM2（2B）构建的浏览器端语音合成界面，**三推理引擎**（PyTorch / MLX / GGUF），模型常驻内存，无需重复加载。

## 启动

```bash
cd voxcpm-studio
./webapp/start.sh                    # http://127.0.0.1:8817

MLX=0  ./webapp/start.sh             # 不起 MLX 引擎
GGUF=0 ./webapp/start.sh             # 不起 GGUF 引擎
PORT=9000 ./webapp/start.sh          # 换端口
HOST=0.0.0.0 ./webapp/start.sh       # 局域网可访问（手机测试）
```

`start.sh` 会同时拉三个进程：主服务（`.venv`，PyTorch，端口 8817）、MLX worker（`mlxrun/.venv`，8818）、GGUF server（C++ 二进制 `llama-tts-server`，8819）。三者依赖互斥，只能进程隔离。页面顶部状态点变绿「就绪」后可生成；MLX 约 1–2 秒就绪，GGUF 约 1 秒，PyTorch 冷启动 40–90 秒，三个引擎按钮会逐个点亮。

## 三个引擎

界面上「推理引擎」可随时切换，选择存在 localStorage。实测数据（M2 Max / 本机 / 同一段 35 字中文）：

| | MLX 8bit | GGUF Q8_0 | PyTorch fp32 |
|---|---|---|---|
| 权重 | 3.0 GB | 3.3 GB（BaseLM 1.6 + Acoustic 1.7） | 4.6 GB |
| 常驻内存 | 3.12 GB | **3.27 GB（完全稳定）** | 11.6–12.3 GB |
| 加载 | **0.2–0.7 s** | 0.87 s | 15 s（暖）/ 47–90 s（冷） |
| RTF | **0.83**（区间 0.69–0.83） | 1.33（区间 1.33–1.53） | 1.04（区间 0.85–1.5） |
| 流式首包 | 0.53 s | **0.30–0.36 s** | 0.79–0.88 s |
| 流式节奏 | 快于实时，播放不断点 | **慢于实时**（160 ms 音频 / 240 ms 到达） | 快于实时 |
| ASR 回环 CER | 5.71%（同音误识） | **0 %** | 0 % |

默认走 MLX；worker 不在线时自动回落到其它可用引擎，界面会把不可用的引擎按钮置灰。请求体里 `"engine": "auto" | "mlx" | "gguf" | "torch"` 可显式指定。

三条路线的完整对比（架构图、性能图表、五个发现、复现命令）见 [tri-route-comparison.html](../tri-route-comparison.html)。

## 功能

| 能力 | 说明 |
|---|---|
| 引擎切换 | MLX 8bit / GGUF Q8_0 / PyTorch fp32，一键切换并记住选择 |
| 基础 TTS | 输入文本直接朗读，使用模型默认音色 |
| 音色设计 | 括号内写音色描述，如 `(温和的年轻女声)你好`，括号内容不会被念出 |
| 声音克隆 | 拖拽/点击上传参考音频（wav/mp3/flac），复刻其音色 |
| 极致克隆 | 克隆时额外填入参考音频的原话，相似度更高 |
| 流式播放 | 生成的同时边收边播；三个引擎都是真流式，首包 0.3–0.9 秒 |
| 参数调节 | 引导强度 cfg（1–4）、推理步数（4–30）、随机种子 |
| 波形交互 | 点击波形任意位置跳转播放 |
| 历史记录 | 服务端留存最近记录，带引擎标记，可回放、下载、删除 |

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 三个引擎各自的状态、设备、采样率、加载耗时 |
| POST | `/api/tts` | 一次性合成，返回 JSON（含引擎/时长/RTF/波形峰值）与音频 URL |
| POST | `/api/stream` | 流式合成，返回 L16 裸 PCM 分块（响应头带 `X-Sample-Rate`、`X-Engine`） |
| POST | `/api/ref` | 上传参考音频，返回 `ref_id` |
| GET | `/api/audio/{id}.wav` | 取回生成的音频 |
| GET | `/api/history` | 最近生成记录 |
| DELETE | `/api/history/{id}` | 删除某条记录 |

`POST /api/tts` 请求体：

```json
{
  "text": "要合成的文本",
  "mode": "basic | design | clone",
  "engine": "auto | mlx | gguf | torch",
  "ref_id": "克隆模式下的参考音频 id",
  "prompt_text": "参考音频对应的原话（可选，极致克隆用）",
  "cfg": 2.0,
  "steps": 10,
  "seed": 42,
  "chunk_patches": 4
}
```

`chunk_patches` 只对 MLX 生效：每攒够几个 patch 解码吐一块（1 patch = 160 ms 音频）。默认 4 → 首包 0.53 s；调到 16 首包变 1.85 s 但总耗时少 9%。

一次性合成（`/api/tts` 走 MLX）在 worker 侧按 `chunk_patches=16` 分块解码：整段一次解码的峰值与文本长度线性相关（19.7 秒音频峰值 8.1 GB），分块后固定 4.3 GB。代价是与"整段精确解码"有 0.039% 的样本差 1 LSB（SNR 103 dB，听不出）。要逐样本一致时给 worker 传 `"exact": true`。

MLX worker 自己也有三个接口（供主服务代理，也可直接用）：`GET /health`、`POST /gen`（整段 PCM）、`POST /stream`（chunked PCM）。显存缓冲池上限由环境变量 `MLX_CACHE_LIMIT_GB` 控制（默认 2.0），每次生成结束后会 `clear_cache`。

GGUF 引擎是 llama.cpp-omni 的 `llama-tts-server`（需自行编译，见 [../docs/ASSETS.md](../docs/ASSETS.md) 第 3 步；本仓库只提供对上游的 [patch](../extensions/llamacpp-omni/voxcpm2-perf.patch)），OpenAI 兼容接口，主服务用 httpx 代理。接它有两个坑：

- `/health` 恒返回 ok，**不能**用它判断就绪；要看 `GET /v1/audio/speech/models` 是否返回非空列表。
- 它的流式响应是「44 字节 WAV 头 + 裸 PCM」，主服务会把头剥掉，才能对齐另外两个引擎的 L16 口径。

```bash
# 直接调 GGUF worker（不经主服务）
curl -X POST http://127.0.0.1:8819/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"voxcpm2","input":"你好","seed":42}' --output out.wav
```

GGUF 的权重由 `llamacpp-omni/llama.cpp-omni-master/tools/omni/voxcpm2/convert_voxcpm2_to_gguf.py` 转换，
Q8_0 量化用同目录编译出的 `llama-quantize`，产物在 `gguf-models/`（`start.sh` 默认取 `VoxCPM2-BaseLM-Q8_0.gguf` + `VoxCPM2-Acoustic-F16.gguf`，可用环境变量覆盖）。

```bash
curl -X POST http://127.0.0.1:8817/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","mode":"basic","engine":"mlx","cfg":2,"steps":10,"seed":42}'
```

## 架构

```
浏览器（单文件 HTML，零外部依赖）
   │  fetch / ReadableStream
   ▼
FastAPI（webapp/server.py，.venv）
   │ engine=mlx / gguf 时用 httpx 流式代理
   │ threading.Lock 串行
   ├──────────────► VoxCPM2 / MPS fp32（常驻 11.6 GB）
   ├──────────────► MLX worker（mlxrun/mlx_worker.py，8818，stdlib HTTP）→ 3.1 GB
   └──────────────► llama-tts-server（C++/ggml Metal，8819）→ 3.3 GB
```

四个关键设计：

1. **模型常驻 + 进程隔离**：torch、mlx 与 ggml 三套运行时依赖互斥（同一 venv 装不下，GGUF 干脆是 C++ 二进制），所以各自独立进程、独立端口，主服务代理转发。三个模型可同时常驻（本机 64 GB 内存够用），切换引擎零延迟。
2. **串行推理**：MPS 上并发调用同一模型会相互干扰，PyTorch 侧所有请求共用一把 `threading.Lock`；MLX worker 内部同理。端点写成同步函数，由 Starlette 放进线程池，不阻塞事件循环。
3. **流式协议**：裸 PCM（int16 小端）分块传输，采样率走响应头。前端用 `AudioContext` 逐块建 `AudioBuffer` 排队播放，同时累积数据供波形绘制与下载。**这个 AudioContext 跨请求复用，一段合成结束时绝不 `close()`**——见下方「流式尾音被截断」。
4. **MLX 真流式**：mlx-audio 自带的 `generate()` 是假流式（整段跑完只 yield 一次，首包 4.46 s）。`mlxrun/voxcpm_stream.py` 把它的自回归循环搬出来，每 4 个 patch 解码一次；分块解码靠解码器的纯因果结构，带 16 帧左上下文后与整段解码的 SNR 116 dB、与 `generate()` 逐样本一致。细节与实测曲线见 [mlxrun/BENCH.md](../mlxrun/BENCH.md)。

## 验证结果

真实 Chrome 走完整交互链路，`verify_mlx.cjs`（MLX 引擎）：

```
引擎控件       MLX 8bit / PyTorch 两个按钮均可点，默认 MLX，状态栏「MLX 8bit · Metal」
一次性合成     6.56 s 音频，RTF 0.69，波形 57028 像素，下载链接正常
流式合成       点按到首块 PCM 528 ms（服务端请求头 15 ms），7.04 s 音频，RTF 0.72
              blob 可解码，末 0.9 s 峰值 RMS 0.1116，尾音未被切断
历史记录       18 条，引擎标记 MLX / PT 正确
控制台错误     0
```

`verify_web.cjs`（原有五场景：基础/音色设计/克隆/历史/流式）依旧全绿。`verify_gguf.cjs`（GGUF 引擎）：

```
引擎控件       三个按钮（MLX / GGUF / PyTorch）均可用
切换到 GGUF    状态栏「GGUF Q8_0 · Metal」，spec 面板同步
一次性合成     8.32 s 音频，RTF 1.34，下载链接正常
流式合成       首块 PCM 362 ms，6.88 s 音频，RTF 1.51
              blob 可解码，末 0.9 s 峰值 RMS 0.2105，尾音未被切断
历史记录       引擎标记 GGUF / MLX / PT 正确
控制台错误     0
```

GGUF 侧另做了内存浸泡：9 次混合长度请求，进程 footprint 3.27 GB → 3.27 GB，无增长。

ASR 回环（FunASR paraformer-zh）对 MLX 流式产物：

```
参考: 流式播放的问题是尾部的几个字始终没有读，这句话的最后一个字必须完整念出来。
识别: 刘氏播放的问题是尾部的几个字始终没有读这句话的最后一个字必须完整念出来
CER 5.71%（两处误差都是「流式→刘氏」同音误识，尾句完整）
```

同一段话走 PyTorch 与 GGUF 都是 CER 0%。

移动端（390px）单列布局，无横向滚动。

## 已修复的坑：流式尾音被截断

**现象**：流式播放时最后两三个字永远不念。

**定位**（`measure_tail.cjs`，钩住 `AudioBufferSourceNode.start` 与 `AudioContext.close`）：

```
scheduledEnd 5.996s   closedAt 5.483s   cutSeconds 0.513s
```

**根因**：`generateStream()` 在读完流的瞬间 `await ac.close()`。但音频是排队播放的——最后几块 `AudioBuffer` 还排在**未来的**时间点上，`close()` 会连同它们一起丢弃。生成越快丢得越多（RTF 0.85 时约丢 15% 时长），所以表现为「总是」丢尾巴。

**关键区分**：这不是模型少生成。同一文本同时走 `/api/tts` 与 `/api/stream`：

```
一次性 276480 样本 / 5.760s
流式   276480 样本 / 5.760s
整体相关系数 1.0000    末 0.6s 相关系数 1.0000
ASR 回环 CER 0.00%（27 字，末尾「丢尾巴」命中）
```

**修法**：
- `AudioContext` 改为跨请求复用（`getAudioCtx(sr)`），一段合成结束**不关闭**；
- 用 `streamSources` 跟踪已排期音源，播完 `onended` 出队；
- 新一次生成 / 回放整段时先 `stopStreamSources()`，避免叠音；
- 波形进度改为按时间轴推进（`played / 总时长`），不再每块都画满。

**回归**：`measure_tail.cjs` → `tailReached true / closes 0 / forcedStops 0`；MLX 引擎复测末 0.9 s 峰值 RMS 0.1116。

## 已修复的坑：低步数下输出「声音非常小」

现象：某些文本在默认 10 步下整段变成气声，RMS 只有正常的 1/8、峰值 1/5。前端 seed 默认固定 42，
所以同一段文本每次重试都是**同一段**轻音频，看起来像引擎坏了。

根因是**低扩散步数未收敛**，不是量化、不是攒批（`CHUNK_PATCHES=1` 与 `4` 逐位相同）：

| 实现 | 16 个种子的命中率 |
|---|---|
| PyTorch（参考） | 0 / 16 |
| GGUF Q8_0 | 1 / 16 |
| GGUF F16 | 1 / 16 |
| MLX 8bit | 2 / 16 |

同一段文本把 timesteps 10 → 16 → 20 → 30 → 40，RMS 单调回到 0.013 → 0.038 → 0.062 → 0.084 → 0.092。
退化样本放大后 CER 仍 21%（正常 7.9%），说明内容已损伤，**必须换种子重生成**。

修复在 `/api/tts`：

1. 生成后测 RMS/峰值，命中 `VOXCPM_DEGEN_*` 阈值就换 seed 重生成（最多 `VOXCPM_MAX_ATTEMPTS` 次），取最响的一次；
2. 正常样本统一响度到 `VOXCPM_TARGET_RMS`（默认 0.10 ≈ −20 dBFS），峰值上限 `VOXCPM_PEAK_CEIL`。

三个引擎实测都稳定在 RMS 0.100、峰值 0.48–0.95；退化案例（文本 + seed 42）自动换到 seed 43，
CER 从 21.05% 回到 7.89%。结果栏会标注「换种子重试 N 次」「响度归一 ×X」。

**流式播放不做这两件事**（边生成边播，无法回头重来），要稳定电平请用非流式生成（界面默认）。

## 已修复的坑：GGUF 流式「断断续续」

**现象**：打开流式播放后，GGUF 的声音一块一块地往外蹦，中间反复静音。

**量化**（`measure_stream.py` + `verify_stream_smooth.cjs`，记录每块的到达时刻与音频时长）：

```
74 字长句：音频 12.96s，生成耗时 18.5s（RTF 1.42），共 27 块
块大小 130–640ms，块间隔均值 698ms、最长 943ms
按「收到第一块就播」：20 次断供，累计静音 5.8s
```

**根因**：播放器恒按 1× 消耗，而合成速率 r（音频秒/墙钟秒）只有 **0.70**。
收到第一块就开播时，缓冲必然在中途耗尽——这不是调度抖动，是**结构性欠载**：
整段音频要 18.5s 才生成完，播放却在 13s 内就要放完，缺口必须由缓冲垫住。

**修法**（`static/index.html` / `generateStream`）：加缓冲闸门。播放全程不断供的充要条件是

```
开播时已缓冲 B ≥ (1 - r) · T + 抖动余量      （T 为整段音频长度）
```

- `r` 由本次流实时估算（用第一块之后的音频/耗时，排除首包延迟与模型预热），并按引擎缓存进 `localStorage`；
- `T` 由字数 × 实测「字数→音频秒」估算，缓存同样跨会话复用；
- 抖动余量取观测到的最长块间隔的 0.6 倍、并按段长缩放（短句不必多等）；
- `r ≥ 0.98` 的引擎（MLX 实测 1.39、PyTorch 1.04）闸门退化为 0.12s，等于立即播；
- 缓冲期间结果栏显示「缓冲 2.1/4.9s」，播完自动隐藏。

**实测**（真实 Chrome，同一段 74 字文本）：

| 策略 | 开播时刻 | 断供次数 | 累计静音 |
|---|---|---|---|
| 低延迟（旧行为，开关可切回） | 0.35 s | 20 | 5.8 s |
| **顺滑（新默认）** | **7.6 s** | **0** | **0** |
| 短句 2.4s 音频 | 2.1 s | 0 | 0 |
| MLX / PyTorch 流式 | 0.58 s / 0.88 s | 0 | 0 |

注意 7.6s 是被物理下限决定的：要播完 13s 且不断供，最早只能在 12.96 ×(1/0.7 − 1) ≈ 5.6s 开播，
再加上块间隔抖动与 640ms 的块粒度，实际落在 7.6s。想更快只能提 GGUF 吞吐或改用 MLX。

界面新增「低延迟优先」开关（记忆选择）：打开即回到旧行为（收到就播，但会断续），用来对比。

## 已知限制

- **单实例串行**：一个请求处理时同一引擎的后续请求排队。要并发需起多实例或换 vLLM-Omni 之类方案。
- **GGUF 流式慢于实时**：合成速率只有实时的 0.70 倍，无法边生成边连续播放，页面会先缓冲到欠载量被覆盖再开播（长句约 7.6s 才出声，但不间断）。要低延迟且无缝请用 MLX（RTF 0.72，首声 0.58s）。
- **GGUF 量化不提速**：Q8_0 相对 F16 只省 1.4 GB 体积，RTF 反而略高（1.40 vs 1.36），因为 Metal 上反量化开销吃掉了带宽收益。
- **Chrome 的 `ERR_ABORTED` 标注**：流式响应使用 chunked 传输，Chrome DevTools 会在流正常结束后把该请求标记为 aborted。实测数据完整（PCM 字节数、解码时长均正确），curl 侧连接也正常收尾，属浏览器侧标注而非故障。
- **内存**：三个引擎同时常驻约 18.2 GB（PyTorch 11.6 + MLX 3.1 + GGUF 3.3），推理峰值更高。只要轻量引擎就设 `MLX=0` 或 `GGUF=0`；主服务始终会加载 PyTorch 作为兜底。
- **历史记录**：音频存在 `webapp/runs/`，不自动清理，定期手动删。
- **无鉴权**：默认只监听 127.0.0.1。用 `HOST=0.0.0.0` 暴露到局域网时请自行加访问控制。

## 文件

```
webapp/
├── server.py           FastAPI 后端（三引擎路由 + MLX/GGUF 代理）
├── static/index.html   单文件前端（全部 CSS/JS 内联）
├── start.sh            启动脚本（主服务 + MLX worker + GGUF server）
├── verify_web.cjs      端到端验证（真实 Chrome，五场景）
├── verify_mlx.cjs      MLX 引擎端到端验证（切引擎/首包/尾音）
├── verify_gguf.cjs     GGUF 引擎端到端验证
├── verify_stream_smooth.cjs  流式顺滑度（断供次数/开播时刻，含无闸门对照模拟）
├── verify_buffer_ui.cjs      缓冲阶段截图 + 热缓存下的开播时刻
├── measure_tail.cjs    流式尾音是否播完（钩 AudioContext 排期/close）
├── shot_streaming.cjs  播放中途截图
├── check_mobile.cjs    移动端布局检查
├── debug_stream.cjs    流式请求调试
├── runs/               生成的音频与元数据
├── refs/               上传的参考音频
└── shots/              验证截图
```

```
mlxrun/
├── voxcpm_stream.py    真流式生成 + 分块解码 + LiteTokenizer
├── mlx_worker.py       MLX HTTP worker（stdlib，独立 venv）
├── verify_stream.py    等价性 / 首包 / 分块参数验证
├── BENCH.md            方案实测与选型
└── models/VoxCPM2-8bit 8bit 权重 3.0 GB

llamacpp-omni/llama.cpp-omni-master/
├── tools/omni/voxcpm2/ VoxCPM2 的 C++/ggml 实现 + 转换器
├── tools/server/server-voxcpm2.cpp  llama-tts-server（OpenAI 兼容）
└── build-metal/bin/    voxcpm2-cli / llama-tts-server / llama-quantize

gguf-models/
├── VoxCPM2-BaseLM-Q8_0.gguf     1.6 GB（BaseLM，量化）
├── VoxCPM2-Acoustic-F16.gguf    1.7 GB（声学部分）
├── VoxCPM2-BaseLM-F16.gguf      3.0 GB（对照用）
├── VoxCPM2-BaseLM-F32.gguf      6.0 GB（量化输入）
└── VoxCPM2-Acoustic-F32.gguf    3.3 GB（对照用）
```
