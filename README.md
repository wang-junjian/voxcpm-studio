# VoxCPM Studio

VoxCPM2（OpenBMB，2B 无分词器 TTS）在 Apple Silicon 上的**本地语音合成工作台**：一个浏览器界面，三条可切换的推理路线，两份实测量化报告。

全部推理在本地完成，不联网、不调云 API。这个项目想回答一个具体问题：**同一个模型，在 Mac 上用什么方式跑最快、最省内存、最容易维护。**

---

## 三条路线，一句话结论

| 路线 | RTF | 首包 | 常驻内存 | 加载 | ASR CER |
|---|---|---|---|---|---|
| **MLX 8bit** | **0.83** | 528 ms | **3.12 GB** | **0.7 s** | 5.71% |
| **GGUF Q8_0** | 1.33\* | **301–362 ms** | 3.27 GB | ~0.9 s | **0%** |
| **PyTorch fp32** | 1.04 | 790–880 ms | 11.64 GB | 90.9 s | 2.94% |

\* 服务态隔离测量；与 PyTorch 同时常驻时因统一内存争用会升到 1.50。

- **日常用 MLX**：四项全胜参考实现（PyTorch），首包进入可用区间，内存只占 3.1 GB。
- **要最低首包延迟用 GGUF**：首包 300 ms 出头，但整体吞吐慢 1.6 倍，且流式节奏慢于实时。
- **PyTorch 只作兜底与对照**：最忠实、最稳，但 11.6 GB 常驻、加载 91 秒。

两份完整实测报告（架构图、性能图表、复现命令、局限说明）：
[**tri-route-comparison.html**](tri-route-comparison.html) · [**gguf-efficiency.html**](gguf-efficiency.html)

---

## 架构

三个进程，一个前端，由主服务统一路由：

```
浏览器 (127.0.0.1:8817)
    │  单页前端：四模式 + 引擎切换 + 缓冲闸门
    ▼
主服务 server.py (8817, FastAPI)
    │  引擎选择 / 退化重试 / 响度归一 / 流式代理 / 历史管理
    ├──────────────► MLX worker (8818)   stdlib HTTP，mlxrun/voxcpm_stream.py
    ├──────────────► GGUF server (8819)  llama.cpp-omni，llama-tts-server
    └──────────────► PyTorch 进程内     voxcpm 2.0.3，MPS + fp32
```

四个使用模式：基础 TTS、音色设计（括号内写音色描述）、声音克隆（参考音频 + 可选转录）、流式播放。

---

## 快速开始

> **仓库不含模型权重与编译产物**（约 29 GB）。首次运行请先按 [docs/ASSETS.md](docs/ASSETS.md) 重建资产，再回到这里。

```bash
git clone <this-repo> voxcpm-studio && cd voxcpm-studio

# 1. 主环境（Python 3.12，官方要求 <3.13）
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
#    若 voxcpm 装不上，见 docs/deployment-notes.md 踩坑第 2 条
#    （uv 会因构建产物删除被安全策略拦下而整体回滚，改用 pip 即可）

# 2. MLX 环境（独立 venv，不能与主环境混装）
python3.12 -m venv mlxrun/.venv
mlxrun/.venv/bin/python -m pip install -r mlxrun/requirements.txt

# 3. 启动（默认三个引擎全起）
./webapp/start.sh
#    打开 http://127.0.0.1:8817
```

按需裁剪：

```bash
MLX=0  ./webapp/start.sh                 # 只起 PyTorch + GGUF
GGUF=0 ./webapp/start.sh                 # 只起 PyTorch + MLX
GGUF_CHUNK_PATCHES=1 ./webapp/start.sh   # GGUF 逐 patch 解码，用于对照排查（默认 4）
```

命令行单句合成：

```bash
.venv/bin/python tts.py --text "要说的内容" --out outputs/hello.wav
.venv/bin/python tts.py --text "(温和的年轻女声)你好" --out outputs/design.wav
.venv/bin/python tts.py --text "新台词" --ref reference.wav --out outputs/clone.wav
```

注意：`start.sh` 里主服务是**前台**进程，退出时靠 `trap ... EXIT` 联动回收 MLX / GGUF。所以别用 `&` 手动后台化，否则会话结束进程会被带走 —— 要么直接跑脚本，要么用进程托管工具。

---

## 仓库结构

```
voxcpm-studio/
├── webapp/                      Web 主服务 + 单页前端 + 端到端自检
│   ├── server.py                引擎路由、退化重试、响度归一、流式代理
│   ├── start.sh                 一键拉起三进程（EXIT trap 联动回收）
│   ├── static/index.html        单文件前端，零外部依赖
│   ├── verify_*.cjs             Puppeteer 端到端断言脚本
│   └── README.md                接口、参数、验证脚本清单
├── mlxrun/                      MLX 8bit 引擎
│   ├── voxcpm_stream.py         自研真流式 + LiteTokenizer
│   ├── mlx_worker.py            stdlib HTTP worker（8818）
│   ├── verify_stream.py         分块解码等价性验证（SNR / 逐样本）
│   ├── bench_mlx.py             MLX 基准
│   └── BENCH.md                 MLX 侧完整实测记录
├── extensions/llamacpp-omni/
│   └── voxcpm2-perf.patch       对上游 llama.cpp-omni 的改动（打点 + 攒批解码）
├── docs/
│   ├── ASSETS.md                仓库不含的权重/二进制，及重建步骤
│   ├── deployment-notes.md      最初的部署与 5 场景验证记录（含 8 条踩坑）
│   └── shots/                   报告与界面截图
├── tri-route-comparison.html    三路线对比报告（单文件、离线可读）
├── gguf-efficiency.html         GGUF 瓶颈定位与优化报告（单文件）
├── tts.py                       模型 CLI（PyTorch 侧）
├── bench_live.py                服务态基准（首包 / 断供 / RTF）
├── bench_gguf*.py               GGUF 的 CLI 口径与服务态口径基准
├── measure_stream.py            流式分块节奏测量（每块到达时刻与间隔）
├── scan_degenerate.py           低能量样本命中率扫描（GGUF Q8_0 / F16）
├── scan_engines.py              同上，MLX / PyTorch 对照
├── asr_check.py / asr_one.py    ASR 回环 CER 评测（FunASR）
├── monitor_mem.py               libproc 读物理内存（沙箱内 ps 被拦时的替代）
├── requirements.txt             主环境直接依赖（.lock.txt 为全量锁定）
└── mlxrun/requirements.txt      MLX 环境直接依赖
```

---

## 六条工程结论

都是踩过坑、有数字支撑的结论，不是推测。细节在对应文档里。

### 1. mlx-audio 的「流式」是假的，真流式要自己写

`mlx-audio` 的 generate 只在整段合成完后 `yield` 一次。自研 `voxcpm_stream.py` 利用 AudioVAE 解码器是**纯因果卷积**这一点，按块解码、带 16 帧左上下文再裁掉：与整段解码相比 SNR 116.5 dB，**逐样本最大差 0.00e+00**（仅少尾部 8 个卷积余量样本）。首包因此从「整段时长」降到 0.53 s。

### 2. `import transformers` 值 68 秒

MLX worker 启动一度要 68 s，`python -X importtime` 定位到 `import transformers`。改成 Rust `tokenizers` 直接读 `tokenizer.json`（只实现 `tokenize` / `convert_tokens_to_ids`），启动降到 **0.70 s**，9 组用例 token id 完全一致（唯一差异是「文本以空格开头」时的首 token）。

### 3. 内存涨到 48 GB 的根因不是泄漏

同一文本连跑 5 次，活跃对象恒为 3.00 GB —— 说明没有对象泄漏。真因是 **MLX 缓存池无上限**（上限 60.8 GB），而 `/gen` 整段解码的 8.1 GB 峰值被永久记进池里，于是混合长度请求把水位从 8.8 GB 一路推到 23.7 GB。修复：`mx.set_cache_limit(2GB)` + `finally` 里 `clear_cache()` + `/gen` 改为有界分块。25 次浸泡测试稳定在 3.09 GB。代价是 0.039% 样本与精确解码相差 1 LSB（SNR 103 dB），ASR CER 1.22%。

### 4. GGUF 慢在 AudioVAE，不在语言模型

给 runtime 加分段计时后，每 160 ms 音频要 238 ms，拆开是：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| **AudioVAE 解码** | **167 ms** | **70%** |
| CFM 扩散（10 步 + CFG） | 61.5 ms | 26% |
| BaseLM 28 层（Q8_0） | 5.8 ms | 2.4% |
| ResidualLM + FSQ | 3.7 ms | 1.6% |

BaseLM 只占 2.4%，所以 Q8_0 在 Metal 上不提速是必然的 —— 它的价值是省 1.9 GB 内存，不是提速。

已落地的优化是**攒批解码**：连续 4 个 patch 共进一张 VAE 图（首块仍单独出，首包不变），服务态 RTF **1.83 → 1.33**。仍在瓶颈的是 6 层转置卷积走的是朴素 kernel（`kernel_conv_transpose_1d_f32_f32`），算力利用率只有 0.2–2%；要真正跑进实时线以内必须把它改成「相位分解 + GEMM」，属按周计的工作量，本轮未做。

另外三个被实测证伪的假设（别再重复）：建图开销只占 2–3%；im2col 改 F16 走 `mul_mm_f16_f16` 也**仍是 162 ms**；整段解码 4/8/20/32 patch 单 patch 成本恒定 134–140 ms，没有可摊薄的固定开销。

### 5. 低步数扩散会落进「低能量盆地」，且与量化无关

某些「文本 × 噪声种子」组合在 10 步扩散下不收敛，整段退化成气声：RMS 只有正常的 1/8，高频占比从 5.3% 涨到 14.4%。命中率实测（16 个种子）：PyTorch **0/16**、GGUF Q8_0 **1/16**、GGUF F16 **1/16**、MLX **2/16** —— 与量化无关，是两个重实现共同存在的问题。

关键是**放大音量救不回来**：退化样本归一化后 CER 仍是 21.05%，正常样本 7.89%。主服务因此做了两道自动补救：命中阈值就**换种子重生成**（最多 3 次，取最响的一次），再统一**响度归一到 RMS 0.10**。修复后同一案例 CER 回到 7.89%，三引擎输出电平一致。

> 前端默认 seed 固定为 42，所以同一段文本每次都是同一段气声 —— 这是用户侧「重试也没用」的来源。把默认 seed 改成随机也能解决表象，但会牺牲可复现性，所以选择了服务端判定 + 重试。

### 6. 流式断续是结构性欠载，不是调度抖动

播放器恒按 1× 消耗，而 GGUF 的合成速率 r 只有 **0.70**（音频秒 / 墙钟秒）。74 字长句：13 s 音频要 18.5 s 才生成完，缺的 5.8 s 只能靠缓冲垫住。旧代码「收到第一块就播」，实测断供 **20 次、累计静音 5.8 s**。

前端因此改成缓冲闸门，开播条件是 `B ≥ (1 − r) · T + 抖动余量`（r 用第一块之后的音频/耗时实时估算，T 由「字数 → 音频秒」缓存推算）：

| 引擎 | 开播时刻（74 字） | 断供 |
|---|---|---|
| MLX | 0.58 s | 0 |
| PyTorch | 0.88 s | 0 |
| GGUF（顺滑，默认） | 7.6 s | **0** |
| GGUF（低延迟开关） | 0.35 s | 20 次 / 5.8 s |

`r ≥ 0.98` 的引擎闸门退化为 0.12 s，所以 MLX / PyTorch 完全无感。**7.6 s 是物理下限**：想 0.6 s 出声又不断供，只有一个办法 —— 流式用 MLX。

调参时有个坑值得记：只按平均值 `(1−r)T` 开播仍会掉 2 次共 358 ms，因为末尾缓冲恰好耗尽，任何局部速率下探就直接成缺口，所以必须留抖动余量。

---

## 环境变量

**主服务（webapp/server.py）**

| 变量 | 默认 | 作用 |
|---|---|---|
| `VOXCPM_TARGET_RMS` | `0.10` | 响度归一目标（≈ −20 dBFS） |
| `VOXCPM_PEAK_CEIL` | `0.95` | 增益上限，避免削顶 |
| `VOXCPM_DEGEN_RMS` | `0.035` | 低于此 RMS 判为低能量退化 |
| `VOXCPM_DEGEN_PEAK` | `0.22` | 低于此峰值判为低能量退化 |
| `VOXCPM_MAX_ATTEMPTS` | `3` | 换种子重试上限（含首次） |

**GGUF 引擎（由 patch 引入，默认关闭，不影响生产路径）**

| 变量 | 作用 |
|---|---|
| `VOXCPM2_PROF=1` | 每帧打印 LM / CFM / VAE 分段耗时，含 VAE 内部 build/alloc/compute |
| `VOXCPM2_VAE_UNTIL=n` | 解码器只跑前 n 个 block（测每个 block 的边际成本） |
| `VOXCPM2_CHUNK_PATCHES=n` | 每张 VAE 图解码的 patch 数（1=逐帧，4=默认，2 可与逐帧逐样本一致） |
| `VOXCPM2_IM2COL_F16=1` | im2col 输出改 F16（实测无性能收益，仅供 A/B） |

**启动脚本**

| 变量 | 默认 | 作用 |
|---|---|---|
| `MLX=0` / `GGUF=0` | 均启用 | 关闭对应引擎 |
| `GGUF_CHUNK_PATCHES` | `4` | 传给 GGUF server |
| `GGUF_BASE_LM` / `GGUF_ACOUSTIC` | `gguf-models/...` | 覆盖权重路径 |

---

## 已知限制

- **单实例串行**：同一引擎的并发请求会排队。要并发需多实例或换 vLLM-Omni 之类方案。
- **GGUF 流式慢于实时**：每块 160 ms 音频要 240 ms 才到。缓冲闸门能保证不断供，但长句要等 7.6 s 才开播。它真正适合「整段生成 + 一次返回」。
- **流式不做电平补救**：`/api/stream` 边生成边播，无法回头重试。要稳定电平请用非流式（界面默认）。
- **三引擎常驻约 19 GB**：PyTorch 兜底 12.6 GB 是大头。只要 MLX + GGUF 可降到约 6.4 GB，代价是 PyTorch 引擎不可用。
- **仅 macOS + Apple Silicon**：MLX 与 Metal 后端都是 Apple 专有。CUDA 上 PyTorch 的 RTF 约 0.30（4090），本机 MPS 约 0.9。
- **验证覆盖度**：界面与接口有 12 个 Puppeteer 端到端脚本 + ASR 回环，但**没有单元测试**，也没有在除 M2 Max / 64 GB 之外的机器上验证过。

---

## 许可与伦理

- VoxCPM2 模型与代码 Apache-2.0，可商用。
- **声音克隆是高危能力**：几秒音频即可复刻音色，代码层没有任何授权校验。不要克隆未获授权的人声；生成内容建议标注为 AI 合成。
- GGUF 部分的改动基于 llama.cpp-omni 上游实现，以 patch 形式提供而非整树复制，见 [extensions/llamacpp-omni/README.md](extensions/llamacpp-omni/README.md)。
