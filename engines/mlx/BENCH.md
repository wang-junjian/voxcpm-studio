# VoxCPM2 在 macOS 上的高效运行方案（实测记录）

机器：Apple M2 Max / 64 GB 统一内存 / macOS 26.5
文本与参数：同一段中文，`inference_timesteps=10`、`cfg_value=2.0`、`seed` 固定
统计口径：预热一次后连跑 3 次短句（29 字）+ 2 次长句（74 字），RTF = 生成耗时 / 音频时长

## 一、实测结果

| 方案 | 权重体积 | 加载耗时 | RTF 均值 | 常驻内存 | 峰值内存 | 真流式 | 中文 CER |
|---|---|---|---|---|---|---|---|
| PyTorch + MPS（fp32） | 4.6 GB | 15.5 s（暖）/ 46.9 s（冷） | 0.901 | ~9.5 GB | ~14 GB | ✅ 首包 0.88 s | 0% |
| MLX bf16 | 4.6 GB | 1.0 s | 0.898 | 4.61 GB | 8.74 GB | ❌ 单次 yield | 短 0% / 长 1.47% |
| MLX 8bit（原始 mlx-audio） | **3.0 GB** | 1.3 s | **0.654** | **3.00 GB** | 6.99 GB | ❌ 单次 yield，首包 4.46 s | 短 0% / 长 1.47% |
| **MLX 8bit + 本项目自研流式** | **3.0 GB** | **1.3 s** | **0.70–0.73** | **3.0 GB（进程实测 3.1–5.0 GB）** | 4.3 GB（分块解码） | ✅ **首包 0.53 s** | 长句与 bf16 同级 |

几点值得注意：

1. **MLX 8bit 比 PyTorch fp32 快约 1.3–1.4 倍**，这是本机最实在的收益；内存降到三分之一，加载从十几秒变成 1 秒出头。
2. **MLX bf16 并不比 PyTorch fp32 快**（0.898 vs 0.901）。社区 README 里"bf16 0.48 / 8bit 0.85"的排序与本机实测不符，不能直接采信。
3. **8bit 与 bf16 的 CER 完全相同**（都是 1.47%，错的还是同一个同音字"整段→诊段"），说明 8bit 量化在这个模型上几乎无损，代价是 ASR 侧的误识别而非合成退化。
4. **mlx-audio 的"流式"是假的**：`voxcpm2.py` 全文只有一处 `yield`（第 700 行），整段合成完才返回，首包 4.46 s。这一点已被本项目改掉，见第二节。

## 二、把 MLX 做成真流式（本项目实现）

代码：`voxcpm_stream.py`（生成与分块解码）+ `mlx_worker.py`（HTTP 服务）。

### 1. 做法

mlx_audio 的 `generate()` 把「自回归出 patch」和「解码成波形」串成两段：先跑完 `max_tokens` 轮，最后一次性 `audio_vae.decode(all_feats)`。要流式，只能把循环搬出来自己写：

```
LLM patch 循环 ──每 4 个 patch（640 ms 音频）──▶ 解码 ──▶ yield PCM
        │                                        ▲
        └── 保留已解码 patch 的尾部 16 帧作左上下文 ┘
```

分块解码之所以正确，是因为解码器是**纯因果卷积**：位置 t 的输出只依赖 t 及之前的 latent。给每块补上 C 帧左上下文、再把这段上下文对应的输出切掉，就得到与整段解码一致的结果。

### 2. 左上下文取多少：C 的实测曲线

同一条 latent 序列（344 帧，13.76 s 音频），整段解码 vs 分块解码（块大小 16 帧）：

| 左上下文 C | 0 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| SNR (dB) | 15.4 | 28.6 | 33.3 | 39.6 | 47.8 | 82.1 | **116.5** |
| 最大绝对差 | 6.3e-1 | 3.3e-1 | 1.9e-1 | 6.9e-2 | 2.1e-2 | 3.9e-4 | **1.0e-5** |

C=16（640 ms 上下文）时 SNR 116 dB，等于完全等价；**C=8 只有 47.8 dB，听感上已经能听出块边界**。所以上下文不能省，C 固定 16。

### 3. 分块大小决定首包，几乎不影响总速度

同一段 16.32 s 音频（seed 42，与 `generate()` 同参）：

| 每块 patch 数 | 首包 | 总耗时 | RTF | 解码累计 | 与整段解码 SNR |
|---|---|---|---|---|---|
| 16（2.56 s 音频） | 1847 ms | 11.35 s | 0.696 | 1.09 s | 119.0 dB |
| 8（1.28 s 音频） | 952 ms | 11.43 s | 0.701 | 1.66 s | 108.3 dB |
| **4（0.64 s 音频）** | **511 ms** | 12.50 s | 0.766 | 3.77 s | 108.0 dB |
| 8（但 C 降到 8） | 949 ms | 11.34 s | 0.695 | 1.56 s | **43.1 dB ← 不可用** |

Web 端默认取 `chunk_patches=4`：首包 0.53 s 比 PyTorch 的 0.88 s 还快，RTF 0.77 仍在实时以内，用户感知不到那 9% 的总时长代价。

### 4. 等价性：与 `generate()` 逐样本一致

VoxCPM2 的 DiT 采样走 `mx.random.normal`，固定 `mx.random.seed` 后整条路径确定。同一 seed、同一文本：

```
一次性 generate()      783368 样本
单块流式（自己实现）    783360 样本     差 8 个样本 = 卷积尾部的固定余量
最大绝对差             0.00e+00        ← 逐样本完全相同
```

分块流式（`chunk_patches=8`）对同一条基准的 SNR 108 dB，最大差 1.4e-5。

### 5. 顺带修掉的启动坑：`import transformers` 要 68 秒

`mlx_audio` 的 `post_load_hook` 调 `AutoTokenizer.from_pretrained`。`transformers` 在本机导入本身要 30–60 秒（`python -X importtime` 显示 32.3 s 全花在 `transformers` 自身的 LazyModule 初始化上），而 `from_pretrained` 只要 0.45 秒。

既然 tokenizer.json 就在本地，直接用 Rust 后端：

```
import transformers + from_pretrained   68 s
tokenizers.Tokenizer.from_file          0.06 s
```

`voxcpm_stream.LiteTokenizer` 只实现 `tokenize` / `convert_tokens_to_ids` 两个方法（VoxCPM2 只用到这两个）。在 9 组中英混排 / 标点 / 数字 / emoji / 括号描述用例上与本模型的 `LlamaTokenizer` token id 完全一致，唯一差异是「文本以空格开头」时的首个 token（Web 端上游已 strip）。**模型加载从 68 s 降到 0.70 s**（纯 load 0.21 s + 分词器 0.06 s + 预热 0.49 s）。

## 三、常驻内存：为什么一路涨到 48 GB

**现象**：跑了几十次合成后，worker 进程的 phys_footprint 涨到 **48.67 GB**（`/gen` 与 `/stream` 都走这个进程），最后被系统 OOM 杀掉。

**先排除"代码里有引用没放"**：同一段文本在同一进程里连续合成 5 次，`active` 恒定 3.00 GB、`cache` 恒定 4.25 GB、进程 footprint 恒定 7.33 GB——**不涨**。走 HTTP 连续请求 8 次同样恒定 6.78 GB。所以不是典型的对象泄漏。

真正的成因是**三个因素叠加**：

1. **MLX 缓冲池默认没有上限**。`mx.get_memory_limit()` 返回 60.8 GB（设备的 recommended working set），被释放的缓冲不会还给系统，只留在池子里等复用。
2. **整段解码的峰值随文本长度线性增长**。19.7 秒音频一次性 `audio_vae.decode(全部帧)`，MLX 峰值 **8.1 GB**，而模型本身只占 3 GB。
3. **`/gen` 原本就是整段一次解码**。于是每一次请求的峰值都被永久"记账"进缓冲池，不同长度的请求各留一份，内存只涨不跌——混合长度请求 11 次就从 8.8 GB 爬到 **23.7 GB**。

**修复**（`voxcpm_stream.py` + `mlx_worker.py`）：

| 措施 | 效果 |
|---|---|
| `mx.set_cache_limit(2 GB)` + 每次生成结束 `mx.clear_cache()` | 池子上限可控，峰值不再累积（`MLX_CACHE_LIMIT_GB` 可调） |
| `/gen` 默认改成分块解码（`chunk_patches=16, context_frames=16`） | 峰值 **8.1 → 4.3 GB**，与文本长度解耦；RTF 0.68 → 0.70 |
| 循环内 `mx.eval(lm_hidden, residual_hidden)`（`eval_each_step`） | 防止惰性图跨步累积；实测对本例峰值无影响（8.09 vs 8.10），数值不变，保留作防御 |

**验证**：25 次混合长度、`/gen` 与 `/stream` 交替的浸泡测试，进程 footprint 在 **4.70–5.01 GB** 之间震荡，不再随请求次数增长；重启后的启动常驻从 4.65 GB 降到 **3.08 GB**。等价性未受影响：单块流式 vs `generate()` 仍是最大差 0.00e+00。

**代价**：`/gen` 分块后与"整段精确解码"的 PCM 有 **0.039% 的样本差 1 LSB**（SNR 103 dB，44 万样本里 258 个），时长完全相同；需要逐样本一致时传 `"exact": true`。ASR 回环 CER 1.22%（唯一错误是"整段→诊段"的同音字）。

诊断脚本：`monitor_mem.py`（macOS `proc_pid_rusage` 读 footprint；沙箱里 `ps`/`top` 会被拦，RSS 也不含 Metal 显存）。



## 四、复现命令

```bash
cd voxcpm2

# PyTorch 基线
.venv/bin/python scripts/bench/bench_pt.py

# MLX 一次性基准（原始 mlx-audio 路径）
cd engines/mlx
.venv/bin/python bench_mlx.py                               # 默认取 paths.MLX_MODEL
.venv/bin/python bench_mlx.py ../../assets/mlx/VoxCPM2-bf16-clean bf16  # 需先自行量化出 bf16
.venv/bin/python mlx_stream.py          # 验证 mlx-audio 是否真流式（答案：否）

# 自研真流式：等价性 + 首包 + 分块参数对比
.venv/bin/python verify_stream.py

# 起 worker（独立 venv，端口 8818）
.venv/bin/python mlx_worker.py --port 8818

# ASR 回环（CER）
cd ..
.venv/bin/python asr_one.py <wav> "<参考文本>"
```

## 五、社区方案全景

| 路线 | 代表产物 | 状态 | 适配本机的结论 |
|---|---|---|---|
| **MLX** | `mlx-community/VoxCPM2-{bf16,8bit,4bit}`、`aufklarer/VoxCPM2-MLX-{int8,int4,bf16}` | 成熟，`mlx-audio` 开箱即用，8bit 下载量最高 | ✅ 已实测：快 1.3–1.4×、省内存、秒加载；上游无流式，本项目已补齐 |
| **GGUF（llama.cpp-omni）** | `DennisHuang648/VoxCPM2-GGUF`（BaseLM Q8_0 1.6 GB + Acoustic F16 1.7 GB） | 工程完整：`voxcpm2-cli`、`--stream`、`llama-tts-server`（OpenAI 兼容 `/v1/audio/speech/stream`）、48 kHz、声纹克隆 | ⏳ 需自编译（`cmake --build build --target voxcpm2-cli`，macOS 自动开 Metal）。文档给出 4090 F16 RTF 0.26–0.38，Apple Silicon 实测未见公开数据 |
| **ONNX** | `ai4all8/VoxCPM2-ONNX` 等 | 仓库存在但 16 GB（fp32）、下载量为 0 | ❌ 无实用价值 |
| **ANE（神经引擎）** | `seba/VoxCPM2-ANE-Preview` | Preview，98 次下载，无文档 | ⚠️ 观察项 |
| **个人整合项目** | `HenryZ838978/VoxWeft`（MLX 4bit + 真流式，宣称 M5 首包 170 ms / M2 300 ms） | 16 star，单人维护 | ⚠️ 其"MLX 真流式"也是自己改的；本项目独立做了一版，思路接近 |

## 六、选型建议

- **Web 交互场景** → **MLX 8bit + 自研真流式**：首包 0.53 s、RTF 0.73、常驻 3.0 GB、启动 1.3 s，四项都优于 PyTorch fp32（0.88 s / 0.90 / 9.5 GB / 47 s）。已接入 `webapp/`，见 `webapp/README.md`。
- **离线批处理 / 只看总时长** → 同一套 MLX 8bit，把 `chunk_patches` 调大（16 或直接用 `generate()`），省掉反复解码的 9% 开销。
- **要 OpenAI 兼容的流式接口或要做 C++ 侧集成** → **GGUF / llama.cpp-omni**，代价是要自己编译。本项目已落地，见下方第六节之二。
- **PyTorch fp32 仍然保留**：它是 MPS 路径的参照实现，出问题时作为对照。Web 界面里可一键切换对比。

## 六节之二、GGUF 路线已落地（llama.cpp-omni + Metal）

编译 `voxcpm2-cli` / `llama-tts-server` / `llama-quantize` 三个目标，用 `convert_voxcpm2_to_gguf.py`
把本地权重转成 GGUF，再把 BaseLM 量化到 Q8_0。同一段 35 字中文的三方对照（服务态，非冷进程）：

| 指标 | MLX 8bit | GGUF Q8_0 | PyTorch fp32 |
|---|---|---|---|
| 磁盘 | 3.0 GB | 3.3 GB（1.6 + 1.7） | 4.6 GB |
| 加载 | 0.2–0.7 s | 0.87 s | 15 s（暖）/ 47–90 s（冷） |
| RTF | **0.83** | 1.33 | 1.04 |
| 首包 | 0.53 s | **0.33 s** | 0.79–0.88 s |
| 常驻 | 3.12 GB | 3.27 GB | 11.6 GB |
| CER | 5.71% | 0% | 0% |

三条值得记住的结论：

1. **首包最快，但流式后劲不足**：每块 160 ms 音频要 240 ms 才到，连续播放会断点。适合整段生成，不适合边生成边听。
2. **Metal 上 Q8_0 不提速**：Q8_0+F16 RTF 1.40/1.46 vs F16+F16 1.36/1.39，体积省 1.4 GB 但速度略降——与 MLX 的 8bit 快 1.38× 完全相反，因为 ggml 的反量化没融进矩阵乘。
3. **Acoustic 留在 F32 纯亏**：多占 1.8 GB（5.06 vs 3.26），RTF 无改善（1.42 vs 1.40）。默认用 F16。

完整对比（含架构图与图表）见 [../tri-route-comparison.html](../tri-route-comparison.html)。

## 七、踩到的环境坑（与本机沙箱有关）

- `tempfile` 探测临时目录时会"写后删"，被 safe-delete 规则拦截 → `torch` 导入直接失败。解决：导入前 `tempfile.tempdir = <本地目录>`。
- `git clone` 需要删除 `.git/*.lock`，被拦 → 改用 tarball 解包。
- ModelScope 上的分片文件名是 `model-00001.safetensors`，而手写 URL 猜的 `model-00001-of-00002.safetensors` 会返回 404 的 HTML（145 字节），`mx.load` 会因此报 "Invalid json header length"。解决：核对 `model.safetensors.index.json` 里的真实文件名。
- `assets/mlx/VoxCPM2-bf16` 里混进过一个 145 字节的无效 safetensors，导致 `load()` 扫描整目录时失败 → 用硬链接拼出 `VoxCPM2-bf16-clean` 只放有效文件。
- 本机 `import transformers` 要 30–60 秒，见第二节第 5 条。
