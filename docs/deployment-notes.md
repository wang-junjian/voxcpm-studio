# 部署与验证记录（PyTorch 路线，最初一版）

> 这是项目最初阶段的完整记录：环境搭建、5 个场景的实证验证、8 条踩坑。
> 项目主页见 [../README.md](../README.md)，三路线对比见 [../tri-route-comparison.html](../tri-route-comparison.html)。

OpenBMB VoxCPM2（2B 无分词器 TTS）在 Apple Silicon 上的离线部署与验证。

## 一、结论先说

| 项目 | 结果 |
|---|---|
| 部署状态 | ✅ 可用，完全离线（权重已落盘，推理不联网） |
| 推理设备 | MPS（Apple GPU），自动降级 float32 |
| 模型加载 | 8.7 s（暖启动）/ 48.7 s（首次冷读盘） |
| 生成速度 | RTF 0.85–1.51（本机实测，详见下表） |
| 峰值内存 | 14.42 GB / 64 GB |
| 输出格式 | 48 kHz 单声道 PCM16 WAV |
| 语音准确度 | 5 段音频中 4 段 ASR 回环 CER = 0%，1 段差 1 字符（ASR 自身误识） |
| 核心能力 | 基础 TTS / 音色设计 / 声音克隆 / 流式输出 全部跑通 |

**一句话**：2B 模型在 M2 Max 上以 float32 跑在 MPS，速度接近实时（RTF≈0.9），长文本 60 字逐字全对，可以当本地 TTS 用。

## 二、环境

```
机器      Apple M2 Max (arm64), 12 核 CPU, 64 GB 统一内存
系统      macOS 26.5.2
Python    3.12.13（虚拟环境；系统默认是 3.13，不能用）
PyTorch   2.15.0 (MPS available)
voxcpm    2.0.3
transformers 5.22.2 / numpy 2.5.4 / funasr 1.4.24
```

## 三、部署步骤

```bash
cd voxcpm-studio

# 1. 建 3.12 环境（官方要求 Python ≥3.10 <3.13）
uv venv --python 3.12 .venv

# 2. 装 PyTorch（先单独装，避免和 voxcpm 一起装被大包拖垮）
uv pip install --python .venv/bin/python torch torchaudio

# 3. 装 voxcpm（必须用 pip，原因见"踩坑记录"第 2 条）
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install voxcpm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 下权重（4.62 GB，走 ModelScope，HF 直连大文件不通）
#    文件清单：model.safetensors / audiovae.pth / config.json /
#              tokenizer.json / tokenizer_config.json /
#              special_tokens_map.json / tokenization_voxcpm2.py
for f in model.safetensors audiovae.pth config.json tokenizer.json \
         tokenizer_config.json special_tokens_map.json tokenization_voxcpm2.py; do
  curl -sL -C - -o models/VoxCPM2/$f \
    "https://modelscope.cn/models/OpenBMB/VoxCPM2/resolve/master/$f"
done
```

权重校验：`model.safetensors` 应为 4,580,080,592 字节，头部为 BF16 张量表。

## 四、验证结果

### 1. 五种能力实测

| 场景 | 文本 | 音频时长 | 生成耗时 | RTF | ASR CER |
|---|---|---|---|---|---|
| A 基础中文 | 22 字，含 "VoxCPM2"/"Mac" | 4.32 s | 6.74 s | 1.56 | 2.94%* |
| B 音色设计 | 括号描述 + 中文 | 3.04 s | 2.96 s | 0.97 | **0%** |
| C 长文本 | 60 字 | 15.52 s | 14.17 s | 0.91 | **0%** |
| D 流式 | 33 字 | 7.52 s | 7.23 s | 0.96 | **0%** |
| E 声音克隆 | 以 A 的音频为参考 | 4.64 s | 7.02 s | 1.51 | **0%** |
| F 命令行入口 | 27 字 | 5.73 s | 6.72 s | 0.85 | — |

\* A 段的 2.94% 来自 1 个字符：合成是正确的，但 FunASR 把 "Vox" 听成了 "voc"。归一化（去标点/空格、统一小写、数字转中文）后的平均 CER 为 **0.98%**。

平均 RTF 0.9 左右意味着**生成 1 秒语音约需 0.9 秒**，基本跟得上实时朗读；A 段偏高是因为第一次推理要付 MPS kernel 预热成本。

### 2. 音频格式核验（ffprobe）

```
codec_name=pcm_s16le  sample_rate=48000  channels=1  bits_per_sample=16
```

无削波（`clipped_ratio=0`）、静音比 2%–9%（正常语句停顿）、峰值 0.51–0.92 未触顶。

### 3. 流式

47 个音频块，**首包延迟 0.88 s**（整段 7.5 s 音频），适合边生成边播放。

⚠️ 流式模式下 `retry_badcase` 被强制关闭（模型内部警告），质量兜底能力弱于整段模式。

### 4. 可视化

`outputs/verification_overview.png` —— 5 段音频波形对照图。

## 五、踩坑记录（都踩过了，可直接绕过）

1. **Python 版本卡死**：系统 Python 是 3.13，voxcpm 要求 `<3.13`，必须单独建 3.12 环境。

2. **uv 装不了 voxcpm**：`antlr4-python3-runtime==4.9.3` 只有 sdist 需要现场构建，构建收尾时删除 `build/bdist.../wheel` 目录被本机 safe-delete hook 拦下（`SAFE_DELETE_BULK_CONFIRM_REQUIRED`），uv 事务整体回滚，install 静默失败（`exit=0` 但 site-packages 是空的）。**改用 pip 装即可**——pip 内部走 Python API，不触发 shell 层拦截。

3. **HuggingFace 大文件下不动**：API 元数据能拿到（200），但 `resolve` 下大文件速度 0 B/s。ModelScope 实测 9.6 MB/s，4.6 GB 约 8 分钟。

4. **README 与 API 不一致**：官方 README 的示例带 `seed=42`，但 2.0.3 版 `_generate()` 没有 `seed` 参数，直接报 `TypeError`。可复现性用 `torch.manual_seed()` 替代。

5. **MPS 必须 float32**：checkpoint 是 BF16，源码里 `pick_runtime_dtype()` 在 MPS 上强制降为 float32，日志会打印 `[voxcpm2] adjusted dtype bfloat16 -> float32 for device mps`。原因写在代码注释里：bf16/fp16 在扩散自回归循环里数值漂移足够大，输出会崩坏并触发 badcase 检测无限重试。代价是显存/内存翻倍（约 9.2 GB 权重），本机 14.4 GB 峰值可接受。可用 `VOXCPM_MPS_DTYPE` 强改，但不建议。

6. **`optimize=True` 在 MPS 上不值得**：默认会 `torch.compile`，MPS 后端收益小、启动开销大。全部用例都用 `optimize=False`。

7. **`load_denoiser=False`**：默认会去 ModelScope 拉 ZipEnhancer 降噪模型（`iic/speech_zipenhancer_ans_multioloss_16k_base`）。纯 TTS 用不到，关掉省一次下载。

8. 关于官方 issue #337「VoxCPM2 暂不支持 Mac MPS 加速」：当前 2.0.3 代码里 MPS 路径是打通的（含 RoPE 的非连续张量 workaround），实测可用；该 issue 大概率指的是旧版本或指性能不及 CUDA。

## 六、日常用法

```bash
# 基础合成
.venv/bin/python tts.py --text "要说的内容" --out outputs/hello.wav

# 音色设计（括号内写描述）
.venv/bin/python tts.py --text "(A warm young female voice)你好" --out outputs/design.wav

# 声音克隆
.venv/bin/python tts.py --text "新台词" --ref reference.wav --out outputs/clone.wav

# 极致克隆（参考音频 + 其转录文本，相似度更高）
.venv/bin/python tts.py --text "新台词" --ref ref.wav --prompt-text "ref 的原话" --out outputs/clone2.wav

# 用 CPU 跑（慢很多，但内存占用低）
.venv/bin/python tts.py --text "..." --device cpu --out outputs/cpu.wav
```

Python 侧：

```python
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("models/VoxCPM2", load_denoiser=False,
                               device="mps", optimize=False)
wav = model.generate(text="你好", cfg_value=2.0, inference_timesteps=10)
# wav 是 numpy float32 一维数组，采样率 model.tts_model.sample_rate (48000)
```

## 七、目录

当时的工作目录（只有 PyTorch 路线，尚无 mlxrun / gguf）：

```
voxcpm-studio/
├── models/VoxCPM2/          权重 4.62 GB（不在仓库里，见 ASSETS.md）
├── .venv/                   Python 3.12 环境
├── outputs/                 验证音频 + report.json + asr_check.json + 波形图
├── tts.py                   本地推理 CLI（推荐入口）
├── verify_tts.py            三场景验证脚本
├── asr_check.py             ASR 回环 CER 验证
├── test_extra.py            流式 + 克隆验证
├── make_figure.py           波形总览图
└── smoke_test.py            最小冒烟测试
```

后续加入 MLX / GGUF 两条路线后的完整结构见 [../README.md](../README.md)。

## 八、注意事项

- 模型许可证 Apache-2.0，代码与权重均可商用。
- **声音克隆是高危能力**：几秒音频即可复刻音色，代码层没有任何授权校验。不要克隆未获授权的人声，生成内容按需标注为 AI 合成。
- 不支持 macOS 之外的设备时性能差异很大：CUDA 上 RTF 约 0.30（4090），本机 MPS 约 0.9，纯 CPU 会慢一个数量级。

## 九、Web 语音合成应用

`webapp/` 下是基于本模型的浏览器界面：模型常驻内存，支持基础 TTS / 音色设计 / 声音克隆 / 流式播放四种模式，并可在 **MLX 8bit**、**GGUF Q8_0**、**PyTorch fp32** 三个推理引擎之间一键切换。

```bash
./webapp/start.sh          # 打开 http://127.0.0.1:8817（主服务 + MLX worker + GGUF server）
MLX=0  ./webapp/start.sh   # 不起 MLX 引擎
GGUF=0 ./webapp/start.sh   # 不起 GGUF 引擎
GGUF_CHUNK_PATCHES=1 ./webapp/start.sh   # GGUF 逐 patch 解码（对照/排查用，默认 4）
```

三条推理路线的完整实测对比（架构图、性能图表、五个发现、复现命令）见
[**tri-route-comparison.html**](../tri-route-comparison.html)。一句话：
**MLX 综合最优（RTF 0.83，内存 3.12 GB）；GGUF 首包最快（331 ms）、内存 3.27 GB、CER 0%，但整体慢 1.6 倍且流式节奏慢于实时；PyTorch 最稳但最重（11.6 GB）。**

GGUF 为什么慢、慢在哪一层、还能优化多少，见
[**gguf-efficiency.html**](../gguf-efficiency.html)：每 160 ms 音频的 238 ms 里，AudioVAE 解码独占 167 ms（70%），
BaseLM 只占 5.8 ms。已落地的改动是把连续 4 个 patch 攒进同一张 VAE 图解码（首块仍单独出、首包不变），
服务态 RTF 从 1.6–1.8 降到 1.29–1.34；再往下必须改 `voxcpm2_audiovae.cpp` 里的转置卷积实现。

`mlxrun/` 是 MLX 侧的实现与实测记录：8bit 量化权重（3.0 GB）、自研真流式（首包 0.53 s），详见 [mlxrun/BENCH.md](../mlxrun/BENCH.md)。
`llamacpp-omni/` 是 GGUF 侧：llama.cpp-omni 的 Metal 构建（`voxcpm2-cli` / `llama-tts-server` / `llama-quantize`），权重转换与量化产物在 `gguf-models/`。

GGUF 侧新增的诊断开关（默认关闭，不影响生产路径）：

| 环境变量 | 作用 |
|---|---|
| `VOXCPM2_PROF=1` | 每帧打印 LM / CFM / VAE 分段耗时，含 VAE 内部 build/alloc/compute |
| `VOXCPM2_VAE_UNTIL=n` | 解码器只跑前 n 个 block（测每个 block 边际成本） |
| `VOXCPM2_CHUNK_PATCHES=n` | 每次 VAE 图解码的 patch 数（1=逐帧，4=默认，2 可与逐帧逐样本一致） |
| `VOXCPM2_IM2COL_F16=1` | im2col 输出改 F16（实测无性能收益，仅供 A/B） |

### 电平为什么会「突然很小」

模型在低扩散步数（默认 10 步）下，某些「文本 × 噪声种子」组合会落进低能量盆地：
整段退化成气声，RMS 只有正常样本的 1/8、峰值 1/5，听感就是**声音非常小**。

实测（同一文本 × 16 个种子，`inference_timesteps=10`）：

| 实现 | 命中低能量盆地 | 说明 |
|---|---|---|
| PyTorch（参考实现） | 0 / 16 | 未命中 |
| GGUF Q8_0 | 1 / 16 | 与量化无关 |
| GGUF F16 | 1 / 16 | 命中率与 Q8_0 相同 |
| MLX 8bit | 2 / 16 | 同样存在 |

这说明问题在**低步数下扩散未收敛**，不是量化、也不是攒批（`CHUNK_PATCHES=1` 与 `4`
输出逐位相同）。同一段文本把 `timesteps` 从 10 加到 30/40，RMS 会从 0.013 单调回到 0.092。
退化样本放大音量后 CER 仍是 21%（正常样本 7.9%），所以**只放大没用，必须换种子重生成**。

主服务因此在 `/api/tts` 上做了两道自动补救，前端结果栏会如实标注：

1. **换种子重试**：生成后测 RMS/峰值，命中退化阈值就换 `seed` 重生成，最多 3 次，取最响的一次；
2. **响度归一**：正常样本统一到 RMS 0.10（≈ −20 dBFS），峰值上限 0.95，消除同一引擎不同文本间
   RMS 0.04–0.25 的波动。三个引擎现在输出电平一致。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `VOXCPM_TARGET_RMS` | `0.10` | 响度归一的目档（≈ −20 dBFS） |
| `VOXCPM_PEAK_CEIL` | `0.95` | 增益上限，避免削顶 |
| `VOXCPM_DEGEN_RMS` | `0.035` | 低于此 RMS 判为退化 |
| `VOXCPM_DEGEN_PEAK` | `0.22` | 低于此峰值判为退化 |
| `VOXCPM_MAX_ATTEMPTS` | `3` | 最多尝试次数（含首次） |

注意：**流式播放（`/api/stream`）不做重试与归一**——音频边生成边播，无法回头重来。
需要稳定电平请用非流式的「生成」按钮（界面默认就是非流式）。

### 流式播放的缓冲闸门

前端不再「收到第一块就播」。播放器恒按 1× 消耗，而合成速率 r（音频秒/墙钟秒）在 GGUF 上只有 0.70，
开播时缓冲必须 ≥ `(1 - r) · T + 抖动余量`，否则中途必然断供（实测 74 字长句断 20 次、累计静音 5.8 s）。
r 与「字数→音频秒」都由上一段流实测后缓存，`r ≥ 0.98` 的引擎（MLX 1.39 / PyTorch 1.04）闸门退化为 0.12 s。

| 引擎 | 开播时刻（74 字） | 断供 |
|---|---|---|
| MLX | 0.58 s | 0 |
| PyTorch | 0.88 s | 0 |
| GGUF（顺滑，默认） | 7.6 s | 0 |
| GGUF（低延迟开关） | 0.35 s | 20 次 / 5.8 s |

界面「低延迟优先」开关可切回旧行为（记忆选择）。详见 [webapp/README.md](../webapp/README.md)。

