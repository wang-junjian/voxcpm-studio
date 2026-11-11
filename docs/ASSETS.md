# 仓库不含的资产，以及怎么重建

这个仓库只放**代码、文档与报告**，不放权重、虚拟环境与编译产物。原因很直接：完整工作目录约 **14 GB**，其中绝大部分是可重建的资产，塞不进 git，也没有必要——它们都能从公开源重建。

**权重统一收在 `assets/` 下，虚拟环境与上游构建产物各自留在原位**。也就是说：克隆下来只有几十 MB，跑起来需要自己往里填；填完之后 `git status` 依然干净。

| 资产 | 目标路径 | 体积 | 来源 |
|---|---|---|---|
| PyTorch 权重 | `assets/pytorch/VoxCPM2/` | 4.6 GB | ModelScope `OpenBMB/VoxCPM2` |
| MLX 8bit 权重 | `assets/mlx/VoxCPM2-8bit/` | 3.0 GB | HF `mlx-community/VoxCPM2-8bit` |
| GGUF 权重 | `assets/gguf/` | 3.3 GB（生产两档） | 由 PyTorch 权重转换 + 量化 |
| llama.cpp-omni 源码 + 构建 | `llamacpp-omni/` | 70 MB | 上游快照 + 本仓库 patch + cmake |
| 主环境虚拟环境 | `.venv/` | 2.0 GB | `uv venv` + `requirements.txt`（149 包） |
| MLX 虚拟环境 | `engines/mlx/.venv/` | 584 MB | `python -m venv` + `engines/mlx/requirements.txt` |

> **为什么权重集中、而 venv 与构建产物不集中**：权重是被多个引擎共享的纯数据，集中后换盘、备份、算体积都只对一个目录；虚拟环境和 CMake 构建产物则含绝对路径（`bin/` 下 console script 的 shebang、CMake 缓存、Metal shader 的相对查找），挪动就会失效，收益为负。
>
> 所有路径由项目根的 [paths.py](../paths.py) 统一解析，默认相对该文件定位，所以在任何工作目录下运行一致。权重换位置只改这一个文件，或用环境变量（`VOXCPM_ASSETS` 等）覆盖，不必改代码。

---

## 0. 两个虚拟环境（先建这个，后面几步的脚本都要用）

```bash
# 主环境：torch + voxcpm + fastapi，2.0 GB
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# MLX 环境：独立 venv，不能与主环境混装（torch 与 mlx 依赖互斥）
python3.12 -m venv engines/mlx/.venv
engines/mlx/.venv/bin/python -m pip install -r engines/mlx/requirements.txt
```

两个 venv 都直接用 `.venv/bin/python` 这样的**相对路径**调用，不 `source activate`，所以整个项目目录可以任意移动位置——只有一个例外：`bin/` 下 console script 的 shebang 是绝对路径，移动后 `pip` 之类的入口会失效（`python` 本身不受影响）。需要时批量修正：

```bash
OLD=/旧/项目/路径; NEW=$(pwd)
grep -rl "$OLD" .venv/bin engines/mlx/.venv/bin | xargs sed -i '' "s|$OLD|$NEW|g"
```

## 1. PyTorch 权重（所有路线的共同底座）

```bash
mkdir -p assets/pytorch/VoxCPM2
for f in model.safetensors audiovae.pth config.json tokenizer.json \
         tokenizer_config.json special_tokens_map.json tokenization_voxcpm2.py; do
  curl -sL -C - -o assets/pytorch/VoxCPM2/$f \
    "https://modelscope.cn/models/OpenBMB/VoxCPM2/resolve/master/$f"
done
```

- 校验：`model.safetensors` 应为 **4,580,080,592** 字节。
- 为什么不用 HuggingFace：API 元数据能拿到，但 `resolve` 下大文件速度 0 B/s。ModelScope 实测 9.6 MB/s，4.6 GB 约 8 分钟。

## 2. MLX 8bit 权重（可选，但要 MLX 引擎就必需）

`mlx-audio` 的加载器直接吃 MLX 格式目录（含 `config.json` / `model.safetensors` / `tokenizer.json`）：

```bash
# 用 huggingface-cli 或 hf 下载整目录
hf download mlx-community/VoxCPM2-8bit --local-dir assets/mlx/VoxCPM2-8bit
```

也可以自己从 bf16 量化（`mlx_audio` 自带 convert 工具），但本机实测口径用的是社区 8bit 转换：3.0 GB，与 bf16 的 CER 完全相同（1.47%），说明这个模型上 8bit 几乎无损。

> 自己量化会额外在 `assets/mlx/` 下留 `VoxCPM2-bf16/`（4.6 GB）与 `VoxCPM2-bf16-clean/`，两者是**中间产物，跑推理不需要**，量化完即可删。注意后者是前者的硬链接（同一份 inode，链接数 2），删一个不省空间，必须两个一起删。当前仓库已清理，`assets/mlx/` 只剩 8bit 的 3.0 GB。
>
> 注意 `engines/mlx/` 有独立 venv。主环境的 torch 与 mlx 不要混装，原因见 [engines/mlx/requirements.txt](../engines/mlx/requirements.txt) 顶部注释。

## 3. llama.cpp-omni 源码与构建（GGUF 引擎）

```bash
# 3.1 取得上游源码树（本项目基于 2026-10-30 的 master 快照）
#     解压/克隆到 llamacpp-omni/llama.cpp-omni-master
#     ⚠ 目录别直接叫 gguf —— 会遮蔽 pip 的 gguf 命名空间包，导致转换脚本 import 失败

# 3.2 应用本仓库的改动（性能打点 + 攒批解码，共 168 行新增 / 22 行删除）
cd llamacpp-omni/llama.cpp-omni-master
patch -p1 < ../../../extensions/llamacpp-omni/voxcpm2-perf.patch

# 3.3 构建（Metal 后端）
cmake -B build-metal -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal -j8 --target voxcpm2-cli llama-tts-server llama-quantize
```

产物：`build-metal/bin/{voxcpm2-cli, llama-tts-server, llama-quantize}`。

> 上游解压出来是 304 MB，其中只有约 70 MB 与 GGUF 路线有关。可以直接删掉这四块：`models/`（上游 LLM 词表测试件，75 MB）、`docs/`（文档站图片，30 MB）、`build/`（另一套非 Metal 构建，23 MB）、`build-metal/` 下除 `bin/` 以外的编译中间产物（106 MB）。`bin/` 里的可执行全部走 `@loader_path` 找 dylib、Metal shader 也内嵌在 `libggml-metal.dylib`，所以那 10 MB 是自包含的。
>
> 重新编译时 `cmake -B build-metal` 会重新 configure，删掉缓存与 `.o` 不影响上面这套流程。

> 构建中途不要重命名源码目录 —— CMake 缓存里存的是绝对路径，改名会让缓存整体失效，只能删掉 build 目录重新配置。同理，这棵树**不要**搬进 `assets/`。

## 4. GGUF 权重（由第 1 步的权重转换而来）

```bash
# 4.1 转 GGUF（默认 f16；量化输入建议用 --dtype f32）
python llamacpp-omni/llama.cpp-omni-master/tools/omni/voxcpm2/convert_voxcpm2_to_gguf.py \
  --model assets/pytorch/VoxCPM2/model.safetensors \
  --vae assets/pytorch/VoxCPM2/audiovae.pth \
  --config assets/pytorch/VoxCPM2/config.json \
  --output assets/gguf --dtype f16
# 产出 VoxCPM2-BaseLM-F16.gguf（3.0 GB）+ VoxCPM2-Acoustic-F16.gguf（1.7 GB）

# 4.2 把 BaseLM 量化到 Q8_0
llamacpp-omni/llama.cpp-omni-master/build-metal/bin/llama-quantize \
  assets/gguf/VoxCPM2-BaseLM-F16.gguf \
  assets/gguf/VoxCPM2-BaseLM-Q8_0.gguf Q8_0
```

**生产只需要两个文件**（合计 3.3 GB）：

```
assets/gguf/VoxCPM2-BaseLM-Q8_0.gguf     1.6 GB
assets/gguf/VoxCPM2-Acoustic-F16.gguf    1.7 GB
```

其余三档（`BaseLM-F16` 3.0 GB、`BaseLM-F32` 6.0 GB、`Acoustic-F32` 3.4 GB）是**量化中间产物兼对照档**，当前仓库已清理（连同 MLX 的 bf16 目录共回收 17 GB）。要复跑基准或换量化档时，按上面 4.1/4.2 从 `assets/pytorch/` 重新导出即可；`scripts/bench/bench_gguf.py` 与 `scripts/eval/scan_degenerate.py` 遇到缺档会自动跳过，不会报错。

Acoustic 保持 F16 是有实测依据的：它承载 AudioVAE，量化收益为零而风险全在音质上。BaseLM 量化到 Q8_0 也**不提速**（Metal 上 1.40 vs F16 的 1.36），只省 1.4 GB —— 真正占时的部分是 VAE 解码，见 [../gguf-efficiency.html](../gguf-efficiency.html)。

想把 `assets/gguf/` 控制在 3.3 GB 以内，删掉这两个之外的 F32 / F16 BaseLM 即可（共 13.4 GB）。

## 5. 运行时目录（脚本会自动创建）

| 目录 | 用途 |
|---|---|
| `webapp/runs/` | 每次合成的 wav + 同名 json 元信息 |
| `webapp/refs/` | 声音克隆上传的参考音频 |
| `engines/mlx/logs/` | MLX 侧的基准与验证日志 |
| `outputs/` | `tts.py` 与 `scripts/` 下各脚本的输出 |
| `tmpdir/` | 钉死的临时目录（见 `paths.py` 的 `pin_tempdir`） |

这些目录都在 `.gitignore` 里，`paths.ensure_dirs()` 会在首次运行时创建。
