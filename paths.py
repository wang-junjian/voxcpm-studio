"""VoxCPM Studio 的路径唯一真源。

所有脚本都从这里取路径，不再各自拼字符串。这样目录改名、权重挪位置只需要
改这一个文件，也不会再出现「同一个 models/ 前缀在两个脚本里指向不同目录」
的情况。

默认值全部相对本文件解析（`Path(__file__).resolve()`），所以在任何工作目录
下运行结果都一致。

覆盖优先级：环境变量 > 默认值。支持的环境变量见 README「环境变量」一节。
注意：本模块不能叫 `voxcpm`，那会遮蔽 pip 安装的同名模型库。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# ── 根与资产 ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("VOXCPM_ASSETS") or ROOT / "assets")

# ── 三条推理路线各自的权重 ──────────────────────────────────────────────
PYTORCH_MODEL = Path(os.environ.get("VOXCPM_PT_MODEL") or ASSETS / "pytorch" / "VoxCPM2")
MLX_MODEL = Path(os.environ.get("VOXCPM_MLX_MODEL") or ASSETS / "mlx" / "VoxCPM2-8bit")

GGUF_DIR = Path(os.environ.get("VOXCPM_GGUF_DIR") or ASSETS / "gguf")
# 生产只用 Q8_0 + Acoustic-F16；其余 F16/F32 是转换与对照用的中间档
GGUF_BASE_LM = Path(os.environ.get("GGUF_BASE_LM") or GGUF_DIR / "VoxCPM2-BaseLM-Q8_0.gguf")
GGUF_ACOUSTIC = Path(os.environ.get("GGUF_ACOUSTIC") or GGUF_DIR / "VoxCPM2-Acoustic-F16.gguf")
GGUF_BASE_LM_F16 = GGUF_DIR / "VoxCPM2-BaseLM-F16.gguf"
GGUF_ACOUSTIC_F32 = GGUF_DIR / "VoxCPM2-Acoustic-F32.gguf"

# ── llama.cpp-omni 上游树与 Metal 构建产物 ──────────────────────────────
# 刻意不搬进 assets/：CMake 构建出的可执行依赖同目录/上级的 dylib 与 Metal
# shader，整体迁移会破坏查找路径。上游树按 third_party 语义留在原位。
LLAMACPP_ROOT = Path(
    os.environ.get("VOXCPM_LLAMACPP") or ROOT / "llamacpp-omni" / "llama.cpp-omni-master"
)
LLAMACPP_BIN = LLAMACPP_ROOT / "build-metal" / "bin"
GGUF_SERVER_BIN = LLAMACPP_BIN / "llama-tts-server"
GGUF_CLI_BIN = LLAMACPP_BIN / "voxcpm2-cli"
GGUF_QUANTIZE_BIN = LLAMACPP_BIN / "llama-quantize"

# ── 引擎与服务目录 ──────────────────────────────────────────────────────
MLX_ENGINE = ROOT / "engines" / "mlx"
WEBAPP_DIR = ROOT / "webapp"
STATIC_DIR = WEBAPP_DIR / "static"
SCRIPTS_DIR = ROOT / "scripts"

# ── 运行时产物（不进 git）───────────────────────────────────────────────
OUTPUTS = Path(os.environ.get("VOXCPM_OUTPUTS") or ROOT / "outputs")
TMPDIR = Path(os.environ.get("VOXCPM_TMPDIR") or ROOT / "tmpdir")


def require(path: Path, what: str) -> Path:
    """确认资产存在，不存在时给出可执行的修复指引而不是裸 FileNotFoundError。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} 不存在：{p}\n  重建方式见 docs/ASSETS.md")
    return p


def ensure_dirs() -> Path:
    """建好输出目录并返回它。写文件前调用，避免首次运行直接崩在 open()。"""
    for d in (OUTPUTS, TMPDIR):
        d.mkdir(parents=True, exist_ok=True)
    return OUTPUTS


def pin_tempdir() -> Path:
    """把 tempfile 钉到项目内 tmpdir。

    某些受限环境下 tempfile 的「写入后删除」探测会失败，torch / dill 会因此
    在 import 阶段就抛错。必须在导入这些库之前调用。
    """
    TMPDIR.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(TMPDIR)
    return TMPDIR


if __name__ == "__main__":
    # 资产必须已存在（缺了要按 docs/ASSETS.md 重建）；运行时目录可自动创建。
    assets = [
        ("ASSETS", ASSETS),
        ("PYTORCH_MODEL", PYTORCH_MODEL),
        ("MLX_MODEL", MLX_MODEL),
        ("GGUF_DIR", GGUF_DIR),
        ("GGUF_BASE_LM", GGUF_BASE_LM),
        ("GGUF_ACOUSTIC", GGUF_ACOUSTIC),
        ("GGUF_SERVER_BIN", GGUF_SERVER_BIN),
        ("GGUF_CLI_BIN", GGUF_CLI_BIN),
        ("MLX_ENGINE", MLX_ENGINE),
    ]
    runtime = [("OUTPUTS", OUTPUTS), ("TMPDIR", TMPDIR)]

    print(f"ROOT  {ROOT}")
    for name, p in assets:
        mark = "ok  " if Path(p).exists() else "MISS"
        print(f"[{mark}] {name:16s} {p}")
    for name, p in runtime:
        mark = "ok  " if Path(p).exists() else "auto"
        print(f"[{mark}] {name:16s} {p}")

    missing = [n for n, p in assets if not Path(p).exists()]
    raise SystemExit(1 if missing else 0)
