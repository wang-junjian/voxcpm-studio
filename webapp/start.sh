#!/usr/bin/env bash
# 启动 VoxCPM2 Web 语音合成服务
# 用法: ./webapp/start.sh              PyTorch 主服务 + MLX worker + GGUF server（默认都起）
#       MLX=0 ./webapp/start.sh        不起 MLX 引擎
#       GGUF=0 ./webapp/start.sh       不起 GGUF 引擎
#       GGUF_CHUNK_PATCHES=1 ./webapp/start.sh  GGUF 逐 patch 解码（对照/排查用）
#       PORT=9000 ./webapp/start.sh    指定主服务端口
#       HOST=0.0.0.0 ./webapp/start.sh 局域网可访问（手机测试用）
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8817}"
HOST="${HOST:-127.0.0.1}"
MLX_PORT="${MLX_PORT:-8818}"
GGUF_PORT="${GGUF_PORT:-8819}"
MLX_ENABLED="${MLX:-1}"
GGUF_ENABLED="${GGUF:-1}"
PY=".venv/bin/python"
MLX_PY="mlxrun/.venv/bin/python"

GGUF_BIN="llamacpp-omni/llama.cpp-omni-master/build-metal/bin/llama-tts-server"
GGUF_BASE_LM="${GGUF_BASE_LM:-gguf-models/VoxCPM2-BaseLM-Q8_0.gguf}"
GGUF_ACOUSTIC="${GGUF_ACOUSTIC:-gguf-models/VoxCPM2-Acoustic-F16.gguf}"
# AudioVAE 每图的 patch 数。1=逐 patch 解码（首块最小、最慢）；4=攒 4 个一起解
# （实测 RTF 1.6→1.3，首包不变，波形与 1 的差异 63 dB SNR，听不出来）。
# 首块始终单独出，所以首包延迟不受影响。
GGUF_CHUNK_PATCHES="${GGUF_CHUNK_PATCHES:-4}"

MLX_PID=""
GGUF_PID=""

if [ ! -x "$PY" ]; then
  echo "未找到虚拟环境 $PY，请先按 README 部署" >&2
  exit 1
fi

cleanup() {
  for p in "$MLX_PID" "$GGUF_PID"; do
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# MLX worker 跑在独立 venv 里（torch 与 mlx 依赖互斥），主服务按 engine 代理过去
if [ "$MLX_ENABLED" = "1" ] && [ -x "$MLX_PY" ] && [ -d mlxrun/models/VoxCPM2-8bit ]; then
  export VOXCPM_MLX_URL="${VOXCPM_MLX_URL:-http://127.0.0.1:${MLX_PORT}}"
  echo "MLX 8bit worker → ${VOXCPM_MLX_URL}"
  "$MLX_PY" mlxrun/mlx_worker.py --port "$MLX_PORT" --host "$HOST" &
  MLX_PID=$!
else
  echo "MLX worker 未启用（MLX=0 或缺 mlxrun/.venv、mlxrun/models/VoxCPM2-8bit）"
fi

# GGUF 引擎是 llama.cpp-omni 的 llama-tts-server，纯 C++/Metal，自带 venv 无关
if [ "$GGUF_ENABLED" = "1" ] && [ -x "$GGUF_BIN" ] \
   && [ -f "$GGUF_BASE_LM" ] && [ -f "$GGUF_ACOUSTIC" ]; then
  export VOXCPM_GGUF_URL="${VOXCPM_GGUF_URL:-http://127.0.0.1:${GGUF_PORT}}"
  echo "GGUF Q8_0 server → ${VOXCPM_GGUF_URL}（VAE 攒批 ${GGUF_CHUNK_PATCHES} patch/图）"
  VOXCPM2_CHUNK_PATCHES="$GGUF_CHUNK_PATCHES" \
  "$GGUF_BIN" --voxcpm2-base-lm "$GGUF_BASE_LM" --voxcpm2-acoustic "$GGUF_ACOUSTIC" \
    --host "$HOST" --port "$GGUF_PORT" >/tmp/llama-tts-server.log 2>&1 &
  GGUF_PID=$!
else
  echo "GGUF 引擎未启用（GGUF=0 或缺 $GGUF_BIN / gguf-models 下的权重）"
fi

echo "VoxCPM2 Studio → http://${HOST}:${PORT}"
echo "PyTorch 首次加载约 40–90 秒；MLX 约 1 秒；GGUF 约 3 秒（页面会逐个点亮引擎按钮）"
"$PY" webapp/server.py --port "$PORT" --host "$HOST"
