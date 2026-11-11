"""VoxCPM2 本地部署冒烟测试：加载 + 单句合成。"""
import os, sys, time, json
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

MODEL_DIR = str(paths.PYTORCH_MODEL)
OUT_DIR = str(paths.ensure_dirs())

DEVICE = os.environ.get("VOXCPM_DEVICE", "mps")
t_load = time.time()
print(f"[1/3] 加载模型 device={DEVICE} path={MODEL_DIR}", flush=True)

from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    MODEL_DIR,
    load_denoiser=False,   # 跳过 ModelScope 降噪模型，避免额外下载
    device=DEVICE,
    optimize=False,        # MPS 上不做 torch.compile
)
load_s = time.time() - t_load
print(f"[1/3] 加载完成 {load_s:.1f}s | sample_rate={model.tts_model.sample_rate}", flush=True)

text = "你好，这是 VoxCPM 二在本地 Mac 上的部署验证，采样率四十八千赫兹。"
print(f"[2/3] 合成: {text}", flush=True)
t0 = time.time()
wav = model.generate(
    text=text,
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
gen_s = time.time() - t0

sr = model.tts_model.sample_rate
wav = np.asarray(wav, dtype=np.float32).reshape(-1)
dur = len(wav) / sr
path = os.path.join(OUT_DIR, "smoke_zh.wav")
sf.write(path, wav, sr)
print(f"[2/3] 生成 {dur:.2f}s 音频，耗时 {gen_s:.2f}s，RTF={gen_s/max(dur,1e-9):.3f}", flush=True)

rms = float(np.sqrt(np.mean(wav ** 2)))
peak = float(np.max(np.abs(wav)))
print(f"[3/3] 波形: dtype={wav.dtype} n={len(wav)} sr={sr} rms={rms:.4f} peak={peak:.4f} 静音比={float(np.mean(np.abs(wav)<1e-4)):.3f}", flush=True)
print(json.dumps({"load_s": load_s, "gen_s": gen_s, "duration": dur, "sr": sr, "rms": rms, "peak": peak, "path": path}, ensure_ascii=False))
print("SMOKE_OK", flush=True)
