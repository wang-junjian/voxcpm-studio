#!/usr/bin/env python3
"""对照扫描：同文本同 seed 序列，比较 MLX / PyTorch 是否也会产出低能量样本。"""
import json
import os
import subprocess
import sys

from pathlib import Path

import httpx
import numpy as np
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

BASE = os.environ.get("VOXCPM_BASE", "http://127.0.0.1:8817")
ROOT = os.environ.get("VOXCPM_ROOT") or str(paths.ROOT)
TEXT = ("包含分段堆叠图、逐 block 边际成本柱图、四个证伪的原始数据、"
        "分层优化表和复现命令。")
SEEDS = list(range(1, 17))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
THRESH = 0.05


def rms_of(path):
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768
    return float(np.sqrt((a ** 2).mean())), float(np.abs(a).max())


for engine in ("mlx", "torch"):
    bad, vals = [], []
    for s in SEEDS:
        body = {"text": TEXT, "engine": engine, "steps": STEPS, "seed": s}
        try:
            with httpx.Client(timeout=None, trust_env=False) as c:
                d = c.post(f"{BASE}/api/tts", json=body).json()
        except Exception as e:
            print(f"  {engine} seed={s} 请求失败 {e}")
            continue
        if "id" not in d:
            print(f"  {engine} seed={s} 失败 {d}")
            continue
        rms, peak = rms_of(f"{ROOT}/webapp/runs/{d['id']}.wav")
        vals.append(rms)
        flag = ""
        if rms < THRESH:
            flag = "  <== 退化"
            bad.append(s)
        print(f"  {engine} seed={s:2d}  RMS={rms:.4f}  peak={peak:.3f}{flag}", flush=True)
    n = len(vals)
    if n:
        good = [v for v in vals if v >= THRESH]
        print(f"[{engine}] ts={STEPS} 退化 {len(bad)}/{n} ({100*len(bad)/n:.0f}%)  "
              f"坏 seed={bad}  正常均值 RMS={sum(good)/len(good) if good else 0:.4f}\n", flush=True)
