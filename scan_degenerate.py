#!/usr/bin/env python3
"""统计 GGUF 低能量退化样本的出现率：Q8_0 vs F16 BaseLM，多 seed 扫描。"""
import os
import re
import subprocess
import sys

ROOT = os.environ.get("VOXCPM_ROOT", os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get(
    "VOXCPM_GGUF_BIN",
    f"{ROOT}/llamacpp-omni/llama.cpp-omni-master/build-metal/bin/voxcpm2-cli",
)
ACC = f"{ROOT}/gguf-models/VoxCPM2-Acoustic-F16.gguf"
TEXT = ("包含分段堆叠图、逐 block 边际成本柱图、四个证伪的原始数据、"
        "分层优化表和复现命令。")
SEEDS = list(range(1, 17))
TS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MODELS = {
    "Q8_0": f"{ROOT}/gguf-models/VoxCPM2-BaseLM-Q8_0.gguf",
    "F16": f"{ROOT}/gguf-models/VoxCPM2-BaseLM-F16.gguf",
}
THRESH = 0.05  # RMS 低于此值视为退化（正常样本在 0.07–0.13）

pat = re.compile(r"RMS=([0-9.]+), peak=([0-9.]+)")

for tag, lm in MODELS.items():
    bad = []
    vals = []
    for s in SEEDS:
        out = f"/tmp/rate_{tag}_{s}.wav"
        r = subprocess.run(
            [BIN, "-t", TEXT, "--stream", "--timesteps", str(TS), "--seed", str(s),
             "-o", out, lm, ACC],
            capture_output=True, text=True, cwd=ROOT,
        )
        m = pat.search(r.stdout + r.stderr)
        if not m:
            print(f"  {tag} seed={s} 解析失败")
            continue
        rms, peak = float(m.group(1)), float(m.group(2))
        vals.append(rms)
        flag = ""
        if rms < THRESH:
            flag = "  <== 退化"
            bad.append(s)
        print(f"  {tag} seed={s:2d}  RMS={rms:.4f}  peak={peak:.3f}{flag}", flush=True)
    n = len(vals)
    if n:
        good = [v for v in vals if v >= THRESH]
        avg = sum(good) / len(good) if good else 0
        print(f"[{tag}] ts={TS} 退化 {len(bad)}/{n} ({100*len(bad)/n:.0f}%)  "
              f"坏 seed={bad}  正常样本均值 RMS={avg:.4f}\n", flush=True)
