#!/usr/bin/env python3
"""统计 GGUF 低能量退化样本的出现率：Q8_0 vs F16 BaseLM，多 seed 扫描。"""
import os
import re
import subprocess
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

ROOT = str(paths.ROOT)
BIN = os.environ.get("VOXCPM_GGUF_BIN") or str(paths.GGUF_CLI_BIN)
ACC = str(paths.GGUF_ACOUSTIC)
TEXT = ("包含分段堆叠图、逐 block 边际成本柱图、四个证伪的原始数据、"
        "分层优化表和复现命令。")
SEEDS = list(range(1, 17))
TS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MODELS = {
    "Q8_0": str(paths.GGUF_BASE_LM),
    "F16": str(paths.GGUF_BASE_LM_F16),
}
THRESH = 0.05  # RMS 低于此值视为退化（正常样本在 0.07–0.13）

pat = re.compile(r"RMS=([0-9.]+), peak=([0-9.]+)")

for tag, lm in MODELS.items():
    if not Path(lm).exists():
        # F16 是对照档，默认不保留；缺档跳过，需复跑时按 docs/ASSETS.md 重建
        print(f"[{tag}] 权重不存在，跳过：{lm}\n", flush=True)
        continue
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
