"""生成验证结果总览图：5 段音频波形 + 关键指标。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "outputs"
items = [
    ("A_basic_zh", "A 基础中文 TTS", 1),
    ("B_voice_design", "B 音色设计", 1),
    ("C_long_zh", "C 长文本 60 字", 1),
    ("D_streaming", "D 流式合成", 1),
    ("E_clone", "E 声音克隆", 1),
]
rep = json.load(open(f"{OUT}/report.json", encoding="utf-8"))
cer = {c["case"]: c["cer"] for c in json.load(open(f"{OUT}/cer_normalized.json", encoding="utf-8"))["cases"]}
cer.update({k: v["cer"] for k, v in json.load(open(f"{OUT}/asr_extra.json", encoding="utf-8")).items()})

fig, axes = plt.subplots(len(items), 1, figsize=(11, 9.5), sharex=False)
fig.suptitle("VoxCPM2 本地部署验证 · Apple M2 Max / MPS / float32", fontsize=14, fontweight="bold", y=0.985)

for ax, (key, label, _) in zip(axes, items):
    path = f"{OUT}/{key}.wav"
    wav, sr = sf.read(path, dtype="float32")
    t = np.arange(len(wav)) / sr
    ax.plot(t, wav, lw=0.35, color="#1f6feb")
    ax.set_xlim(0, max(t[-1], 0.01))
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("幅度", fontsize=8)
    ax.tick_params(labelsize=7)
    rms = float(np.sqrt(np.mean(wav ** 2)))
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(f"{label}　{len(wav)/sr:.2f}s · 48kHz · RMS {rms:.3f} · ASR CER {cer.get(key, -1)*100:.1f}%",
                 fontsize=9, loc="left", pad=3)

axes[-1].set_xlabel("时间 (秒)", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}/verification_overview.png", dpi=150)
print("saved", os.path.abspath(f"{OUT}/verification_overview.png"))
