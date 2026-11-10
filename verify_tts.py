"""VoxCPM2 本地部署验证：三场景合成 + 音频质量指标。

场景 A 基础中文 TTS / B 音色设计 / C 较长文本稳定性
输出：outputs/*.wav + outputs/report.json
"""
import os, sys, time, json, resource

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import soundfile as sf

MODEL_DIR = os.path.abspath("models/VoxCPM2")
OUT_DIR = os.path.abspath("outputs")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = os.environ.get("VOXCPM_DEVICE", "mps")

def maxrss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)

def metrics(wav: np.ndarray, sr: int) -> dict:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    silent = float(np.mean(np.abs(wav) < 1e-4)) if wav.size else 1.0
    clipped = float(np.mean(np.abs(wav) >= 0.999)) if wav.size else 0.0
    # 过零率：粗略反映语音活跃度（静音段极低）
    zcr = float(np.mean(np.abs(np.diff(np.signbit(wav).astype(np.int8))))) if wav.size > 1 else 0.0
    return {
        "samples": int(wav.size),
        "duration_s": round(wav.size / sr, 3),
        "sr": sr,
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "dbfs": round(20 * np.log10(max(rms, 1e-9)), 2),
        "silence_ratio": round(silent, 4),
        "clipped_ratio": round(clipped, 6),
        "zcr": round(zcr, 4),
    }

print(f"== 加载模型 device={DEVICE} ==", flush=True)
t0 = time.time()
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, device=DEVICE, optimize=False)
load_s = time.time() - t0
sr = model.tts_model.sample_rate
print(f"== 加载完成 {load_s:.1f}s, sr={sr}, 峰值内存 {maxrss_gb():.2f} GB ==", flush=True)

CASES = [
    ("A_basic_zh", "你好，这是 VoxCPM2 在 Mac 本地部署的验证音频，采样率为四万八千赫兹。", {}),
    ("B_voice_design", "(A warm, gentle young female voice, speaking slowly)这是一段通过文字描述生成的音色。", {}),
    ("C_long_zh", "人工智能正在改变内容生产的方式。语音合成作为其中重要的一环，"
                  "不仅需要准确的发音，还需要自然的韵律和恰当的情感表达。"
                  "这段较长的文本用于检验模型在持续生成时的稳定性。", {}),
]

results = {"device": DEVICE, "load_s": round(load_s, 2), "sample_rate": sr, "cases": {}}

for name, text, kw in CASES:
    print(f"\n== {name} ==", flush=True)
    print(f"text: {text}", flush=True)
    torch.manual_seed(42)
    try:
        t = time.time()
        wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10, **kw)
        gen_s = time.time() - t
        path = os.path.join(OUT_DIR, f"{name}.wav")
        sf.write(path, np.asarray(wav, dtype=np.float32).reshape(-1), sr)
        m = metrics(wav, sr)
        m["gen_s"] = round(gen_s, 2)
        m["rtf"] = round(gen_s / max(m["duration_s"], 1e-9), 3)
        m["path"] = path
        m["text"] = text
        results["cases"][name] = m
        print(f"OK {m['duration_s']}s 音频 / 耗时 {gen_s:.2f}s / RTF {m['rtf']} / RMS {m['rms']} / 静音比 {m['silence_ratio']}", flush=True)
    except Exception as e:
        results["cases"][name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)

results["peak_mem_gb"] = round(maxrss_gb(), 2)
with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n== 汇总 ==", flush=True)
print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
print("VERIFY_DONE", flush=True)
