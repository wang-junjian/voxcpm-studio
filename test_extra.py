"""补充验证：流式生成（首包延迟） + 声音克隆（用自生成音频作参考）。"""
import os, time, json
import numpy as np
import torch, soundfile as sf

MODEL_DIR = os.path.abspath("models/VoxCPM2")
OUT_DIR = os.path.abspath("outputs")

from voxcpm import VoxCPM
t0 = time.time()
model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, device="mps", optimize=False)
sr = model.tts_model.sample_rate
print(f"加载完成 {time.time()-t0:.1f}s sr={sr}", flush=True)

res = {}

# ---- 流式 ----
text = "流式合成可以让用户在第一个音频包生成后立刻开始播放，而不必等待整段文本结束。"
print("\n== 流式生成 ==", flush=True)
torch.manual_seed(7)
chunks, t_first, t_start = [], None, time.time()
for ch in model.generate_streaming(text=text, cfg_value=2.0, inference_timesteps=10):
    if t_first is None:
        t_first = time.time() - t_start
    chunks.append(np.asarray(ch).reshape(-1))
total_s = time.time() - t_start
wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
dur = len(wav) / sr
p = os.path.join(OUT_DIR, "D_streaming.wav")
sf.write(p, wav, sr)
res["streaming"] = {"chunks": len(chunks), "ttfb_s": round(t_first or -1, 3),
                    "total_gen_s": round(total_s, 2), "duration_s": round(dur, 2),
                    "rtf": round(total_s / max(dur, 1e-9), 3), "path": p}
print(f"chunk={len(chunks)}  首包={t_first:.2f}s  总耗时={total_s:.2f}s  音频={dur:.2f}s  RTF={total_s/dur:.2f}", flush=True)

# ---- 声音克隆（参考音频 = 场景 A 的合成结果）----
ref = os.path.join(OUT_DIR, "A_basic_zh.wav")
clone_text = "现在我用刚才那段音频的音色来说一句新的话，检验声音克隆是否生效。"
print("\n== 声音克隆 ==", flush=True)
try:
    torch.manual_seed(11)
    t = time.time()
    wav2 = model.generate(text=clone_text, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=10)
    gen_s = time.time() - t
    wav2 = np.asarray(wav2, dtype=np.float32).reshape(-1)
    p2 = os.path.join(OUT_DIR, "E_clone.wav")
    sf.write(p2, wav2, sr)
    res["clone"] = {"gen_s": round(gen_s, 2), "duration_s": round(len(wav2)/sr, 2),
                    "rtf": round(gen_s / max(len(wav2)/sr, 1e-9), 3),
                    "rms": round(float(np.sqrt(np.mean(wav2**2))), 4), "path": p2}
    print(f"OK 耗时={gen_s:.2f}s 音频={len(wav2)/sr:.2f}s RTF={gen_s/(len(wav2)/sr):.2f}", flush=True)
except Exception as e:
    res["clone"] = {"error": f"{type(e).__name__}: {e}"}
    print(f"FAILED {type(e).__name__}: {e}", flush=True)

json.dump(res, open(os.path.join(OUT_DIR, "extra_check.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nEXTRA_DONE", flush=True)
