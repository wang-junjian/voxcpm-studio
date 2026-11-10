#!/usr/bin/env python
"""VoxCPM2 本地推理入口（权重已离线，不访问网络）。

示例:
  python tts.py --text "你好，世界" --out outputs/hello.wav
  python tts.py --text "(A deep male voice)Hello" --out outputs/en.wav
  python tts.py --text "新的话" --ref outputs/A_basic_zh.wav --out outputs/clone.wav
  python tts.py --text "..." --device cpu --out outputs/cpu.wav
"""
import argparse, os, time
import numpy as np
import torch, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "VoxCPM2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="mps", help="mps / cpu / cuda")
    ap.add_argument("--ref", default=None, help="参考音频，启用声音克隆")
    ap.add_argument("--prompt-text", default=None, help="参考音频对应的文字，启用极致克隆")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--denoiser", action="store_true",
                    help="加载降噪模型（需首次联网从 ModelScope 下载）")
    args = ap.parse_args()

    from voxcpm import VoxCPM

    t0 = time.time()
    model = VoxCPM.from_pretrained(args.model, load_denoiser=args.denoiser,
                                   device=args.device, optimize=False)
    print(f"[load] {time.time()-t0:.1f}s device={args.device}", flush=True)
    sr = model.tts_model.sample_rate

    torch.manual_seed(args.seed)
    kw = {}
    if args.ref:
        kw["reference_wav_path"] = args.ref
    if args.prompt_text:
        kw["prompt_text"] = args.prompt_text
        if not args.ref:
            kw["prompt_wav_path"] = args.out

    t = time.time()
    wav = model.generate(text=args.text, cfg_value=args.cfg,
                         inference_timesteps=args.steps, **kw)
    gen_s = time.time() - t
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    out = args.out or os.path.join(HERE, "outputs", f"tts_{int(time.time())}.wav")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sf.write(out, wav, sr)

    dur = len(wav) / sr
    print(f"[gen] {dur:.2f}s 音频 / {gen_s:.2f}s / RTF {gen_s/max(dur,1e-9):.2f}")
    print(f"[out] {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
