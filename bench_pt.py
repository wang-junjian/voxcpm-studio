"""PyTorch/MPS 基线基准：与 bench_mlx.py 完全相同的文本与统计口径。"""
import json
import os
import tempfile
import time
from pathlib import Path

# 沙箱里 tempfile 探测（写后删）会失败，torch/dill 因此无法导入，先钉死目录
_HERE = os.path.dirname(os.path.abspath(__file__))
tempfile.tempdir = os.path.join(_HERE, "tmpdir")
os.makedirs(tempfile.tempdir, exist_ok=True)

import soundfile as sf  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

TEXT_SHORT = "这句话的最后几个字必须在流式播放里完整念出来，不能丢尾巴。"
TEXT_LONG = (
    "本地部署的语音合成系统，最关键的不是峰值效果，而是稳定性和响应速度。"
    "当模型常驻内存之后，用户点击生成的瞬间就可以开始出声，不必等待整段文本合成完毕。"
)
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def main():
    t0 = time.perf_counter()
    model = VoxCPM.from_pretrained("models/VoxCPM2", load_denoiser=False)
    load_s = time.perf_counter() - t0
    sr = model.tts_model.sample_rate

    t0 = time.perf_counter()
    model.generate(text="预热。", cfg_value=2.0, inference_timesteps=10)
    warm_s = time.perf_counter() - t0

    rows = []
    for label, text, n in (("short", TEXT_SHORT, 3), ("long", TEXT_LONG, 2)):
        for i in range(n):
            t = time.perf_counter()
            wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
            gen_s = time.perf_counter() - t
            dur = len(wav) / sr
            p = OUT / f"pt_fp32_{label}{i}.wav"
            sf.write(str(p), wav, sr)
            rows.append({
                "label": f"{label}{i}",
                "text_len": len(text),
                "audio_s": round(dur, 3),
                "gen_s": round(gen_s, 3),
                "rtf": round(gen_s / dur, 3),
                "wav": str(p),
            })

    report = {
        "engine": "pytorch-mps-fp32",
        "sample_rate": sr,
        "load_s": round(load_s, 1),
        "warmup_s": round(warm_s, 2),
        "runs": rows,
        "rtf_mean": round(sum(r["rtf"] for r in rows) / len(rows), 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path("outputs/report_pt_fp32.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
