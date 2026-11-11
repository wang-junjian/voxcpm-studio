"""MLX 版 VoxCPM2 基准：与 PyTorch MPS fp32 基线同文本、同参数对比。

用法：.venv/bin/python bench_mlx.py [模型目录] [量化标签]
产出：outputs/mlx_<tag>_<i>.wav + 控制台 JSON 报告
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_audio.tts.utils import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else str(paths.MLX_MODEL)
TAG = sys.argv[2] if len(sys.argv) > 2 else "8bit"

TEXT_SHORT = "这句话的最后几个字必须在流式播放里完整念出来，不能丢尾巴。"
TEXT_LONG = (
    "本地部署的语音合成系统，最关键的不是峰值效果，而是稳定性和响应速度。"
    "当模型常驻内存之后，用户点击生成的瞬间就可以开始出声，不必等待整段文本合成完毕。"
)
OUT = paths.ensure_dirs()
OUT.mkdir(exist_ok=True)


def write_wav(path, samples, sr):
    """用标准库写 16-bit PCM，绕开 soundfile 依赖。"""
    import wave

    import numpy as np

    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = (arr * 32767.0).clip(-32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def run(model, text, label, idx, timesteps=10, cfg=2.0):
    t0 = time.perf_counter()
    result = next(model.generate(text=text, inference_timesteps=timesteps, cfg_value=cfg))
    gen_s = time.perf_counter() - t0
    audio = result.audio
    dur = float(audio.shape[0]) / float(model.sample_rate)
    path = OUT / f"mlx_{TAG}_{label}{idx}.wav"
    write_wav(path, audio.tolist(), int(model.sample_rate))
    return {
        "label": f"{label}{idx}",
        "text_len": len(text),
        "audio_s": round(dur, 3),
        "gen_s": round(gen_s, 3),
        "rtf": round(gen_s / dur, 3),
        "wav": str(path),
    }


def main():
    t_load = time.perf_counter()
    model = load(MODEL)
    load_s = time.perf_counter() - t_load
    sr = int(model.sample_rate)

    # 预热一次，排除首次编译/分配开销
    t0 = time.perf_counter()
    next(model.generate(text="预热。", inference_timesteps=10, cfg_value=2.0))
    warm_s = time.perf_counter() - t0
    mx.eval(mx.zeros(1))  # 确保 MLX 计算队列排空

    rows = []
    for i in range(3):
        rows.append(run(model, TEXT_SHORT, "short", i))
    for i in range(2):
        rows.append(run(model, TEXT_LONG, "long", i))

    peak = mx.get_peak_memory() / 1024**3
    active = mx.get_active_memory() / 1024**3

    report = {
        "engine": "mlx-audio",
        "model": MODEL,
        "tag": TAG,
        "sample_rate": sr,
        "load_s": round(load_s, 1),
        "warmup_s": round(warm_s, 2),
        "peak_mem_gb": round(peak, 2),
        "active_mem_gb": round(active, 2),
        "runs": rows,
        "rtf_mean": round(sum(r["rtf"] for r in rows) / len(rows), 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path(f"outputs/report_mlx_{TAG}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
