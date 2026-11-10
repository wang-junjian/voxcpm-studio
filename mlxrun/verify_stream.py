"""验证 MLX 真流式与一次性生成的等价性，并测首包延迟。

用法：.venv/bin/python verify_stream.py [模型目录]
"""
import sys
import time

import mlx.core as mx
import numpy as np

from voxcpm_stream import load_model_lite, stream_generate

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/VoxCPM2-8bit"
TEXT = (
    "本地部署的语音合成系统，最关键的不是峰值效果，而是稳定性和响应速度。"
    "当模型常驻内存之后，用户点击生成的瞬间就可以开始出声，"
    "不必等待整段文本合成完毕，而且最后一个字也必须完整念出来。"
)
SEED = 42


def once(model):
    mx.random.seed(SEED)
    t0 = time.perf_counter()
    res = next(model.generate(text=TEXT, inference_timesteps=10, cfg_value=2.0))
    gen = time.perf_counter() - t0
    return np.asarray(res.audio, dtype=np.float32).reshape(-1), gen


def stream(model, chunk_patches, context_frames):
    stats = {}
    t0 = time.perf_counter()
    chunks = []
    marks = []
    for audio in stream_generate(
        model, TEXT, cfg_value=2.0, inference_timesteps=10, seed=SEED,
        chunk_patches=chunk_patches, context_frames=context_frames, stats=stats,
    ):
        if not chunks:
            marks.append(time.perf_counter() - t0)
        chunks.append(audio)
    total = time.perf_counter() - t0
    return np.concatenate(chunks), marks[0], total, stats


def snr(a, b):
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    return 10 * np.log10((a[:n] ** 2).mean() / max((d ** 2).mean(), 1e-20)), float(np.abs(d).max())


def main():
    t_load = time.perf_counter()
    model, info = load_model_lite(MODEL)
    print(
        f"模型 {MODEL} 加载 {time.perf_counter() - t_load:.2f}s "
        f"(纯 load {info['load_s']}s, 分词器 {info['tokenizer']})",
        flush=True,
    )
    sr = int(model.sample_rate)

    next(model.generate(text="预热。", inference_timesteps=10, cfg_value=2.0))

    ref, gen_once = once(model)
    print(f"[基准] 一次性 {len(ref)/sr:.2f}s 音频 / {gen_once:.2f}s 耗时 / RTF {gen_once/(len(ref)/sr):.2f}", flush=True)

    big, _, _, _ = stream(model, 10**6, 0)
    s, m = snr(ref, big)
    print(f"[等价性] 单块流式 vs generate(): 长度 {len(big)}/{len(ref)} SNR {s:.1f} dB 最大差 {m:.2e}", flush=True)

    for cp, cf in ((16, 16), (8, 16), (8, 8), (4, 16)):
        audio, ttfb, total, st = stream(model, cp, cf)
        s, m = snr(ref, audio)
        print(
            f"[分块 cp={cp:2d} ctx={cf:2d}] 音频 {len(audio)/sr:6.2f}s 首包 {ttfb*1000:5.0f}ms "
            f"总 {total:5.2f}s RTF {total/(len(audio)/sr):.3f} 分块 {st['chunks']:3d} "
            f"解码 {st['decode_s']:.2f}s SNR {s:6.1f} dB 最大差 {m:.1e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
