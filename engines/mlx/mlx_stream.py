"""测量 MLX 版流式首包延迟与分块节奏。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

from mlx_audio.tts.utils import load  # noqa: E402

TEXT = "流式合成可以让用户在第一个音频包生成后立刻开始播放，而不必等待整段文本结束。"


def main():
    model = load(str(paths.MLX_MODEL))
    # 预热
    next(model.generate(text="预热。", inference_timesteps=10, cfg_value=2.0))

    t0 = time.perf_counter()
    first = None
    chunks = []
    total_samples = 0
    sr = int(model.sample_rate)
    for r in model.generate(text=TEXT, inference_timesteps=10, cfg_value=2.0):
        if first is None:
            first = time.perf_counter() - t0
        total_samples += int(r.audio.shape[0])
        chunks.append(round(time.perf_counter() - t0, 3))
    total = time.perf_counter() - t0
    dur = total_samples / sr
    print(f"首包 {first * 1000:.0f} ms | 分块 {len(chunks)} | 音频 {dur:.2f}s | 总耗时 {total:.2f}s | RTF {total / dur:.2f}")
    print("前 6 个分块到达时刻(s):", chunks[:6])


if __name__ == "__main__":
    main()
