#!/usr/bin/env python3
"""服务态基准（可指定端口），用于对比攒批/步数改动前后的首包与 RTF。"""
import io
import sys
import time
import wave

import httpx
import numpy as np

BASE = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8829}"
TS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

TEXTS = {
    "短(24字)": "本地语音合成系统的响应速度，比峰值效果更影响体感。",
    "长(76字)": ("本地部署的语音合成系统，最关键的不是峰值效果，而是稳定性和响应速度。"
                 "当模型常驻内存之后，用户点击生成的瞬间就可以开始出声，不必等待整段文本合成完毕。"
                 "流式播放在这里扮演的角色，是把首包延迟压到几百毫秒以内。"),
}


def stream(text):
    body = {"model": "voxcpm2", "input": text, "seed": 42,
            "cfg_value": 2.0, "inference_timesteps": TS}
    t0, ttfb, chunks, sizes = time.perf_counter(), None, [], []
    with httpx.Client(timeout=None, trust_env=False) as c:
        with c.stream("POST", f"{BASE}/v1/audio/speech/stream", json=body) as r:
            r.raise_for_status()
            head = b""
            for ch in r.iter_raw():
                if not ch:
                    continue
                if len(head) < 44:
                    head += ch
                    if len(head) < 44:
                        continue
                    ch = head[44:]
                    if not ch:
                        continue
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                chunks.append(ch)
                sizes.append(len(ch) // 2)
    pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
    return pcm.size / 48000, time.perf_counter() - t0, ttfb, sizes


def main():
    stream("预热。")
    for label, text in TEXTS.items():
        rows = [stream(text) for _ in range(3)]
        dur = sum(r[0] for r in rows) / 3
        tot = sum(r[1] for r in rows) / 3
        ttf = sum(r[2] for r in rows) / 3
        print(f"  {label}: 音频 {dur:5.2f}s  总 {tot:5.2f}s  RTF {tot/dur:.3f}  "
              f"首包 {ttf*1000:6.0f}ms  逐次首包 {[round(r[2]*1000) for r in rows]}")
    _, _, _, sizes = stream(TEXTS["长(76字)"])
    print(f"  块数 {len(sizes)}  块大小(样本) {sorted(set(sizes))[:4]}")


if __name__ == "__main__":
    main()
