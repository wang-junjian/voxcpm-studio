#!/usr/bin/env python3
"""GGUF 服务态基准：走 llama-tts-server 的 HTTP 接口，与 PyTorch/MLX 口径对齐。

对比口径要点：
  - 一次性 = POST /v1/audio/speech（wav），耗时取客户端墙钟
  - 流式   = POST /v1/audio/speech/stream，首包 = 收到第一块 PCM 的时刻
  - 内存   = 进程 ri_phys_footprint（RSS 不含 Metal 显存）
"""
import ctypes
import ctypes.util
import io
import os
import sys
import time
import wave

import httpx
import numpy as np

BASE = "http://127.0.0.1:8819"
PID = int(os.environ.get("PID") or 0)
SHORT = "本地语音合成系统的响应速度，比峰值效果更影响体感。"
LONG = ("本地部署的语音合成系统，最关键的不是峰值效果，而是稳定性和响应速度。"
        "当模型常驻内存之后，用户点击生成的瞬间就可以开始出声，不必等待整段文本合成完毕。")

libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")


def fp(pid: int) -> float:
    b = ctypes.create_string_buffer(1024)
    libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(4), b)
    return int.from_bytes(b.raw[72:80], "little") / 2 ** 30


def speak(text: str):
    body = {"model": "voxcpm2", "input": text, "seed": 42,
            "cfg_value": 2.0, "inference_timesteps": 10}
    t0 = time.perf_counter()
    with httpx.Client(timeout=None) as c:
        r = c.post(f"{BASE}/v1/audio/speech", json=body)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    with wave.open(io.BytesIO(r.content), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
    return n / sr, dt


def stream(text: str):
    body = {"model": "voxcpm2", "input": text, "seed": 42,
            "cfg_value": 2.0, "inference_timesteps": 10}
    t0 = time.perf_counter()
    ttfb = None
    chunks = []
    with httpx.Client(timeout=None) as c:
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
    pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
    return pcm.size / 48000, time.perf_counter() - t0, ttfb, pcm


def main():
    if not PID:
        print("需要 PID=<llama-tts-server 进程号>")
        return 1
    print(f"启动常驻内存 fp={fp(PID):.2f}G")
    # 预热
    speak("预热。")

    for label, text in (("短", SHORT), ("长", LONG)):
        rows = [speak(text) for _ in range(3)]
        dur = sum(r[0] for r in rows) / 3
        dt = sum(r[1] for r in rows) / 3
        print(f"[一次性] {label} 音频 {dur:5.2f}s 耗时 {dt:5.2f}s RTF {dt/dur:.3f} "
              f"| 逐次 {[round(r[1],2) for r in rows]}")

    print()
    for label, text in (("短", SHORT), ("长", LONG)):
        rows = [stream(text) for _ in range(3)]
        dur = sum(r[0] for r in rows) / 3
        dt = sum(r[1] for r in rows) / 3
        ttfb = sum(r[2] for r in rows) / 3
        print(f"[流式]   {label} 音频 {dur:5.2f}s 总 {dt:5.2f}s RTF {dt/dur:.3f} "
              f"首包 {ttfb*1000:.0f}ms | 逐次首包 {[round(r[2]*1000) for r in rows]}")

    # 尾音完整性：末 0.9 s 的峰值
    _, _, _, pcm = stream(LONG)
    tail = pcm[-int(0.9 * 48000):].astype(np.float32) / 32768
    print(f"\n末尾 0.9s 峰值 {np.abs(tail).max():.4f}  RMS {np.sqrt((tail**2).mean()):.4f}")
    print(f"生成后 fp={fp(PID):.2f}G")
    return 0


if __name__ == "__main__":
    sys.exit(main())
