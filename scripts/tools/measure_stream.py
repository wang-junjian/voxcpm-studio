"""测量流式接口的分块到达节奏：每块字节数、到达间隔、累计音频时长。

用法: python measure_stream.py <port> <engine> [steps] [seed]
输出：逐块表格 + 汇总（首包、总生成时长、音频总长、RTF、平均/最大间隔、
「音频时长 vs 墙钟」的欠载量）。
"""
import sys
import time
import json
import httpx

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8817
ENGINE = sys.argv[2] if len(sys.argv) > 2 else "gguf"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42

TEXTS = {
    "short": "今天天气不错，我们出去走走吧。",
    "mid": "本地语音合成系统的响应速度，比峰值效果更影响体感。",
    "long": "包含分段堆叠图、逐 block 边际成本柱图、四个证伪的原始数据、分层优化表和复现命令，"
            "这些内容共同构成了这份报告的完整结构，也决定了它的阅读节奏。",
}

for name, text in TEXTS.items():
    body = {"text": text, "engine": ENGINE, "steps": STEPS, "seed": SEED}
    rows = []
    t0 = time.perf_counter()
    with httpx.Client(timeout=None, trust_env=False) as c:
        with c.stream("POST", f"http://127.0.0.1:{PORT}/api/stream", json=body) as r:
            sr = int(r.headers.get("X-Sample-Rate", 48000))
            for chunk in r.iter_raw():
                if not chunk:
                    continue
                rows.append((time.perf_counter() - t0, len(chunk) / 2 / sr))
    total_s = time.perf_counter() - t0

    acc = 0.0
    print(f"\n=== {name} ({len(text)} 字, {ENGINE}, ts={STEPS}) ===")
    print("  序号   到达(ms)   本块音频(ms)   累计音频(ms)   与墙钟差(ms)")
    prev_t, prev_acc = 0.0, 0.0
    for i, (t, dur) in enumerate(rows):
        acc += dur
        if i < 22 or i >= len(rows) - 3:
            print(f"  {i:3d}  {t*1000:9.0f}  {dur*1000:12.0f}  {acc*1000:13.0f}  {(acc-t)*1000:13.0f}")
    gaps = [(rows[i][0] - rows[i - 1][0]) for i in range(1, len(rows))]
    audio = sum(d for _, d in rows)
    # 播放欠载：按「到第一块就开始播」的朴素策略，累计缺口 = 墙钟 - 音频
    naive_under = max(0.0, total_s - audio - rows[0][0] * 0)
    print(f"  块数={len(rows)}  首包={rows[0][0]*1000:.0f}ms  音频总长={audio:.2f}s  "
          f"生成总时长={total_s:.2f}s  RTF={total_s/audio:.2f}")
    if gaps:
        print(f"  块间隔: 均值={sum(gaps)/len(gaps)*1000:.0f}ms  最大={max(gaps)*1000:.0f}ms  "
              f"最小={min(gaps)*1000:.0f}ms")
    print(f"  块大小: 最小={min(d for _,d in rows)*1000:.0f}ms  最大={max(d for _,d in rows)*1000:.0f}ms")
    # 模拟：给定「缓冲到 lead 秒才开播」策略下的累计静音缺口
    def sim(lead):
        """事件驱动模拟。返回 (累计静音缺口秒, 缺口次数, 开播时刻)。"""
        buf, gap, gaps_n, started, start_at = 0.0, 0.0, 0, False, None
        last_t = rows[0][0]
        for i, (t, dur) in enumerate(rows):
            dt = t - last_t
            if started and dt > 0:
                if buf >= dt:
                    buf -= dt
                else:
                    gap += dt - buf
                    gaps_n += 1
                    buf = 0.0
            buf += dur
            if not started and t >= lead:
                started = True
                start_at = t
            last_t = t
        if not started:
            start_at = rows[-1][0]
        return gap, gaps_n, start_at

    print("  起步延迟 vs 播放期间累计静音缺口：")
    for lead in (0.15, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 99):
        g, n, st = sim(lead)
        tag = "（=等整段生成完）" if lead == 99 else ""
        print(f"    lead={lead:5.2f}s -> 开播 {st:5.2f}s, 缺口 {g*1000:6.0f} ms / {n} 次{tag}")
