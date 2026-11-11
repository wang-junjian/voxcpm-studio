#!/usr/bin/env python3
"""GGUF 路线基准测试：三种权重组合 × 短/长句，顺带采样真实内存峰值。

内存用 libproc 的 ri_phys_footprint 读（RSS 不含 Metal 显存）。
"""
import ctypes
import ctypes.util
import os
import re
import subprocess
import sys
import threading
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

ROOT = str(paths.ROOT)
CLI = str(paths.GGUF_CLI_BIN)
M = str(paths.GGUF_DIR)

COMBOS = [
    ("Q8_0 + Acoustic-F16", "VoxCPM2-BaseLM-Q8_0.gguf", "VoxCPM2-Acoustic-F16.gguf"),
    ("Q8_0 + Acoustic-F32", "VoxCPM2-BaseLM-Q8_0.gguf", "VoxCPM2-Acoustic-F32.gguf"),
    ("F16 + Acoustic-F16", "VoxCPM2-BaseLM-F16.gguf", "VoxCPM2-Acoustic-F16.gguf"),
]

# F16 / F32 是对照档，体积大且非生产必需，默认不保留。缺档跳过，
# 需要复跑对照时按 docs/ASSETS.md 从 assets/pytorch 重新导出即可。
def _present(*names: str) -> bool:
    return all((Path(M) / n).exists() for n in names)

SHORT = "本地语音合成系统的响应速度，比峰值效果更影响体感。"
LONG = ("本地部署的语音合成系统，最关键的不是峰值效果，而是稳定性和响应速度。"
        "当模型常驻内存之后，用户点击生成的瞬间就可以开始出声，不必等待整段文本合成完毕。")

libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")


def fp(pid: int) -> float:
    b = ctypes.create_string_buffer(1024)
    libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(4), b)
    return int.from_bytes(b.raw[72:80], "little") / 2 ** 30


def run(base: str, acoustic: str, text: str, out: str, stream: bool = False):
    """跑一次，返回 (音频秒数, elapsed, 首包秒, 峰值GB, 常驻末尾GB)。"""
    cmd = [CLI, "-t", text, "-o", out, "--seed", "42"]
    if stream:
        cmd.append("--stream")
    cmd += [os.path.join(M, base), os.path.join(M, acoustic)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak = 0.0
    samples = []
    first_audio = None
    t0 = time.perf_counter()

    def watch():
        nonlocal peak, first_audio
        while p.poll() is None:
            try:
                v = fp(p.pid)
            except Exception:
                break
            peak = max(peak, v)
            samples.append(v)
            # 首包：输出文件开始变大
            if first_audio is None and os.path.exists(out) and os.path.getsize(out) > 44:
                first_audio = time.perf_counter() - t0
            time.sleep(0.05)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    out_txt = p.communicate()[0]
    th.join(timeout=1)
    tail = fp(p.pid) if False else (samples[-1] if samples else 0.0)

    dur = re.search(r"Audio: ([\d.]+)s", out_txt)
    el = re.search(r"Elapsed: ([\d.]+)s, RTF=([\d.]+)", out_txt)
    if not dur or not el:
        print(out_txt[-1500:])
        raise RuntimeError("解析失败")
    return float(dur.group(1)), float(el.group(1)), first_audio, peak, tail


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    combos = [c for c in COMBOS if _present(c[1], c[2])]
    if skipped := [c[0] for c in COMBOS if c not in combos]:
        print(f"跳过（权重不存在）：{', '.join(skipped)}\n")
    if not combos:
        sys.exit(f"{M} 下没有可用的权重组合，先按 docs/ASSETS.md 重建")
    if only:
        combos = [c for c in combos if only in c[0]] or combos
    print(f"{'组合':26s} {'文本':6s} {'音频':>7s} {'耗时':>7s} {'RTF':>6s} {'首包':>8s} {'峰值':>7s}")
    results = {}
    for name, base, ac in combos:
        # 预热（首跑要编译 Metal kernel）
        run(base, ac, "预热。", "/tmp/gguf_warm.wav")
        for label, text in (("短", SHORT), ("长", LONG)):
            rows = []
            for _ in range(3):
                dur, el, fa, pk, tl = run(base, ac, text, "/tmp/gguf_bench.wav")
                rows.append((dur, el, fa, pk))
            dur = sum(r[0] for r in rows) / 3
            el = sum(r[1] for r in rows) / 3
            pk = max(r[3] for r in rows)
            fa = next((r[2] for r in rows if r[2]), None)
            fa_s = f"{fa*1000:.0f}ms" if fa else "整段"
            print(f"{name:26s} {label:6s} {dur:7.2f} {el:7.2f} {el/dur:6.3f} {fa_s:>8s} {pk:6.2f}G")
            results[(name, label)] = (dur, el, el / dur, fa, pk)

    # 流式首包（单独测，复用第一组可用权重）
    name, base, ac = combos[0]
    run(base, ac, "预热。", "/tmp/gguf_warm.wav", stream=True)
    print()
    for label, text in (("短", SHORT), ("长", LONG)):
        rows = []
        for _ in range(3):
            dur, el, fa, pk, _ = run(base, ac, text, "/tmp/gguf_stream.wav", stream=True)
            rows.append((dur, el, fa))
        dur = sum(r[0] for r in rows) / 3
        el = sum(r[1] for r in rows) / 3
        fa = next((r[2] for r in rows if r[2]), None)
        print(f"{'[流式] ' + name:26s} {label:6s} {dur:7.2f} {el:7.2f} {el/dur:6.3f} "
              f"{(f'{fa*1000:.0f}ms' if fa else '整段'):>8s}")


if __name__ == "__main__":
    main()
