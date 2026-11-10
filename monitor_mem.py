"""读取进程物理内存占用（phys_footprint），macOS 下用 libproc。

沙箱里 ps/top 会被拦截，RSS 也不含 Metal 显存，所以直接用
proc_pid_rusage(RUSAGE_INFO_V4) 取 ri_phys_footprint。

用法:
  python monitor_mem.py 4716 17647            # 打印一次
  python monitor_mem.py 4716 17647 -n 20 -i 2 # 采样 20 次，间隔 2 秒
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import time

libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
RUSAGE_INFO_V4 = 4
# rusage_info_v4: uuid[16] + 7 个 u64 后才到 ri_phys_footprint
OFF_RESIDENT = 64
OFF_FOOTPRINT = 72


def footprint(pid: int) -> tuple[int, int]:
    buf = ctypes.create_string_buffer(1024)
    rc = libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(RUSAGE_INFO_V4), buf)
    if rc != 0:
        raise OSError(f"proc_pid_rusage({pid}) 失败 rc={rc}")
    raw = buf.raw
    resident = int.from_bytes(raw[OFF_RESIDENT:OFF_RESIDENT + 8], "little")
    phys = int.from_bytes(raw[OFF_FOOTPRINT:OFF_FOOTPRINT + 8], "little")
    return resident, phys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pids", nargs="+", type=int)
    ap.add_argument("-n", "--count", type=int, default=1)
    ap.add_argument("-i", "--interval", type=float, default=2.0)
    args = ap.parse_args()

    hdr = "  ".join(f"{p:>16d}" for p in args.pids)
    print(f"{'t(s)':>6}  {hdr}")
    t0 = time.perf_counter()
    for k in range(args.count):
        cells = []
        for p in args.pids:
            try:
                _resident, phys = footprint(p)
                cells.append(f"{phys / 2**30:15.2f}G")
            except OSError as e:
                cells.append(f"{'ERR':>16}")
                print(e, flush=True)
        print(f"{time.perf_counter() - t0:6.1f}  " + "  ".join(cells), flush=True)
        if k + 1 < args.count:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
