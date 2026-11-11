"""VoxCPM2 × MLX 8bit 推理 worker。

跑在 engines/mlx/.venv 里（与主服务的 torch 环境隔离），用标准库 HTTP 服务
对外提供同一套语义：一次性生成 + 真流式生成。主服务 webapp/server.py 按
engine 字段把请求代理过来。

  .venv/bin/python mlx_worker.py --port 8818

接口：
  GET  /health   → 引擎状态 JSON
  POST /gen      → 整段 16-bit PCM（响应头带 X-Gen-Seconds / X-Sample-Rate）
  POST /stream   → chunked 16-bit PCM（真流式，边生成边发）

两个生成接口的请求体都是：
  {"text": "...", "ref_audio": "/abs/path.wav", "prompt_text": "...",
   "cfg": 2.0, "steps": 10, "seed": 42, "chunk_patches": 8}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

BASE = Path(__file__).resolve().parent
MODEL_DIR = paths.MLX_MODEL

# MLX 缓冲池默认没有上限：被释放的缓冲不会还给系统，只留在池里等复用。
# 上限是设备的 recommended working set（这台机器 60.8 GB），所以任何一次
# 峰值都会被永久记在池子里，进程内存只涨不跌。这里主动设一个够用且可控的
# 上限，配合每次生成结束后 clear_cache，把常驻内存钉在模型大小附近。
CACHE_LIMIT_GB = float(os.environ.get("MLX_CACHE_LIMIT_GB", "2.0"))

# MLX 在 GPU 上是单队列，并发推理会互相拖慢并放大内存峰值，必须串行。
GEN_LOCK = threading.Lock()

state = {
    "model": None,
    "ready": False,
    "error": None,
    "sample_rate": 48000,
    "load_s": None,
    "warm_s": None,
    "busy": False,
    "model_dir": str(MODEL_DIR),
    "tokenizer": None,
    "mem": None,
}


def release_memory() -> dict:
    """归还 MLX 缓冲池里没在用的显存，并回传当前占用。

    只在一次生成彻底结束后调用：生成过程中清缓存会把复用中的缓冲也丢掉，
    下一次分配要重新向系统申请，白付一次延迟。
    """
    import mlx.core as mx

    mx.clear_cache()
    return {
        "active_gb": round(mx.get_active_memory() / 2**30, 2),
        "cache_gb": round(mx.get_cache_memory() / 2**30, 2),
        "peak_gb": round(mx.get_peak_memory() / 2**30, 2),
    }


def load_model() -> None:
    t0 = time.perf_counter()
    try:
        import mlx.core as mx

        if CACHE_LIMIT_GB > 0:
            mx.set_cache_limit(int(CACHE_LIMIT_GB * 2**30))

        from voxcpm_stream import load_model_lite, stream_generate

        model, info = load_model_lite(MODEL_DIR)
        state["model"] = model
        state["sample_rate"] = info["sample_rate"]
        state["load_s"] = info["load_s"]
        state["tokenizer"] = info["tokenizer"]

        # 预热：把首块 kernel 编译与显存分配的代价挪到启动阶段，
        # 否则第一次请求的首包会凭空多出几百毫秒。
        t1 = time.perf_counter()
        for _ in stream_generate(model, "预热。", cfg_value=2.0, inference_timesteps=10, seed=0):
            pass
        state["warm_s"] = round(time.perf_counter() - t1, 2)
        mem = release_memory()
        state["mem"] = mem
        state["ready"] = True
        print(
            f"[mlx] 就绪 {state['load_s']}s 预热 {state['warm_s']}s "
            f"sr={state['sample_rate']} 分词器={state['tokenizer']} "
            f"缓存上限={CACHE_LIMIT_GB}GB 常驻={mem['active_gb']}GB "
            f"model={MODEL_DIR.name} 进程总耗时 {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        state["error"] = f"{type(e).__name__}: {e}"
        print(f"[mlx] 加载失败: {state['error']}\n{traceback.format_exc()}", flush=True)


def to_i16(x: np.ndarray) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def build_kwargs(req: dict) -> dict:
    """把请求体翻译成 stream_generate 的参数。"""
    kw = {}
    ref = req.get("ref_audio") or None
    prompt_text = (req.get("prompt_text") or "").strip() or None
    if ref:
        kw["ref_audio"] = ref
        if prompt_text:
            # 有参考文本时走「参考 + 续写」组合模式，与 PyTorch 侧一致
            kw["prompt_text"] = prompt_text
            kw["prompt_audio"] = ref
    return kw


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "VoxCPMLX/1.0"

    # ---------- 基础 ----------
    def log_message(self, fmt, *args):  # 只保留异常，正常请求不刷屏
        if str(args[1] if len(args) > 1 else "")[:1] in ("4", "5"):
            print(f"[mlx] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    def _require_ready(self) -> bool:
        if state["error"]:
            self._json(503, {"detail": f"MLX 引擎加载失败: {state['error']}"})
            return False
        if not state["ready"]:
            self._json(503, {"detail": "MLX 引擎正在加载"})
            return False
        return True

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode())
        self.wfile.write(data)
        self.wfile.write(b"\r\n")

    # ---------- 路由 ----------
    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == "/health":
            self._json(
                200,
                {
                    "engine": "mlx",
                    "ready": state["ready"],
                    "error": state["error"],
                    "sample_rate": state["sample_rate"],
                    "load_s": state["load_s"],
                    "warm_s": state["warm_s"],
                    "busy": state["busy"],
                    "model": MODEL_DIR.name,
                    "tokenizer": state["tokenizer"],
                    "cache_limit_gb": CACHE_LIMIT_GB,
                    "mem": state["mem"],
                },
            )
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path not in ("/gen", "/stream"):
            self._json(404, {"detail": "not found"})
            return
        if not self._require_ready():
            return
        try:
            req = self._read_json()
        except Exception as e:  # noqa: BLE001
            self._json(400, {"detail": f"请求体解析失败: {e}"})
            return

        text = (req.get("text") or "").strip()
        if not text:
            self._json(400, {"detail": "文本不能为空"})
            return

        if path == "/gen":
            self._gen(req, text)
        else:
            self._stream(req, text)

    # ---------- 一次性 ----------
    def _gen(self, req: dict, text: str) -> None:
        from voxcpm_stream import DEFAULT_CONTEXT_FRAMES, stream_generate

        state["busy"] = True
        try:
            with GEN_LOCK:
                t0 = time.perf_counter()
                pieces = []
                # 默认按 16 个 patch（约 2.6 秒）分块解码：
                #   整段一次解码的峰值与文本长度线性相关（19.7 秒音频实测
                #   峰值 8.1 GB，其中模型只占 3 GB），长文本会击穿内存；
                #   分块后峰值固定在 4.3 GB 左右，代价是 RTF 从 0.68 涨到 0.70。
                # 需要与 model.generate() 逐样本一致时传 chunk_patches=1000000
                # + context_frames=0（exact=true 等价写法）。
                if req.get("exact"):
                    cp, ctx = 10**6, 0
                else:
                    cp = int(req.get("chunk_patches", 16))
                    ctx = int(req.get("context_frames", DEFAULT_CONTEXT_FRAMES))
                for audio in stream_generate(
                    state["model"],
                    text,
                    cfg_value=float(req.get("cfg", 2.0)),
                    inference_timesteps=int(req.get("steps", 10)),
                    seed=int(req.get("seed", 42)),
                    chunk_patches=cp,
                    context_frames=ctx,
                    **build_kwargs(req),
                ):
                    pieces.append(audio)
                gen_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            print(f"[mlx] /gen 失败: {type(e).__name__}: {e}", flush=True)
            self._json(500, {"detail": f"生成失败: {e}"})
            return
        finally:
            state["busy"] = False
            state["mem"] = release_memory()

        wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        body = to_i16(wav)
        sr = state["sample_rate"]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("X-Samples", str(wav.size))
        self.send_header("X-Gen-Seconds", f"{gen_s:.4f}")
        self.end_headers()
        self.wfile.write(body)

    # ---------- 流式 ----------
    def _stream(self, req: dict, text: str) -> None:
        from voxcpm_stream import stream_generate

        sr = state["sample_rate"]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        state["busy"] = True
        stats: dict = {}
        try:
            with GEN_LOCK:
                for audio in stream_generate(
                    state["model"],
                    text,
                    cfg_value=float(req.get("cfg", 2.0)),
                    inference_timesteps=int(req.get("steps", 10)),
                    seed=int(req.get("seed", 42)),
                    chunk_patches=int(req.get("chunk_patches", 4)),
                    stats=stats,
                    **build_kwargs(req),
                ):
                    self._chunk(to_i16(audio))
            self.wfile.write(b"0\r\n\r\n")
            print(
                f"[mlx] /stream 音频 {stats.get('audio_s')}s 首包 "
                f"{(stats.get('first_chunk_s') or 0) * 1000:.0f}ms 总 {stats.get('gen_s')}s "
                f"RTF {stats.get('rtf')} 分块 {stats.get('chunks')}",
                flush=True,
            )
        except (BrokenPipeError, ConnectionResetError):
            # 前端中途停止播放，模型这边没有副作用，直接丢弃剩余产出
            print("[mlx] /stream 客户端断开，提前结束", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[mlx] /stream 失败: {type(e).__name__}: {e}", flush=True)
            try:
                self._chunk(b"")  # 收尾，避免连接悬挂
            except Exception:  # noqa: BLE001
                pass
        finally:
            state["busy"] = False
            state["mem"] = release_memory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("MLX_PORT", 8818)))
    ap.add_argument("--host", default=os.environ.get("MLX_HOST", "127.0.0.1"))
    a = ap.parse_args()

    if not MODEL_DIR.exists():
        raise SystemExit(f"模型目录不存在: {MODEL_DIR}")

    load_model()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"[mlx] 监听 http://{a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[mlx] 退出", flush=True)


if __name__ == "__main__":
    main()
