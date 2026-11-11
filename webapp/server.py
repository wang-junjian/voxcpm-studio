"""VoxCPM2 Web 语音合成服务。

双引擎：

* ``torch`` —— 本项目 .venv 里的 PyTorch MPS fp32，进程内常驻，MPS 单实例串行。
* ``mlx``   —— engines/mlx/.venv 里的 MLX 8bit（3.0 GB，RTF 0.70），由
  ``engines/mlx/mlx_worker.py`` 以独立进程提供，本服务按请求代理过去。两个引擎的
  Python 依赖互不兼容（torch 与 mlx 分处两个 venv），所以必须进程隔离。

请求体里的 ``engine`` 字段决定走哪条路径；缺省用 MLX，不可用时回落到 torch。

启动:
  .venv/bin/python webapp/server.py --port 8817
  engines/mlx/.venv/bin/python engines/mlx/mlx_worker.py --port 8818   # 可选，MLX 引擎
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

# 路径唯一真源。paths 只依赖标准库，可以安全地放在 torch 之前导入。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

# 先钉死临时目录再导入 torch：某些受限环境（如 safe-delete 拦截写后删）里，
# tempfile 的默认目录探测会失败，导致 torch 直接 ImportError。
_TMP = paths.pin_tempdir()

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MODEL_DIR = paths.PYTORCH_MODEL
RUNS = BASE / "runs"
REFS = BASE / "refs"
STATIC = BASE / "static"

DEVICE = os.environ.get("VOXCPM_DEVICE", "mps")
MAX_TEXT = 800

# MLX worker（独立进程、独立 venv）
MLX_URL = os.environ.get("VOXCPM_MLX_URL", "http://127.0.0.1:8818").rstrip("/")
MLX_TIMEOUT = 2.0

RUNS.mkdir(parents=True, exist_ok=True)
REFS.mkdir(parents=True, exist_ok=True)

# MPS 上并发推理会相互干扰，必须串行。一次性与流式共用同一把线程锁，
# 端点写成同步函数，由 Starlette 放进线程池执行，不阻塞事件循环。
GEN_LOCK = threading.Lock()

state = {
    "model": None,
    "sr": None,
    "ready": False,
    "error": None,
    "load_s": None,
    "device": DEVICE,
    "busy": False,
}

# MLX worker 的健康状态缓存：页面每 2 秒轮询一次，没必要每次都打通 worker
_mlx = {"checked": 0.0, "info": None}


def _load_model():
    try:
        t0 = time.time()
        from voxcpm import VoxCPM

        model = VoxCPM.from_pretrained(
            str(MODEL_DIR), load_denoiser=False, device=DEVICE, optimize=False
        )
        state["model"] = model
        state["sr"] = int(model.tts_model.sample_rate)
        state["load_s"] = round(time.time() - t0, 1)
        state["ready"] = True
        print(f"[voxcpm] 模型就绪 {state['load_s']}s device={DEVICE} sr={state['sr']}", flush=True)
    except Exception as e:  # noqa: BLE001
        state["error"] = f"{type(e).__name__}: {e}"
        print(f"[voxcpm] 模型加载失败: {state['error']}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_model, daemon=True).start()
    yield


app = FastAPI(title="VoxCPM2 Studio", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT)
    mode: str = "basic"                # basic | design | clone
    engine: str = "auto"               # auto | mlx | torch
    ref_id: str | None = None
    prompt_text: str | None = None
    cfg: float = 2.0
    steps: int = 10
    seed: int = 42
    chunk_patches: int = 4             # 仅 MLX：每块含多少个 patch（1 patch = 160 ms）


def _require_ready():
    if state["error"]:
        raise HTTPException(503, f"模型加载失败: {state['error']}")
    if not state["ready"]:
        raise HTTPException(503, "模型正在加载，请稍候")


# ---------------- 电平：退化检测 + 响度统一 ----------------
#
# 模型在低扩散步数（10 步）下，某些「文本 × 噪声种子」会落进低能量盆地：
# 整段变成气声，RMS 只有正常样本的 1/8、峰值 1/5，听感就是「声音非常小」。
# 实测（16 个 seed，同一文本，ts=10）：GGUF 6%、MLX 12%、PyTorch 0% 会命中，
# 且与量化无关（Q8_0 与 F16 命中率相同）。换 seed 即可恢复，所以两道防线：
#
#   1. 检测到退化 → 换 seed 重试（最多 VOXCPM_MAX_ATTEMPTS 次），保留最响的一次；
#   2. 正常样本统一到目标响度，消除同引擎不同文本间 RMS 0.04–0.25 的波动。
#
# 只放大不重试是不够的：退化样本放大后 CER 21%，正常样本 7.9%，
# 内容本身已经损伤，必须换种子重新生成。
LOUD_TARGET_RMS = float(os.environ.get("VOXCPM_TARGET_RMS", "0.10"))   # ≈ -20 dBFS
LOUD_PEAK_CEIL = float(os.environ.get("VOXCPM_PEAK_CEIL", "0.95"))     # ≈ -0.4 dBFS
LOUD_GAIN_RANGE = (0.25, 16.0)                                        # 合理的增益区间
DEGEN_RMS = float(os.environ.get("VOXCPM_DEGEN_RMS", "0.035"))
DEGEN_PEAK = float(os.environ.get("VOXCPM_DEGEN_PEAK", "0.22"))
MAX_ATTEMPTS = max(1, int(os.environ.get("VOXCPM_MAX_ATTEMPTS", "3")))


def _levels(wav: np.ndarray) -> tuple[float, float]:
    """返回 (RMS, 峰值)，均在 [-1, 1] 浮点域。"""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(a * a))), float(np.max(np.abs(a)))


def _is_degenerate(rms: float, peak: float) -> bool:
    return rms < DEGEN_RMS or peak < DEGEN_PEAK


def _apply_loudness(wav: np.ndarray) -> tuple[np.ndarray, float]:
    """把整段响度拉到目标值，同时保证不削顶。返回 (波形, 实际增益)。"""
    rms, peak = _levels(wav)
    if rms < 1e-5 or peak <= 1e-6:
        return wav, 1.0
    gain = min(LOUD_TARGET_RMS / rms, LOUD_PEAK_CEIL / peak)
    lo, hi = LOUD_GAIN_RANGE
    gain = float(min(max(gain, lo), hi))
    if 0.95 <= gain <= 1.05:          # 已在目标附近，不引入无意义的增益
        return wav, 1.0
    a = np.asarray(wav, dtype=np.float32).reshape(-1) * gain
    return np.clip(a, -1.0, 1.0), gain


def _resolve_ref(ref_id: str | None) -> str | None:
    if not ref_id:
        return None
    p = REFS / f"{ref_id}.wav"
    if not p.exists():
        raise HTTPException(400, "参考音频不存在，请重新上传")
    return str(p)


def _gen_sync(text: str, ref: str | None, prompt_text: str | None,
              cfg: float, steps: int, seed: int) -> tuple[np.ndarray, float]:
    import torch

    model = state["model"]
    torch.manual_seed(seed)
    kw = {}
    if ref:
        kw["reference_wav_path"] = ref
    if prompt_text and ref:
        kw["prompt_text"] = prompt_text
        kw["prompt_wav_path"] = ref
    t0 = time.time()
    wav = model.generate(text=text, cfg_value=cfg, inference_timesteps=steps, **kw)
    return np.asarray(wav, dtype=np.float32).reshape(-1), time.time() - t0


def _to_i16(x: np.ndarray) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


# ---------------- MLX worker 代理 ----------------

def _mlx_probe(force: bool = False, ttl: float = 3.0) -> dict | None:
    """探测 MLX worker，就绪则返回 /health 内容。结果缓存 ttl 秒。"""
    now = time.time()
    if not force and now - _mlx["checked"] < ttl:
        return _mlx["info"]
    _mlx["checked"] = now
    try:
        with httpx.Client(timeout=MLX_TIMEOUT) as c:
            r = c.get(f"{MLX_URL}/health")
            r.raise_for_status()
            info = r.json()
        _mlx["info"] = info if info.get("ready") else None
    except Exception:  # noqa: BLE001
        _mlx["info"] = None
    return _mlx["info"]


def _pick_engine(want: str) -> str:
    """决定本次请求实际用哪个引擎。"""
    want = (want or "auto").lower()
    if want == "torch":
        return "torch"
    if want == "gguf":
        if _gguf_probe() is None:
            raise HTTPException(503, f"GGUF 引擎不可用（llama-tts-server 未就绪于 {GGUF_URL}）")
        return "gguf"
    if want == "mlx":
        if _mlx_probe() is None:
            raise HTTPException(503, f"MLX 引擎不可用（worker 未启动在 {MLX_URL}）")
        return "mlx"
    # auto：MLX 优先，不可用则回落 PyTorch
    return "mlx" if _mlx_probe() is not None else "torch"


def _mlx_body(req: TTSRequest, text: str, ref: str | None) -> dict:
    body = {
        "text": text,
        "cfg": req.cfg,
        "steps": req.steps,
        "seed": req.seed,
        "chunk_patches": req.chunk_patches,
    }
    if ref:
        body["ref_audio"] = ref
        if req.prompt_text:
            body["prompt_text"] = req.prompt_text
    return body


def _mlx_error(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail") or r.text[:200])
    except Exception:  # noqa: BLE001
        return r.text[:200]


def _mlx_gen(req: TTSRequest, text: str, ref: str | None) -> tuple[np.ndarray, int, float]:
    """一次性生成：拿回整段 int16 PCM，转成 float32。"""
    try:
        with httpx.Client(timeout=None) as c:
            r = c.post(f"{MLX_URL}/gen", json=_mlx_body(req, text, ref))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"MLX 引擎通信失败: {e}")
    if r.status_code != 200:
        raise HTTPException(502, f"MLX 引擎返回错误: {_mlx_error(r)}")
    sr = int(r.headers.get("X-Sample-Rate") or 48000)
    gen_s = float(r.headers.get("X-Gen-Seconds") or 0.0)
    wav = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
    return wav, sr, gen_s


# ---------------- GGUF / llama-tts-server 代理 ----------------
#
# llama-tts-server 是 llama.cpp-omni 的 OpenAI 兼容 TTS 服务：HTTP 层是 C++，
# 权重量化与 Metal kernel 都在 ggml 侧。它只吐 WAV（分块流也是 WAV 头 + 裸 PCM），
# 所以这里把 44 字节头剥掉，统一成和另外两个引擎一样的 L16 裸 PCM。

GGUF_URL = os.environ.get("VOXCPM_GGUF_URL", "http://127.0.0.1:8819").rstrip("/")
GGUF_TIMEOUT = 2.0
WAV_HEADER = 44

_gguf = {"checked": 0.0, "info": None}


def _gguf_probe(force: bool = False, ttl: float = 3.0) -> dict | None:
    """探测 llama-tts-server。只有模型真的加载完成才算就绪。

    /health 恒返回 ok，所以要用 /v1/audio/speech/models——它只在 runtime 初始化
    之后才返回非空列表。
    """
    now = time.time()
    if not force and now - _gguf["checked"] < ttl:
        return _gguf["info"]
    _gguf["checked"] = now
    try:
        with httpx.Client(timeout=GGUF_TIMEOUT) as c:
            r = c.get(f"{GGUF_URL}/v1/audio/speech/models")
            r.raise_for_status()
            data = r.json()
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, dict):          # 万一 res_ok 外面又包了一层
            items = items.get("data")
        if not items:
            _gguf["info"] = None
        else:
            it = items[0]
            _gguf["info"] = {
                "sample_rate": int(it.get("sample_rate") or 48000),
                "patch_size": it.get("patch_size"),
                "model": it.get("id") or "voxcpm2",
            }
    except Exception:  # noqa: BLE001
        _gguf["info"] = None
    return _gguf["info"]


def _gguf_body(req: TTSRequest, text: str, ref: str | None) -> dict:
    import base64

    body = {
        "model": "voxcpm2",
        "input": text,
        "seed": req.seed,
        "cfg_value": req.cfg,
        "inference_timesteps": req.steps,
    }
    if ref:
        body["reference_audio"] = base64.b64encode(Path(ref).read_bytes()).decode()
        if req.prompt_text:
            body["prompt_text"] = req.prompt_text
    return body


def _gguf_gen(req: TTSRequest, text: str, ref: str | None) -> tuple[np.ndarray, int, float]:
    """一次性生成。llama-tts-server 不回传耗时，用客户端墙钟当 gen_s。"""
    import io
    import wave

    body = _gguf_body(req, text, ref)
    body["response_format"] = "wav"
    t0 = time.time()
    try:
        with httpx.Client(timeout=None) as c:
            r = c.post(f"{GGUF_URL}/v1/audio/speech", json=body)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"GGUF 引擎通信失败: {e}")
    gen_s = time.time() - t0
    if r.status_code != 200:
        raise HTTPException(502, f"GGUF 引擎返回错误: {_mlx_error(r)}")
    with wave.open(io.BytesIO(r.content), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return wav, sr, gen_s


@app.get("/", response_class=HTMLResponse)
async def index():
    f = STATIC / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>静态文件缺失</h1>", status_code=500)
    return HTMLResponse(f.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    mlx, gguf = await asyncio.to_thread(_mlx_probe), await asyncio.to_thread(_gguf_probe)
    torch_ready = bool(state["ready"])
    return {
        # 旧字段保持不变，前端只加了 engines
        "ready": torch_ready or bool(mlx) or bool(gguf),
        "error": state["error"],
        "device": state["device"],
        "sample_rate": state["sr"] or (mlx or gguf or {}).get("sample_rate"),
        "load_s": state["load_s"],
        "busy": state["busy"] or bool((mlx or {}).get("busy")),
        "model": "VoxCPM2 (2B)",
        "default_engine": "mlx" if mlx else ("gguf" if gguf else "torch"),
        "engines": {
            "torch": {
                "available": True,
                "ready": torch_ready,
                "error": state["error"],
                "device": state["device"],
                "sample_rate": state["sr"],
                "load_s": state["load_s"],
                "busy": state["busy"],
                "label": "PyTorch fp32",
                "stream": True,
            },
            "mlx": {
                "available": bool(mlx),
                "ready": bool(mlx),
                "error": None if mlx else f"worker 未响应（{MLX_URL}）",
                "device": "mlx",
                "sample_rate": (mlx or {}).get("sample_rate"),
                "load_s": (mlx or {}).get("load_s"),
                "busy": bool((mlx or {}).get("busy")),
                "label": "MLX 8bit",
                "model": (mlx or {}).get("model"),
                "stream": True,
            },
            "gguf": {
                "available": bool(gguf),
                "ready": bool(gguf),
                "error": None if gguf else f"llama-tts-server 未响应（{GGUF_URL}）",
                "device": "metal",
                "sample_rate": (gguf or {}).get("sample_rate"),
                "load_s": None,
                "busy": False,
                "label": "GGUF Q8_0",
                "model": (gguf or {}).get("model"),
                "stream": True,
            },
        },
    }


def _engine_gen(engine: str, req: TTSRequest, text: str,
                ref: str | None) -> tuple[np.ndarray, int, float]:
    """按引擎生成一次。req.seed 决定噪声，重试时由调用方改。"""
    if engine == "mlx":
        return _mlx_gen(req, text, ref)
    if engine == "gguf":
        return _gguf_gen(req, text, ref)
    _require_ready()
    state["busy"] = True
    try:
        with GEN_LOCK:
            wav, gen_s = _gen_sync(text, ref, req.prompt_text, req.cfg,
                                   req.steps, req.seed)
    finally:
        state["busy"] = False
    return wav, state["sr"], gen_s


def _save_run(wav: np.ndarray, sr: int, gen_s: float, req: TTSRequest,
              text: str, engine: str) -> dict:
    aid = uuid.uuid4().hex[:12]
    sf.write(RUNS / f"{aid}.wav", wav, sr, subtype="PCM_16")
    dur = len(wav) / max(sr, 1)
    meta = {
        "id": aid,
        "text": text,
        "mode": req.mode,
        "engine": engine,
        "duration": round(dur, 2),
        "gen_s": round(gen_s, 2),
        "rtf": round(gen_s / max(dur, 1e-9), 2),
        "sr": sr,
        "cfg": req.cfg,
        "steps": req.steps,
        "seed": req.seed,
        "rms": round(_levels(wav)[0], 4),
        "created": time.time(),
    }
    (RUNS / f"{aid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


@app.post("/api/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "文本不能为空")
    ref = _resolve_ref(req.ref_id) if req.mode == "clone" else None
    if req.mode == "clone" and not ref:
        raise HTTPException(400, "克隆模式需要先上传参考音频")

    engine = _pick_engine(req.engine)

    # 退化样本检测 + 换 seed 重试：命中低能量盆地时，换一个噪声种子重生成。
    # 种子沿用用户给的会永远得到同一段气声（前端默认固定 seed=42），所以必须换。
    best = None          # (rms, wav, sr, gen_s, seed)
    used_seed = req.seed
    tried = []
    for attempt in range(MAX_ATTEMPTS):
        sub = req.model_copy(update={"seed": used_seed})
        wav, sr, gen_s = _engine_gen(engine, sub, text, ref)
        rms, peak = _levels(wav)
        tried.append({"seed": used_seed, "rms": round(rms, 4),
                      "peak": round(peak, 3)})
        if best is None or rms > best[0]:
            best = (rms, wav, sr, gen_s, used_seed)
        if not _is_degenerate(rms, peak):
            break
        print(f"[tts] {engine} seed={used_seed} 电平退化 "
              f"(RMS={rms:.4f} peak={peak:.3f})，换 seed 重试", flush=True)
        used_seed = (req.seed + attempt + 1) % 1_000_000
    else:
        # 全部尝试都退化：用其中最响的一次，至少别给用户一段更轻的
        _, wav, sr, gen_s, used_seed = best

    wav, gain = _apply_loudness(wav)
    req.seed = used_seed

    meta = _save_run(wav, sr, gen_s, req, text, engine)
    return JSONResponse({
        **meta,
        "url": f"/api/audio/{meta['id']}.wav",
        "peaks": _peak_envelope(wav, 240),
        "attempts": len(tried),
        "gain": round(gain, 2),
    })


def _peak_envelope(wav: np.ndarray, buckets: int) -> list[float]:
    if wav.size == 0:
        return []
    idx = np.linspace(0, wav.size, buckets + 1).astype(int)
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        seg = wav[a:b]
        out.append(round(float(np.max(np.abs(seg))) if seg.size else 0.0, 4))
    peak = max(out) or 1.0
    return [round(v / peak, 4) for v in out]


@app.post("/api/stream")
def stream(req: TTSRequest):
    """流式合成：以 L16 裸 PCM 分块返回，前端边收边播。

    MLX 引擎是真流式（自回归循环里每 8 个 patch 解码一次，首包约 0.95 s），
    PyTorch 引擎走模型自带的 generate_streaming。
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "文本不能为空")
    ref = _resolve_ref(req.ref_id) if req.mode == "clone" else None
    if req.mode == "clone" and not ref:
        raise HTTPException(400, "克隆模式需要先上传参考音频")

    engine = _pick_engine(req.engine)

    if engine == "gguf":
        info = _gguf_probe() or {}
        sr = int(info.get("sample_rate") or 48000)
        body = _gguf_body(req, text, ref)

        def gen_gguf():
            """llama-tts-server 的分块流是「44 字节 WAV 头 + 裸 PCM」，剥掉头再转发。"""
            head = b""
            try:
                with httpx.Client(timeout=None) as c:
                    with c.stream("POST", f"{GGUF_URL}/v1/audio/speech/stream", json=body) as r:
                        if r.status_code != 200:
                            print(f"[stream/gguf] server 返回 {r.status_code}", flush=True)
                            return
                        for chunk in r.iter_raw():
                            if not chunk:
                                continue
                            if len(head) < WAV_HEADER:
                                head += chunk
                                if len(head) < WAV_HEADER:
                                    continue
                                chunk = head[WAV_HEADER:]
                                if not chunk:
                                    continue
                            yield chunk
            except Exception as e:  # noqa: BLE001
                print(f"[stream/gguf] 中断: {type(e).__name__}: {e}", flush=True)

        return StreamingResponse(
            gen_gguf(),
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(sr), "X-Channels": "1",
                     "X-Engine": "gguf", "Cache-Control": "no-store"},
        )

    if engine == "mlx":
        info = _mlx_probe() or {}
        sr = int(info.get("sample_rate") or 48000)
        body = _mlx_body(req, text, ref)

        def gen_mlx():
            try:
                with httpx.Client(timeout=None) as c:
                    with c.stream("POST", f"{MLX_URL}/stream", json=body) as r:
                        if r.status_code != 200:
                            print(f"[stream/mlx] worker 返回 {r.status_code}", flush=True)
                            return
                        for chunk in r.iter_raw():
                            if chunk:
                                yield chunk
            except Exception as e:  # noqa: BLE001
                print(f"[stream/mlx] 中断: {type(e).__name__}: {e}", flush=True)

        return StreamingResponse(
            gen_mlx(),
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(sr), "X-Channels": "1",
                     "X-Engine": "mlx", "Cache-Control": "no-store"},
        )

    _require_ready()
    sr = state["sr"]

    def gen():
        import torch

        model = state["model"]
        state["busy"] = True
        try:
            with GEN_LOCK:
                torch.manual_seed(req.seed)
                kw = {}
                if ref:
                    kw["reference_wav_path"] = ref
                if req.prompt_text and ref:
                    kw["prompt_text"] = req.prompt_text
                    kw["prompt_wav_path"] = ref
                for chunk in model.generate_streaming(
                    text=text, cfg_value=req.cfg, inference_timesteps=req.steps, **kw
                ):
                    yield _to_i16(chunk)
        except Exception as e:  # noqa: BLE001
            print(f"[stream] 生成中断: {type(e).__name__}: {e}", flush=True)
        finally:
            state["busy"] = False

    return StreamingResponse(
        gen(),
        # 用通用二进制类型：audio/* 会被 Chrome 的媒体加载器接管并主动 abort 连接
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(sr), "X-Channels": "1",
                 "X-Engine": "torch", "Cache-Control": "no-store"},
    )


@app.post("/api/ref")
async def upload_ref(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(413, "参考音频不能超过 30MB")
    rid = uuid.uuid4().hex[:12]
    tmp = REFS / f"{rid}.upload"
    tmp.write_bytes(raw)
    try:
        data, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "无法解析该音频，请上传 wav/mp3/flac")
    if data.ndim > 1:
        data = data.mean(axis=1)
    out = REFS / f"{rid}.wav"
    sf.write(out, data, sr)
    tmp.unlink(missing_ok=True)
    return {"ref_id": rid, "duration": round(len(data) / sr, 2), "sr": sr}


@app.get("/api/audio/{name}")
async def audio(name: str):
    if not name.endswith(".wav") or "/" in name:
        raise HTTPException(400, "非法文件名")
    p = RUNS / name
    if not p.exists():
        raise HTTPException(404, "音频不存在")
    return FileResponse(p, media_type="audio/wav", filename=name)


@app.get("/api/history")
async def history(limit: int = 30):
    items = []
    for f in sorted(RUNS.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        try:
            items.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


@app.delete("/api/history/{aid}")
async def drop(aid: str):
    for ext in (".wav", ".json"):
        (RUNS / f"{aid}{ext}").unlink(missing_ok=True)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8817)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
