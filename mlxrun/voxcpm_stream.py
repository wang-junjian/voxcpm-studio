"""VoxCPM2 × MLX 8bit 真流式合成。

mlx_audio 自带的 `generate()` 是「假流式」——整段跑完才 yield 一次
（voxcpm2.py 全文只有一个 yield，在第 700 行）。Web 端的体感由首包延迟决定，
所以这里把它的自回归循环搬出来，每攒够 chunk_patches 个 patch 就解码一次
并立刻 yield。

分块解码之所以成立，是因为解码器是纯因果卷积：每次解码多带若干帧左上下文，
丢掉上下文对应的输出即可。实测 context_frames=16 时与整段解码的 SNR 为
116 dB（最大差 1e-5），人耳不可分；代价是解码从 0.41s 涨到 0.71s，而解码
只占总耗时 5%，净增约 3%。

与 `model.generate()` 的逐样本等价性由同一个随机种子保证：DiT 采样走
`mx.random.normal`，其余路径（FSQ、Euler 求解、stop head）全确定。
"""
from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# 解码器每帧 latent 展开的采样点数：48 kHz / 25 fps
SAMPLES_PER_FRAME = 1920
# 左上下文帧数，见模块 docstring 的 SNR 实测
DEFAULT_CONTEXT_FRAMES = 16


class LiteTokenizer:
    """只实现 VoxCPM2 用到的两个方法：tokenize / convert_tokens_to_ids。

    为什么不用 transformers：本机 `import transformers` 要 30–60 秒，全花在
    LazyModule 初始化上（`AutoTokenizer.from_pretrained` 本身只要 0.45 秒）。
    worker 每次重启都要付这笔钱，代价过高。tokenizers（Rust 后端）直接读
    tokenizer.json 只要 0.06 秒，且在 9 组中英混排 / 标点 / emoji / 数字用例上
    与本模型的 LlamaTokenizer token id 完全一致；唯一差异是"文本以空格开头"
    时的首个 token，而 Web 端上游已经 strip 过。
    """

    def __init__(self, tokenizer_json: Path):
        from tokenizers import Tokenizer

        self._tk = Tokenizer.from_file(str(tokenizer_json))
        self._unk = self._tk.token_to_id("<unk>")
        if self._unk is None:
            raise ValueError("tokenizer.json 缺少 <unk>")
        if not self._tk.encode("你好", add_special_tokens=False).ids:
            raise ValueError("tokenizer.json 无法编码中文")

    def tokenize(self, text: str) -> list[str]:
        return self._tk.encode(text, add_special_tokens=False).tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        to_id = self._tk.token_to_id
        unk = self._unk
        return [unk if (i := to_id(t)) is None else i for t in tokens]


def _disable_hf_tokenizer_hook() -> bool:
    """让 mlx_audio 的 loader 跳过 post_load_hook 里的 AutoTokenizer 构建。"""
    try:
        from mlx_audio.tts.models.voxcpm2.voxcpm2 import Model

        if getattr(Model, "_lite_tokenizer_patched", False):
            return True
        Model.post_load_hook = classmethod(lambda cls, model, path: model)
        Model._lite_tokenizer_patched = True
        return True
    except Exception:  # noqa: BLE001
        return False


def load_model_lite(model_dir: str | Path, force_hf_tokenizer: bool = False):
    """加载 MLX 模型并用 LiteTokenizer 接管分词。

    返回 (model, info)，info 里带加载耗时与分词器来源，方便 worker 打印。

    实测（M2 Max / 冷启动）：用 HF tokenizer 时 load() 68 秒（其中 67.9 秒是
    post_load_hook 里的 `import transformers`）；换成 LiteTokenizer 后 1.3 秒。
    """
    from mlx_audio.tts.utils import load

    path = Path(model_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"模型目录不存在: {path}")

    t0 = time.perf_counter()
    patched = False if force_hf_tokenizer else _disable_hf_tokenizer_hook()
    model = load(str(path))
    load_s = time.perf_counter() - t0

    using = "hf"
    if patched:
        try:
            model.tokenizer = LiteTokenizer(path / "tokenizer.json")
            using = "lite"
        except Exception as e:  # noqa: BLE001
            # 退回 HF：虽然慢，但至少不会起不来
            from transformers import AutoTokenizer

            model.tokenizer = AutoTokenizer.from_pretrained(str(path))
            using = f"hf(fallback: {e})"

    info = {
        "load_s": round(load_s, 2),
        "tokenizer": using,
        "sample_rate": int(model.sample_rate),
        "model_dir": str(path),
    }
    return model, info


class ChunkDecoder:
    """带左上下文的增量 VAE 解码。

    VAE 解码器全部由 causal conv 组成，所以位置 t 的输出只依赖 t 及之前
    的 latent。分块解码时给每块补上 context_frames 帧历史，再把这段历史
    对应的输出切掉，得到与整段解码一致的结果。
    """

    def __init__(self, model, context_frames: int = DEFAULT_CONTEXT_FRAMES):
        self.av = model.audio_vae
        self.spf = int(getattr(self.av, "decode_chunk_size", SAMPLES_PER_FRAME))
        self.context_frames = max(0, int(context_frames))
        self.tail = None  # 最近 context_frames 帧 latent，形状 (<=C, D)
        self.decode_s = 0.0

    def __call__(self, patches) -> np.ndarray:
        new = patches if isinstance(patches, mx.array) else mx.concatenate(patches, axis=0)
        n_new = int(new.shape[0])
        z = new if self.tail is None else mx.concatenate([self.tail, new], axis=0)

        t0 = time.perf_counter()
        y = self.av.decode(z[None, ...])[0]
        mx.eval(y)
        self.decode_s += time.perf_counter() - t0

        if self.context_frames:
            self.tail = z[-self.context_frames:]

        audio = np.asarray(y, dtype=np.float32).reshape(-1)
        drop = (int(z.shape[0]) - n_new) * self.spf
        return audio[drop: drop + n_new * self.spf]


def stream_generate(
    model,
    text: str,
    *,
    ref_audio=None,
    prompt_text=None,
    prompt_audio=None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    max_tokens: int = 2000,
    min_tokens: int = 2,
    warmup_patches: int = 0,
    chunk_patches: int = 8,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    seed: int | None = None,
    eval_each_step: bool = True,
    stats: dict | None = None,
):
    """逐块 yield float32 单声道音频。

    参数与 mlx_audio 的 `VoxCPM2.generate()` 对齐，另加分块相关的
    chunk_patches / context_frames，以及可选的 stats 回填。

    Args:
        ref_audio: 参考音频（路径字符串、mx.array 或 numpy 数组）。
        prompt_text + prompt_audio: 续写模式的提示文本与音频。
        chunk_patches: 每攒够多少个 patch 解码并吐一次。1 patch = 160 ms 音频。
        context_frames: 分块解码的左上下文帧数。
        seed: 随机种子；给了就固定，保证与一次性生成可逐样本比对。
        eval_each_step: 每步都把新状态求值一次。MLX 是惰性求值，循环里只靠
            stop head 的 `.item()` 同步时，`curr_embed → lm_cache → residual`
            这条支路的计算图会一路累积到最后一次解码才展开；整段文本越长，
            展开时的瞬时峰值越大（实测 21 秒音频多占 7.6 GB）。每步求值只
            多几次同步开销，峰值降到常数级——这是长文本不炸内存的前提。
        stats: 生成结束后回填首包延迟、解码耗时、采样点数等。
    """
    if model.tokenizer is None:
        raise RuntimeError("tokenizer 未加载，无法分词")
    if not isinstance(text, str):
        raise TypeError(f"text 必须是字符串，收到 {type(text)}")
    if seed is not None:
        mx.random.seed(int(seed))

    scale_emb = model.args.lm_config.scale_emb if model.args.lm_config.use_mup else 1.0
    latent_dim = model.audio_vae.latent_dim
    patch = model.patch_size

    t_start = time.perf_counter()

    has_ref = ref_audio is not None
    has_prompt = prompt_audio is not None and prompt_text is not None

    # ---- 构造前缀，四种模式与 mlx_audio 的 generate() 完全一致 ----
    if has_ref and has_prompt:
        combined_text = prompt_text + text
        text_ids = model._tokenize(combined_text)
        text_token = mx.array(text_ids + [model.audio_start_token], dtype=mx.int32)
        text_length = text_token.shape[0]

        ref_feat = model._encode_wav(ref_audio, padding_mode="right")
        prompt_feat = model._encode_wav(prompt_audio, padding_mode="left")
        prompt_audio_length = prompt_feat.shape[0]

        ref_tokens, ref_feats, ref_t_mask, ref_a_mask = model._make_ref_prefix(ref_feat)

        text_pad_feat = mx.zeros((text_length, patch, latent_dim))
        prompt_pad_token = mx.zeros(prompt_audio_length, dtype=mx.int32)

        text_token = mx.concatenate([ref_tokens, text_token, prompt_pad_token])
        audio_feat = mx.concatenate([ref_feats, text_pad_feat, prompt_feat], axis=0)
        text_mask = mx.concatenate(
            [ref_t_mask, mx.ones(text_length, dtype=mx.float32),
             mx.zeros(prompt_audio_length, dtype=mx.float32)]
        )
        audio_mask = mx.concatenate(
            [ref_a_mask, mx.zeros(text_length, dtype=mx.float32),
             mx.ones(prompt_audio_length, dtype=mx.float32)]
        )

    elif has_ref:
        text_ids = model._tokenize(text)
        text_token = mx.array(text_ids + [model.audio_start_token], dtype=mx.int32)
        text_length = text_token.shape[0]

        ref_feat = model._encode_wav(ref_audio, padding_mode="right")
        ref_tokens, ref_feats, ref_t_mask, ref_a_mask = model._make_ref_prefix(ref_feat)
        text_pad_feat = mx.zeros((text_length, patch, latent_dim))

        text_token = mx.concatenate([ref_tokens, text_token])
        audio_feat = mx.concatenate([ref_feats, text_pad_feat], axis=0)
        text_mask = mx.concatenate([ref_t_mask, mx.ones(text_length, dtype=mx.float32)])
        audio_mask = mx.concatenate([ref_a_mask, mx.zeros(text_length, dtype=mx.float32)])

    elif has_prompt:
        combined_text = prompt_text + text
        text_ids = model._tokenize(combined_text)
        text_token = mx.array(text_ids + [model.audio_start_token], dtype=mx.int32)
        text_length = text_token.shape[0]

        prompt_feat = model._encode_wav(prompt_audio, padding_mode="left")
        prompt_audio_length = prompt_feat.shape[0]

        text_pad_feat = mx.zeros((text_length, patch, latent_dim))
        prompt_pad_token = mx.zeros(prompt_audio_length, dtype=mx.int32)

        text_token = mx.concatenate([text_token, prompt_pad_token])
        audio_feat = mx.concatenate([text_pad_feat, prompt_feat], axis=0)
        text_mask = mx.concatenate(
            [mx.ones(text_length, dtype=mx.float32),
             mx.zeros(prompt_audio_length, dtype=mx.float32)]
        )
        audio_mask = mx.concatenate(
            [mx.zeros(text_length, dtype=mx.float32),
             mx.ones(prompt_audio_length, dtype=mx.float32)]
        )

    else:
        text_ids = model._tokenize(text)
        text_token = mx.array(text_ids + [model.audio_start_token], dtype=mx.int32)
        text_length = text_token.shape[0]

        audio_feat = mx.zeros((text_length, patch, latent_dim))
        text_mask = mx.ones(text_length, dtype=mx.float32)
        audio_mask = mx.zeros(text_length, dtype=mx.float32)

    token_count = len(text_ids)

    text_token = text_token[None, :]
    audio_feat = audio_feat[None, :, :, :]
    text_mask = text_mask[None, :]
    audio_mask = audio_mask[None, :]

    feat_embed = model.enc_to_lm_proj(model.feat_encoder(audio_feat))
    text_embed = model.base_lm.embed_tokens(text_token) * scale_emb
    combined_embed = (
        text_mask[:, :, None] * text_embed + audio_mask[:, :, None] * feat_embed
    )

    prefix_feat_cond = audio_feat[:, -1, :, :]

    enc_outputs, lm_cache = model.base_lm(combined_embed)
    enc_outputs = (
        model.fsq_layer(enc_outputs) * audio_mask[:, :, None]
        + enc_outputs * text_mask[:, :, None]
    )
    lm_hidden = enc_outputs[:, -1, :]

    residual_input = model.fusion_concat_proj(
        mx.concatenate([enc_outputs, audio_mask[:, :, None] * feat_embed], axis=-1)
    )
    residual_outputs, res_cache = model.residual_lm(residual_input)
    residual_hidden = residual_outputs[:, -1, :]

    # 前缀阶段一次性算完，避免整张计算图被拖进自回归循环里
    mx.eval(lm_hidden, residual_hidden)

    decode = ChunkDecoder(model, context_frames)
    buf = []          # 待解码的 patch，每个 (P, D)
    n_chunks = 0
    n_samples = 0
    t_first = None

    for i in range(max_tokens + warmup_patches):
        # V2 的 DiT 输入是拼接而非相加
        dit_h = mx.concatenate(
            [model.lm_to_dit_proj(lm_hidden), model.res_to_dit_proj(residual_hidden)],
            axis=-1,
        )
        pred_feat = model.feat_decoder.sample(
            mu=dit_h,
            n_timesteps=inference_timesteps,
            patch_size=patch,
            cond=prefix_feat_cond.transpose(0, 2, 1),
            cfg_value=cfg_value,
        )
        pred_feat = pred_feat.transpose(0, 2, 1)  # (1, P, D)

        # 预热 patch 只用于建立条件，不进音频
        if i >= warmup_patches:
            buf.append(pred_feat[0])

        curr_embed = model.enc_to_lm_proj(model.feat_encoder(pred_feat[:, None, :, :]))

        stop_logits = model.stop_head(nn.silu(model.stop_proj(lm_hidden)))
        stop_flag = int(mx.argmax(stop_logits, axis=-1).item())
        if (i - warmup_patches) > min_tokens and stop_flag == 1:
            break

        new_lm_out, lm_cache = model.base_lm(inputs_embeds=curr_embed, cache=lm_cache)
        lm_hidden = model.fsq_layer(new_lm_out[:, -1, :])

        curr_residual_input = model.fusion_concat_proj(
            mx.concatenate([lm_hidden[:, None, :], curr_embed], axis=-1)
        )
        new_res_out, res_cache = model.residual_lm(
            inputs_embeds=curr_residual_input, cache=res_cache
        )
        residual_hidden = new_res_out[:, -1, :]

        # 关键：把这一步的状态真正算出来。否则惰性图会跨越整个循环累积，
        # 长文本在最后一次解码时同时展开，瞬时峰值随文本长度线性增长。
        # 求值 lm_hidden / residual_hidden 会顺带把 lm_cache、res_cache 与
        # curr_embed 全部落地，数值不变。
        if eval_each_step:
            mx.eval(lm_hidden, residual_hidden)

        prefix_feat_cond = pred_feat

        if len(buf) >= chunk_patches:
            audio = decode(buf)
            buf = []
            if audio.size:
                n_chunks += 1
                n_samples += audio.size
                if t_first is None:
                    t_first = time.perf_counter() - t_start
                yield audio

    if buf:
        audio = decode(buf)
        if audio.size:
            n_chunks += 1
            n_samples += audio.size
            if t_first is None:
                t_first = time.perf_counter() - t_start
            yield audio

    if stats is not None:
        sr = int(model.sample_rate)
        gen_s = time.perf_counter() - t_start
        stats.update(
            {
                "sample_rate": sr,
                "samples": n_samples,
                "audio_s": round(n_samples / sr, 3),
                "gen_s": round(gen_s, 3),
                "first_chunk_s": round(t_first, 3) if t_first is not None else None,
                "chunks": n_chunks,
                "decode_s": round(decode.decode_s, 3),
                "rtf": round(gen_s / max(n_samples / sr, 1e-9), 3),
                "token_count": token_count,
                "chunk_patches": chunk_patches,
            }
        )
