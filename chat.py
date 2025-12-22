#!/usr/bin/env python3
"""
chat_v29.py

Tiny interactive console REPL for sampling from v29 checkpoints.

Why this exists:
- v29 already supports `--mode sample`, but that reloads/initializes every run.
- This script loads the checkpoint once, then lets you type prompts repeatedly.

Usage:
  python chat_v29.py --ckpt runs/oneb_ctx4k_20b_decoupled_seed1337/last.pt --device cuda

Notes:
- Requires `tiktoken` for text prompts (GPT-2 BPE).
- Commands:
    :q / :quit   quit
    :reset       clear conversation history
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from dataclasses import asdict
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from production.config import pick_device
from production.kvcache_backend import KVCacheKind, KVCacheTensorConfig
from production.model import DecoupledLayerKVCache, GPT, LayerKVCache, ModelConfig
from production.selfopt_cache import as_object_list, as_str_object_dict


@runtime_checkable
class _Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


@runtime_checkable
class _EncoderOrd(Protocol):
    # Optional fast path present in some tiktoken versions.
    def encode_ordinary(self, text: str) -> list[int]: ...


def _tiktoken_encoder() -> _Encoder | None:
    # Keep `tiktoken` truly optional at import time.
    if importlib.util.find_spec("tiktoken") is None:
        return None
    try:
        mod = importlib.import_module("tiktoken")
    except ImportError:
        return None
    get_enc = getattr(mod, "get_encoding", None)
    if not callable(get_enc):
        return None
    try:
        enc_obj = get_enc("gpt2")
    except (TypeError, ValueError):
        return None
    return enc_obj if isinstance(enc_obj, _Encoder) else None


def _torch_load_obj(path: str, *, device: torch.device) -> object:
    # `torch.load` is typed as returning Any; isolate behind an object boundary.
    return torch.load(str(path), map_location=device)  # pyright: ignore[reportAny]


def _as_state_dict(o: object) -> dict[str, torch.Tensor] | None:
    m = as_str_object_dict(o)
    if m is None:
        return None
    out: dict[str, torch.Tensor] = {}
    for k, v in m.items():
        if not isinstance(v, torch.Tensor):
            return None
        out[str(k)] = v
    return out


def _dtype_from_state_dict(sd: dict[str, torch.Tensor]) -> torch.dtype | None:
    for v in sd.values():
        return v.dtype
    return None


def _encode(enc: _Encoder, text: str) -> list[int]:
    # tiktoken has both `encode` and `encode_ordinary` depending on version.
    if isinstance(enc, _EncoderOrd):
        return enc.encode_ordinary(text)
    return enc.encode(text)


def _neg_inf(dtype: torch.dtype) -> float:
    try:
        return float(torch.finfo(dtype).min)
    except (TypeError, ValueError):
        return -1e9


def _endswith(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    if not needle:
        return False
    if len(haystack) < len(needle):
        return False
    return list(haystack[-len(needle) :]) == list(needle)


def _make_kv_cfg(
    *,
    kind: KVCacheKind,
    qblock: int,
    residual_len: int,
) -> KVCacheTensorConfig:
    # residual window only applies to quantized caches
    residual = int(residual_len) if kind not in ("fp16", "fp32") else 0
    return KVCacheTensorConfig(kind=kind, qblock=int(qblock), residual_len=int(residual))


def _as_kv_kind(s: str) -> KVCacheKind:
    v = str(s).strip().lower()
    if v == "fp32":
        return "fp32"
    if v == "q8_0":
        return "q8_0"
    if v == "q4_0":
        return "q4_0"
    if v == "nf4":
        return "nf4"
    return "fp16"


def _args_map(ns: argparse.Namespace) -> dict[str, object]:
    d = as_str_object_dict(getattr(ns, "__dict__", {}))
    return {} if d is None else d


def _as_int(o: object, default: int) -> int:
    try:
        return int(str(o))
    except (TypeError, ValueError):
        return int(default)


def _as_float(o: object, default: float) -> float:
    try:
        return float(str(o))
    except (TypeError, ValueError):
        return float(default)


def _as_bool(o: object, default: bool) -> bool:
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, int):
        return bool(o != 0)
    if isinstance(o, str):
        s = o.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
    return bool(default)


def _stream_generate(
    *,
    model: GPT,
    cfg: ModelConfig,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    kv_cache: KVCacheKind,
    kv_qblock: int,
    kv_residual: int,
    kv_decode_block: int,
    kv_fused: str,
):
    """
    Stream tokens using the same KV-cache decode loop as v29's GPT.generate().
    Yields token IDs (ints) as they are sampled.
    """
    _ = kv_decode_block  # metadata only (doesn't affect allocation)
    was_training = bool(model.training)
    _ = model.eval()
    device = prompt.device
    B, T0 = prompt.shape
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        return
    # Don't exceed model context window.
    max_new_tokens = min(max_new_tokens, int(cfg.block_size) - int(T0))
    if max_new_tokens <= 0:
        return

    if kv_fused not in ("none", "auto", "triton1pass", "triton2pass"):
        raise ValueError("kv_fused must be one of: none, auto, triton1pass, triton2pass")

    max_seq = int(T0) + int(max_new_tokens)

    # Match v29's hetero default for decoupled when kv_cache is q4_0.
    kv_cache_k: KVCacheKind | None = None
    kv_cache_v: KVCacheKind | None = None
    kv_cache_k_sem: KVCacheKind | None = None
    kv_cache_k_geo: KVCacheKind | None = None
    if str(cfg.attn_mode) == "decoupled" and kv_cache == "q4_0":
        kv_cache_k_geo = "q8_0"
        kv_cache_k_sem = "q4_0"
        kv_cache_v = "q4_0"

    # Default K/V configs (standard/bottleneck/gqa)
    k_cfg = _make_kv_cfg(kind=(kv_cache_k or kv_cache), qblock=kv_qblock, residual_len=kv_residual)
    v_cfg = _make_kv_cfg(kind=(kv_cache_v or kv_cache), qblock=kv_qblock, residual_len=kv_residual)

    # Decoupled configs
    k_sem_cfg = _make_kv_cfg(kind=(kv_cache_k_sem or kv_cache), qblock=kv_qblock, residual_len=kv_residual)
    k_geo_cfg = _make_kv_cfg(kind=(kv_cache_k_geo or kv_cache), qblock=kv_qblock, residual_len=kv_residual)
    v_dec_cfg = _make_kv_cfg(kind=(kv_cache_v or kv_cache), qblock=kv_qblock, residual_len=kv_residual)

    caches: list[DecoupledLayerKVCache | LayerKVCache] = []
    for _ in range(int(cfg.n_layer)):
        if str(cfg.attn_mode) == "decoupled":
            c = DecoupledLayerKVCache(
                batch_size=int(B),
                max_seq_len=int(max_seq),
                k_sem_dim=int(cfg.sem_dim),
                k_geo_dim=int(cfg.geo_dim),
                v_dim=int(cfg.attn_dim),
                k_sem_cfg=k_sem_cfg,
                k_geo_cfg=k_geo_cfg,
                v_cfg=v_dec_cfg,
                device=device,
            )
            caches.append(c)
        else:
            if str(cfg.attn_mode) == "standard":
                k_dim = v_dim = int(cfg.d_model)
            elif str(cfg.attn_mode) == "bottleneck":
                k_dim = v_dim = int(cfg.attn_dim)
            elif str(cfg.attn_mode) == "gqa":
                head_dim = int(cfg.attn_dim) // int(cfg.n_head)
                kvh = cfg.kv_head
                kv_head = int(kvh) if kvh is not None else int(cfg.n_head)
                k_dim = v_dim = int(kv_head * head_dim)
            else:
                raise ValueError(f"Unknown attn_mode for KV cache: {cfg.attn_mode}")

            c = LayerKVCache(
                batch_size=int(B),
                max_seq_len=int(max_seq),
                k_dim=int(k_dim),
                v_dim=int(v_dim),
                k_cfg=k_cfg,
                v_cfg=v_cfg,
                device=device,
            )
            caches.append(c)

    # Prefill (fills caches)
    with torch.inference_mode():
        logits, caches_out = model.forward(prompt, caches=caches, pos_offset=0)
    if caches_out is None:
        raise RuntimeError("Model returned no KV caches during prefill")
    caches = caches_out
    out = prompt

    for _ in range(int(max_new_tokens)):
        next_logits = logits[:, -1, :] / max(float(temperature), 1e-8)
        if top_k is not None:
            top_k_i = int(top_k)
            v, _ = torch.topk(next_logits, min(top_k_i, next_logits.size(-1)))
            next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], _neg_inf(next_logits.dtype))
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
        out = torch.cat([out, next_id], dim=1)

        yield int(next_id[0, 0].item())

        # decode one token (updates caches)
        with torch.inference_mode():
            logits, caches2 = model.forward(next_id, caches=caches, pos_offset=out.size(1) - 1)
        if caches2 is not None:
            caches = caches2

    if was_training:
        _ = model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pt / last.pt / step*.pt).")
    _ = ap.add_argument("--device", type=str, default=None, help="cuda / cpu / mps. Default: auto.")
    _ = ap.add_argument("--max-new-tokens", type=int, default=200)
    _ = ap.add_argument("--temperature", type=float, default=0.9)
    _ = ap.add_argument("--top-k", type=int, default=50)

    # KV cache knobs (match v29 defaults/choices)
    _ = ap.add_argument("--kv-cache", type=str, default="fp16", choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"])
    _ = ap.add_argument("--kv-qblock", type=int, default=32)
    _ = ap.add_argument("--kv-residual", type=int, default=128)
    _ = ap.add_argument("--kv-decode-block", type=int, default=1024)
    _ = ap.add_argument("--kv-fused", type=str, default="auto", choices=["none", "auto", "triton1pass", "triton2pass"])

    _ = ap.add_argument("--system", type=str, default="", help="Optional system/prefix string prepended once.")
    _ = ap.add_argument(
        "--style",
        type=str,
        default="chat",
        choices=["chat", "chat_fewshot", "raw"],
        help=(
            "Prompt style. 'chat' uses User/Assistant prefixes. "
            "'chat_fewshot' also prepends a tiny in-context example (helps base LMs). "
            "'raw' sends your input as-is."
        ),
    )
    _ = ap.add_argument("--no-history", action="store_true", help="Do not keep conversation history (each prompt is independent).")
    _ = ap.add_argument("--print-config", action="store_true", help="Print the ModelConfig loaded from checkpoint then continue.")
    _ = ap.add_argument("--no-stream", action="store_true", help="Disable streaming; print the full completion at once.")
    _ = ap.add_argument("--chat", action="store_true", default=True, help="Use a simple chat template (User/Assistant). Default: on.")
    _ = ap.add_argument("--no-chat", dest="chat", action="store_false", help="Disable chat template; treat your input as a raw prompt.")
    _ = ap.add_argument("--user-prefix", type=str, default="User:", help="Chat template: user prefix.")
    _ = ap.add_argument("--assistant-prefix", type=str, default="Assistant:", help="Chat template: assistant prefix.")
    _ = ap.add_argument(
        "--stop-seq",
        type=str,
        action="append",
        default=None,
        help="Stop sequence (text) that ends assistant generation. Can be provided multiple times. Default: '\\nUser:'.",
    )
    args = ap.parse_args()
    a = _args_map(args)

    enc = _tiktoken_encoder()
    if enc is None:
        msg = (
            "tiktoken is required for chat.py. Install it (`pip install tiktoken`) "
            "or use `production/runner_sample.py` for token-only workflows."
        )
        raise SystemExit(msg)

    dev_s = a.get("device", None)
    device = pick_device(str(dev_s) if isinstance(dev_s, str) and dev_s else None)
    print(f"[chat_v29] device={device}", file=sys.stderr)
    ckpt_path = str(a.get("ckpt", "") or "")
    print(f"[chat_v29] loading ckpt={ckpt_path}", file=sys.stderr)

    try:
        ckpt_obj = _torch_load_obj(ckpt_path, device=device)
    except PermissionError as e:
        # On macOS, protected folders (Downloads/Desktop/Documents) are guarded by TCC privacy controls.
        # Python itself can be blocked from opening those paths even if `ls` works in the shell.
        msg = "".join(
            [
                "PermissionError while reading checkpoint.\n",
                "Likely macOS Privacy/TCC blocking Python from accessing a protected folder (e.g. ~/Downloads).\n\n",
                "Fastest fix (no copying needed): MOVE the file out of Downloads, then retry:\n",
                "  mv ~/Downloads/step5000.pt /Users/theapemachine/go/src/github.com/theapemachine/experiments/\n",
                "  python3.12 chat.py --ckpt ./step5000.pt --device mps\n\n",
                "Permanent fix:\n",
                "  System Settings → Privacy & Security → Files and Folders (or Full Disk Access)\n",
                "  and allow access for your Terminal and/or python3.12.\n\n",
                f"Original error: {e}",
            ]
        )
        raise SystemExit(msg) from e
    ckpt = as_str_object_dict(ckpt_obj)
    if ckpt is None:
        raise SystemExit("Checkpoint payload must be a dict-like object.")
    cfg_dict_obj = ckpt.get("config", None)
    cfg_dict = as_str_object_dict(cfg_dict_obj)
    if cfg_dict is None:
        raise SystemExit("Checkpoint missing 'config' (can't reconstruct model safely).")

    cfg = ModelConfig.from_dict(cfg_dict, device=device)
    if _as_bool(a.get("print_config", False), False):
        print(asdict(cfg))

    sd_obj = ckpt.get("model", None)
    sd = _as_state_dict(sd_obj)
    if sd is None:
        raise SystemExit("Checkpoint missing 'model' state_dict.")

    model = GPT(cfg)
    # Match model dtype to checkpoint tensors when possible (helps avoid dtype mismatch surprises).
    sd_dtype = _dtype_from_state_dict(sd)
    if sd_dtype is not None:
        try:
            model = model.to(dtype=sd_dtype)
        except (TypeError, ValueError, RuntimeError):
            pass
    model = model.to(device)
    _ = model.load_state_dict(sd, strict=True)
    _ = model.eval()

    # Conversation state (token IDs). We'll keep exact history tokens to avoid re-encoding everything.
    history_ids: list[int] = []
    system_s = str(a.get("system", "") or "")
    if system_s.strip():
        history_ids = _encode(enc, system_s.strip() + "\n")

    # Back-compat: --style overrides --chat flags.
    style_s = str(a.get("style", "chat") or "chat")
    chat_enabled = _as_bool(a.get("chat", True), True)
    if style_s == "raw":
        chat_enabled = False
    elif style_s in ("chat", "chat_fewshot"):
        chat_enabled = True

    # In-context example (helps non-instruction-tuned models follow the format).
    user_prefix = str(a.get("user_prefix", "User:") or "User:")
    assistant_prefix = str(a.get("assistant_prefix", "Assistant:") or "Assistant:")
    if style_s == "chat_fewshot":
        # Keep it short so it doesn't steal too much context.
        demo = (
            f"{user_prefix} Say hello in one sentence.\n"
            f"{assistant_prefix} Hello! How can I help?\n"
            f"{user_prefix} Answer concisely: what is RoPE?\n"
            f"{assistant_prefix} RoPE (rotary positional embeddings) encodes position by rotating Q/K features as a function of token index.\n"
        )
        history_ids = history_ids + _encode(enc, demo)

    # Default stop: when model begins the next user turn.
    stop_texts: list[str] = []
    stop_seq_obj = a.get("stop_seq", None)
    stop_seq_list = as_object_list(stop_seq_obj) if stop_seq_obj is not None else None
    if stop_seq_list is not None:
        for s_obj in stop_seq_list:
            s = str(s_obj)
            if s:
                stop_texts.append(s)
    else:
        stop_texts = ["\n" + user_prefix]
    stop_token_seqs: list[tuple[str, list[int]]] = []
    for s in stop_texts:
        stop_token_seqs.append((s, _encode(enc, s)))
    max_stop_len = max((len(toks) for (_s, toks) in stop_token_seqs), default=0)

    print("\nEnter text; commands: :q / :quit, :reset\n", file=sys.stderr)
    while True:
        try:
            user = input("> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\n[chat_v29] bye", file=sys.stderr)
            return

        if not user:
            continue
        if user in {":q", ":quit", "quit", "exit"}:
            print("[chat_v29] bye", file=sys.stderr)
            return
        if user == ":reset":
            history_ids = _encode(enc, system_s.strip() + "\n") if system_s.strip() else []
            print("[chat_v29] reset", file=sys.stderr)
            continue

        no_history = _as_bool(a.get("no_history", False), False)
        no_stream = _as_bool(a.get("no_stream", False), False)
        if chat_enabled:
            # Chat template: we always prompt as:
            #   User: ...
            #   Assistant:
            user_turn = f"{user_prefix} {user}\n{assistant_prefix} "
        else:
            user_turn = user
            if not user_turn.endswith("\n"):
                user_turn += "\n"

        prompt_ids = _encode(enc, user_turn)
        if no_history:
            cur_ids = (history_ids + prompt_ids) if history_ids else prompt_ids
        else:
            history_ids = history_ids + prompt_ids
            cur_ids = history_ids

        # Ensure prompt fits in context window.
        if len(cur_ids) > int(cfg.block_size):
            cur_ids = cur_ids[-int(cfg.block_size) :]
            if not no_history:
                history_ids = cur_ids

        x = torch.tensor([cur_ids], dtype=torch.long, device=device)
        out_ids = cur_ids[:]
        max_new_tokens = _as_int(a.get("max_new_tokens", 200), 200)
        temperature = _as_float(a.get("temperature", 0.9), 0.9)
        top_k_raw = a.get("top_k", 50)
        top_k = _as_int(top_k_raw, 50)
        kv_cache = _as_kv_kind(str(a.get("kv_cache", "fp16") or "fp16"))
        kv_qblock = _as_int(a.get("kv_qblock", 32), 32)
        kv_residual = _as_int(a.get("kv_residual", 128), 128)
        kv_decode_block = _as_int(a.get("kv_decode_block", 1024), 1024)
        kv_fused = str(a.get("kv_fused", "auto") or "auto")

        if no_stream:
            with torch.inference_mode():
                y = model.generate(
                    x,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=(int(top_k) if top_k > 0 else None),
                    kv_cache=kv_cache,
                    kv_qblock=int(kv_qblock),
                    kv_residual=int(kv_residual),
                    kv_decode_block=int(kv_decode_block),
                    kv_fused=str(kv_fused),
                )
            y0 = y[0]
            out_ids = [int(v.item()) for v in y0]
            new_ids = out_ids[len(cur_ids) :]
            text = enc.decode(new_ids) if new_ids else ""
            print(text, end="" if text.endswith("\n") else "\n")
        else:
            # Stream token pieces as they are produced.
            _ = sys.stdout.flush()
            pieces: list[int] = []
            pending: list[int] = []
            with torch.inference_mode():
                for tid in _stream_generate(
                    model=model,
                    cfg=cfg,
                    prompt=x,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=(int(top_k) if top_k > 0 else None),
                    kv_cache=kv_cache,
                    kv_qblock=int(kv_qblock),
                    kv_residual=int(kv_residual),
                    kv_decode_block=int(kv_decode_block),
                    kv_fused=str(kv_fused),
                ):
                    tid_i = int(tid)
                    pending.append(tid_i)

                    # Stop condition: detect stop token sequences in the *unflushed* tail
                    stop_hit = False
                    stop_len = 0
                    for _s, toks in stop_token_seqs:
                        if toks and _endswith(pending, toks):
                            stop_hit = True
                            stop_len = len(toks)
                            break
                    if stop_hit:
                        # Flush everything *before* the stop marker.
                        keep = pending[:-stop_len] if stop_len > 0 else pending
                        if keep:
                            _ = sys.stdout.write(enc.decode(keep))
                            _ = sys.stdout.flush()
                            pieces.extend(keep)
                        pending = []
                        break

                    # Token-based streaming without printing stop markers:
                    # keep a tail of max_stop_len tokens unflushed so we can detect stop sequences cleanly.
                    if max_stop_len > 0 and len(pending) > max_stop_len:
                        flush_n = len(pending) - max_stop_len
                        flush_tokens = pending[:flush_n]
                        pending = pending[flush_n:]
                        if flush_tokens:
                            _ = sys.stdout.write(enc.decode(flush_tokens))
                            _ = sys.stdout.flush()
                            pieces.extend(flush_tokens)

            # Flush any remaining tokens that we intentionally kept as tail.
            if pending:
                _ = sys.stdout.write(enc.decode(pending))
                _ = sys.stdout.flush()
                pieces.extend(pending)
                pending = []
            _ = sys.stdout.write("\n")
            _ = sys.stdout.flush()
            out_ids = out_ids + pieces

        if not no_history:
            # If using chat template, ensure assistant output is separated from the next prompt.
            if chat_enabled:
                out_ids = out_ids + _encode(enc, "\n")
            history_ids = out_ids
            # Keep history bounded to context window.
            if len(history_ids) > int(cfg.block_size):
                history_ids = history_ids[-int(cfg.block_size) :]


if __name__ == "__main__":
    main()


