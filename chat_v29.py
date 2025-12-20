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
import sys
from dataclasses import asdict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


def _pick_device(s: Optional[str]) -> torch.device:
    if s:
        return torch.device(str(s))
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is fine for small checkpoints; for 1B it will be slow.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_v29_objects():
    # Import lazily so `--help` is fast.
    from v29_transformer_decoupled_bottleneck_instrumented import (  # noqa: WPS433
        DecoupledLayerKVCache,
        GPT,
        KVCacheTensorConfig,
        LayerKVCache,
        ModelConfig,
    )

    return GPT, ModelConfig, LayerKVCache, DecoupledLayerKVCache, KVCacheTensorConfig


def _dtype_from_state_dict(sd: dict) -> Optional[torch.dtype]:
    for v in sd.values():
        if torch.is_tensor(v):
            return v.dtype
    return None


def _encode(enc, text: str) -> List[int]:
    # tiktoken has both `encode` and `encode_ordinary` depending on version.
    if hasattr(enc, "encode_ordinary"):
        return enc.encode_ordinary(text)
    return enc.encode(text)


def _neg_inf(dtype: torch.dtype) -> float:
    try:
        return float(torch.finfo(dtype).min)
    except Exception:
        return -1e9


def _endswith(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    if not needle:
        return False
    if len(haystack) < len(needle):
        return False
    return list(haystack[-len(needle) :]) == list(needle)


def _make_kv_cfg(
    KVCacheTensorConfig,
    *,
    kind: str,
    qblock: int,
    residual_len: int,
):
    # residual window only applies to quantized caches
    residual = int(residual_len) if kind not in ("fp16", "fp32") else 0
    return KVCacheTensorConfig(kind=str(kind), qblock=int(qblock), residual_len=int(residual))


@torch.no_grad()
def _stream_generate(
    *,
    model,
    cfg,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    kv_cache: str,
    kv_qblock: int,
    kv_residual: int,
    kv_decode_block: int,
    kv_fused: str,
    LayerKVCache,
    DecoupledLayerKVCache,
    KVCacheTensorConfig,
):
    """
    Stream tokens using the same KV-cache decode loop as v29's GPT.generate().
    Yields token IDs (ints) as they are sampled.
    """
    was_training = model.training
    model.eval()
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
    kv_cache_k = None
    kv_cache_v = None
    kv_cache_k_sem = None
    kv_cache_k_geo = None
    if str(cfg.attn_mode) == "decoupled" and str(kv_cache) == "q4_0":
        kv_cache_k_geo = "q8_0"
        kv_cache_k_sem = "q4_0"
        kv_cache_v = "q4_0"

    # Default K/V configs (standard/bottleneck/gqa)
    k_cfg = _make_kv_cfg(KVCacheTensorConfig, kind=str(kv_cache_k or kv_cache), qblock=kv_qblock, residual_len=kv_residual)
    v_cfg = _make_kv_cfg(KVCacheTensorConfig, kind=str(kv_cache_v or kv_cache), qblock=kv_qblock, residual_len=kv_residual)

    # Decoupled configs
    k_sem_cfg = _make_kv_cfg(KVCacheTensorConfig, kind=str(kv_cache_k_sem or kv_cache), qblock=kv_qblock, residual_len=kv_residual)
    k_geo_cfg = _make_kv_cfg(KVCacheTensorConfig, kind=str(kv_cache_k_geo or kv_cache), qblock=kv_qblock, residual_len=kv_residual)
    v_dec_cfg = _make_kv_cfg(KVCacheTensorConfig, kind=str(kv_cache_v or kv_cache), qblock=kv_qblock, residual_len=kv_residual)

    caches = []
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
            c.decode_block = int(kv_decode_block)
            c.fused = str(kv_fused)
            caches.append(c)
        else:
            if str(cfg.attn_mode) == "standard":
                k_dim = v_dim = int(cfg.d_model)
            elif str(cfg.attn_mode) == "bottleneck":
                k_dim = v_dim = int(cfg.attn_dim)
            elif str(cfg.attn_mode) == "gqa":
                head_dim = int(cfg.attn_dim) // int(cfg.n_head)
                kv_head = int(cfg.kv_head) if getattr(cfg, "kv_head", None) is not None else int(cfg.n_head)
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
            c.decode_block = int(kv_decode_block)
            c.fused = str(kv_fused)
            caches.append(c)

    # Prefill (fills caches)
    logits, caches = model(prompt, caches=caches, pos_offset=0)
    out = prompt

    for _ in range(int(max_new_tokens)):
        next_logits = logits[:, -1, :] / max(float(temperature), 1e-8)
        if top_k is not None:
            v, _ = torch.topk(next_logits, min(int(top_k), next_logits.size(-1)))
            next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], _neg_inf(next_logits.dtype))
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
        out = torch.cat([out, next_id], dim=1)

        yield int(next_id[0, 0].item())

        # decode one token (updates caches)
        logits, caches = model(next_id, caches=caches, pos_offset=out.size(1) - 1)

    if was_training:
        model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to v29 checkpoint (best.pt / last.pt / step*.pt).")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu / mps. Default: auto.")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)

    # KV cache knobs (match v29 defaults/choices)
    ap.add_argument("--kv-cache", type=str, default="fp16", choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"])
    ap.add_argument("--kv-qblock", type=int, default=32)
    ap.add_argument("--kv-residual", type=int, default=128)
    ap.add_argument("--kv-decode-block", type=int, default=1024)
    ap.add_argument("--kv-fused", type=str, default="auto", choices=["none", "auto", "triton1pass", "triton2pass"])

    ap.add_argument("--system", type=str, default="", help="Optional system/prefix string prepended once.")
    ap.add_argument(
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
    ap.add_argument("--no-history", action="store_true", help="Do not keep conversation history (each prompt is independent).")
    ap.add_argument("--print-config", action="store_true", help="Print the ModelConfig loaded from checkpoint then continue.")
    ap.add_argument("--no-stream", action="store_true", help="Disable streaming; print the full completion at once.")
    ap.add_argument("--chat", action="store_true", default=True, help="(compat) Use a simple chat template (User/Assistant). Default: on.")
    ap.add_argument("--no-chat", dest="chat", action="store_false", help="(compat) Disable chat template; treat your input as a raw prompt.")
    ap.add_argument("--user-prefix", type=str, default="User:", help="Chat template: user prefix.")
    ap.add_argument("--assistant-prefix", type=str, default="Assistant:", help="Chat template: assistant prefix.")
    ap.add_argument(
        "--stop-seq",
        type=str,
        action="append",
        default=None,
        help="Stop sequence (text) that ends assistant generation. Can be provided multiple times. Default: '\\nUser:'.",
    )
    args = ap.parse_args()

    if tiktoken is None:
        raise SystemExit("tiktoken is required for chat_v29.py. Install it (pip install tiktoken) or use v29 --mode sample with token IDs.")
    enc = tiktoken.get_encoding("gpt2")

    device = _pick_device(args.device)
    print(f"[chat_v29] device={device}", file=sys.stderr)
    print(f"[chat_v29] loading ckpt={args.ckpt}", file=sys.stderr)

    GPT, ModelConfig, LayerKVCache, DecoupledLayerKVCache, KVCacheTensorConfig = _load_v29_objects()
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
    except PermissionError as e:
        # On macOS, protected folders (Downloads/Desktop/Documents) are guarded by TCC privacy controls.
        # Python itself can be blocked from opening those paths even if `ls` works in the shell.
        raise SystemExit(
            "PermissionError while reading checkpoint.\n"
            "Likely macOS Privacy/TCC blocking Python from accessing a protected folder (e.g. ~/Downloads).\n\n"
            "Fastest fix (no copying needed): MOVE the file out of Downloads, then retry:\n"
            "  mv ~/Downloads/step5000.pt /Users/theapemachine/go/src/github.com/theapemachine/experiments/\n"
            "  python3.12 chat_v29.py --ckpt ./step5000.pt --device mps\n\n"
            "Permanent fix:\n"
            "  System Settings → Privacy & Security → Files and Folders (or Full Disk Access)\n"
            "  and allow access for your Terminal and/or python3.12.\n\n"
            f"Original error: {e}"
        ) from e
    cfg_dict = ckpt.get("config", None)
    if cfg_dict is None:
        raise SystemExit("Checkpoint missing 'config' (can't reconstruct model safely).")

    cfg = ModelConfig(**cfg_dict)
    if args.print_config:
        print(asdict(cfg))

    sd = ckpt.get("model", None)
    if sd is None:
        raise SystemExit("Checkpoint missing 'model' state_dict.")

    model = GPT(cfg)
    # Match model dtype to checkpoint tensors when possible (helps avoid dtype mismatch surprises).
    sd_dtype = _dtype_from_state_dict(sd)
    if sd_dtype is not None:
        try:
            model = model.to(dtype=sd_dtype)
        except Exception:
            pass
    model = model.to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Conversation state (token IDs). We'll keep exact history tokens to avoid re-encoding everything.
    history_ids: List[int] = []
    if args.system.strip():
        history_ids = _encode(enc, args.system.strip() + "\n")

    # Back-compat: --style overrides --chat flags.
    if args.style == "raw":
        args.chat = False
    elif args.style in ("chat", "chat_fewshot"):
        args.chat = True

    # In-context example (helps non-instruction-tuned models follow the format).
    if args.style == "chat_fewshot":
        # Keep it short so it doesn't steal too much context.
        demo = (
            f"{args.user_prefix} Say hello in one sentence.\n"
            f"{args.assistant_prefix} Hello! How can I help?\n"
            f"{args.user_prefix} Answer concisely: what is RoPE?\n"
            f"{args.assistant_prefix} RoPE (rotary positional embeddings) encodes position by rotating Q/K features as a function of token index.\n"
        )
        history_ids = history_ids + _encode(enc, demo)

    # Default stop: when model begins the next user turn.
    stop_texts: List[str] = []
    if args.stop_seq:
        stop_texts = [str(s) for s in args.stop_seq if str(s)]
    else:
        stop_texts = ["\n" + str(args.user_prefix)]
    stop_token_seqs: List[Tuple[str, List[int]]] = []
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
            history_ids = _encode(enc, args.system.strip() + "\n") if args.system.strip() else []
            print("[chat_v29] reset", file=sys.stderr)
            continue

        if args.chat:
            # Chat template: we always prompt as:
            #   User: ...
            #   Assistant:
            user_turn = f"{args.user_prefix} {user}\n{args.assistant_prefix} "
        else:
            user_turn = user
            if not user_turn.endswith("\n"):
                user_turn += "\n"

        prompt_ids = _encode(enc, user_turn)
        if args.no_history:
            cur_ids = (history_ids + prompt_ids) if history_ids else prompt_ids
        else:
            history_ids = history_ids + prompt_ids
            cur_ids = history_ids

        # Ensure prompt fits in context window.
        if len(cur_ids) > int(cfg.block_size):
            cur_ids = cur_ids[-int(cfg.block_size) :]
            if not args.no_history:
                history_ids = cur_ids

        x = torch.tensor([cur_ids], dtype=torch.long, device=device)
        out_ids = cur_ids[:]
        if bool(args.no_stream):
            with torch.inference_mode():
                y = model.generate(
                    x,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=(int(args.top_k) if args.top_k is not None else None),
                    kv_cache=str(args.kv_cache),
                    kv_qblock=int(args.kv_qblock),
                    kv_residual=int(args.kv_residual),
                    kv_decode_block=int(args.kv_decode_block),
                    kv_fused=str(args.kv_fused),
                )
            out_ids = y[0].tolist()
            new_ids = out_ids[len(cur_ids) :]
            text = enc.decode(new_ids) if new_ids else ""
            print(text, end="" if text.endswith("\n") else "\n")
        else:
            # Stream token pieces as they are produced.
            sys.stdout.flush()
            pieces: List[int] = []
            pending: List[int] = []
            with torch.inference_mode():
                for tid in _stream_generate(
                    model=model,
                    cfg=cfg,
                    prompt=x,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=(int(args.top_k) if args.top_k is not None else None),
                    kv_cache=str(args.kv_cache),
                    kv_qblock=int(args.kv_qblock),
                    kv_residual=int(args.kv_residual),
                    kv_decode_block=int(args.kv_decode_block),
                    kv_fused=str(args.kv_fused),
                    LayerKVCache=LayerKVCache,
                    DecoupledLayerKVCache=DecoupledLayerKVCache,
                    KVCacheTensorConfig=KVCacheTensorConfig,
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
                            sys.stdout.write(enc.decode(keep))
                            sys.stdout.flush()
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
                            sys.stdout.write(enc.decode(flush_tokens))
                            sys.stdout.flush()
                            pieces.extend(flush_tokens)

            # Flush any remaining tokens that we intentionally kept as tail.
            if pending:
                sys.stdout.write(enc.decode(pending))
                sys.stdout.flush()
                pieces.extend(pending)
                pending = []
            sys.stdout.write("\n")
            sys.stdout.flush()
            out_ids = out_ids + pieces

        if not args.no_history:
            # If using chat template, ensure assistant output is separated from the next prompt.
            if args.chat:
                out_ids = out_ids + _encode(enc, "\n")
            history_ids = out_ids
            # Keep history bounded to context window.
            if len(history_ids) > int(cfg.block_size):
                history_ids = history_ids[-int(cfg.block_size) :]


if __name__ == "__main__":
    main()


