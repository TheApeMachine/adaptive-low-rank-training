#!/usr/bin/env python3
"""
Ablate `null_attn` for decoupled checkpoints by measuring teacher-forced NLL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
import tiktoken  # type: ignore

# Allow running as either:
#   - python -m production.ablate_null_attn  (preferred)
#   - python production/ablate_null_attn.py  (convenient)
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repo_root))

# Local imports must come after sys.path modification for direct script execution
from production.config import pick_device  # pylint: disable=wrong-import-position
from production.model import GPT, ModelConfig  # pylint: disable=wrong-import-position
from production.kvcache_backend import KVCacheTensorConfig  # pylint: disable=wrong-import-position
from production.model.cache import Cache  # pylint: disable=wrong-import-position
from production.runtime_tuning import load_token_ids_spec  # pylint: disable=wrong-import-position
from production.selfopt_cache import as_str_object_dict  # pylint: disable=wrong-import-position


def _load_tokens_auto(spec: str, *, want_len: int) -> list[int]:
    """Load tokens from either integer token IDs or raw text.

    - If `spec` points to a file that parses as whitespace-separated ints (or .npy), use it.
    - Otherwise, treat it as raw text and tokenize with tiktoken (GPT-2 encoding).
    """
    want_len = int(max(2, want_len))
    try:
        ids = load_token_ids_spec(str(spec))
        if len(ids) >= want_len:
            return [int(x) for x in ids]
    except (OSError, ValueError, TypeError):
        pass

    p = Path(str(spec))
    if not p.exists():
        msg = f"--tokens must be a path to token IDs (.txt/.npy) or raw text file; not found: {spec}"
        raise ValueError(msg)

    raw = p.read_text(encoding="utf-8", errors="ignore")
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(raw)

    if len(ids) < want_len:
        raise ValueError(f"Not enough tokenized IDs from text: need {want_len}, got {len(ids)}")

    return [int(x) for x in ids]


def eval_nll_chunked(
    model: GPT,
    tokens_1d: torch.Tensor,
    *,
    seq_len: int,
    chunk_size: int,
    device: torch.device,
) -> float:
    """Teacher-forced next-token NLL over `seq_len` tokens using cache-chunked forward passes."""
    if tokens_1d.dim() != 1:
        raise ValueError("tokens_1d must be 1D")
    total_len = int(seq_len)
    if total_len < 2:
        return float("nan")

    with torch.no_grad():
        required_len = total_len + 1
        actual_len = int(tokens_1d.shape[0])
        if actual_len < required_len:
            raise ValueError(
                f"tokens_1d is too short for seq_len={total_len}: got len(tokens_1d)={actual_len}, "
                + f"required >= {required_len}"
            )
        ids = tokens_1d[: total_len + 1].to(device=device, dtype=torch.long).unsqueeze(0)  # (1, total_len+1)
        # predict next token for each position 0..total_len-1 (target is 1..total_len)
        inp = ids[:, :total_len]
        tgt = ids[:, 1 : total_len + 1]

        # fp16 caches everywhere for evaluation (we're isolating the null_attn effect).
        fp16 = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
        caches = Cache.build(model.cfg, batch_size=1, max_seq=total_len, device=device, k_sem=fp16, k_geo=fp16, v=fp16)
        # Runtime hints (set dynamically to avoid tight coupling to cache impl types).
        for c in caches:
            setattr(c, "decode_block", 1024)
            setattr(c, "fused", "none")

        nll_sum = 0.0
        nll_count = 0
        for i in range(0, total_len, int(chunk_size)):
            end = min(total_len, i + int(chunk_size))
            x = inp[:, i:end]
            y = tgt[:, i:end]
            logits, caches_out = model.forward(x, caches=caches, pos_offset=int(i))
            if caches_out is None:
                raise RuntimeError("Expected caches to be returned when caches are provided.")
            caches = caches_out
            loss = F.cross_entropy(logits.reshape(-1, int(logits.size(-1))), y.reshape(-1), reduction="sum")
            nll_sum += float(loss.item())
            nll_count += int(y.numel())

        return float(nll_sum / max(1, nll_count))


def _torch_load_obj(path: Path, *, map_location: torch.device) -> object:
    try:
        return cast(object, torch.load(str(path), map_location=map_location))
    except Exception as e:
        raise RuntimeError(f"Failed to torch.load checkpoint from {path}") from e


def _as_state_dict(obj: object) -> dict[str, torch.Tensor]:
    d = as_str_object_dict(obj)
    if d is None:
        raise TypeError("state_dict must be a dict")
    out: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
    if not out:
        raise ValueError("state_dict is empty or not a tensor dict")
    return out


def _load_model(*, ckpt_path: str, device: torch.device, null_attn: bool, seq_len: int) -> GPT:
    ckpt_obj = _torch_load_obj(Path(str(ckpt_path)), map_location=device)
    ckpt = as_str_object_dict(ckpt_obj)
    if ckpt is None:
        raise TypeError("Checkpoint must be a mapping")

    cfg_obj = ckpt.get("config", None)
    cfg_map = as_str_object_dict(cfg_obj)
    if cfg_map is None:
        raise ValueError("Checkpoint missing 'config'. Can't reconstruct model safely.")

    cfg = ModelConfig.from_dict(cfg_map, device=device)
    if str(getattr(cfg, "attn_mode", "")) != "decoupled":
        raise ValueError("This ablation script targets decoupled attention checkpoints only.")

    cfg.null_attn = bool(null_attn)
    cfg.block_size = int(max(int(cfg.block_size), int(seq_len)))

    model = GPT(cfg).to(device)
    sd = _as_state_dict(ckpt.get("model", None))
    incompatible = model.load_state_dict(sd, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if isinstance(missing, list) and isinstance(unexpected, list) and (missing or unexpected):
        msg = (
            f"[ablate] non-strict load (null_attn={null_attn}). "
            f"Missing={missing} "
            f"Unexpected={unexpected}"
        )
        print(msg)
    _ = model.eval()
    return model


def _as_int(o: object, default: int) -> int:
    if isinstance(o, bool):
        return int(o)
    if isinstance(o, int):
        return int(o)
    if isinstance(o, float):
        return int(o)
    if isinstance(o, str):
        try:
            return int(o.strip())
        except ValueError:
            return int(default)
    return int(default)


def _as_str(o: object, default: str) -> str:
    if o is None:
        return str(default)
    return str(o)


def main(argv: list[str] | None = None) -> int:
    """
    Ablate null_attn for decoupled checkpoints via teacher-forced NLL on a small token window.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Ablate null_attn for decoupled checkpoints via teacher-forced NLL "
            "on a small token window."
        )
    )
    _ = ap.add_argument("--ckpt", type=str, required=True)
    _ = ap.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Token spec: path to .txt/.npy or whitespace-separated ints."
    )
    _ = ap.add_argument("--seq-len", type=int, default=2048)
    _ = ap.add_argument("--chunk-size", type=int, default=256)
    _ = ap.add_argument("--device", type=str, default=None)
    args_ns = ap.parse_args(argv)
    args_map = cast(dict[str, object], vars(args_ns) or {})

    ckpt = _as_str(args_map.get("ckpt", ""), "")
    tok_spec = _as_str(args_map.get("tokens", ""), "")
    seq_len = _as_int(args_map.get("seq_len", 2048), 2048)
    chunk_size = _as_int(args_map.get("chunk_size", 256), 256)
    dev_raw = args_map.get("device", None)
    dev_str = str(dev_raw) if isinstance(dev_raw, str) else None

    dev = pick_device(None) if dev_str is None else torch.device(str(dev_str))
    ids = _load_tokens_auto(tok_spec, want_len=int(seq_len) + 1)
    tok = torch.tensor(ids, dtype=torch.long)

    m_off = _load_model(
        ckpt_path=ckpt, device=dev, null_attn=False, seq_len=int(seq_len)
    )
    m_on = _load_model(
        ckpt_path=ckpt, device=dev, null_attn=True, seq_len=int(seq_len)
    )

    nll_off = eval_nll_chunked(
        m_off,
        tok,
        seq_len=int(seq_len),
        chunk_size=int(chunk_size),
        device=dev,
    )

    nll_on = eval_nll_chunked(
        m_on, tok,
        seq_len=int(seq_len),
        chunk_size=int(chunk_size),
        device=dev,
    )

    dnll = float(nll_on - nll_off)
    ppl_ratio = float(math.exp(dnll)) if math.isfinite(dnll) else float("nan")

    print(json.dumps({
        "cfg_off": asdict(m_off.cfg),
        "cfg_on": asdict(m_on.cfg)
    }, indent=2, sort_keys=True))
    print(f"[ablate] seq_len={int(seq_len)} chunk={int(chunk_size)} device={dev}")
    print(f"[ablate] null_attn=False NLL={nll_off:.6f}")
    print(f"[ablate] null_attn=True  NLL={nll_on:.6f}")
    print(f"[ablate] ΔNLL(on-off)={dnll:.6f} (ppl_ratio={ppl_ratio:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
