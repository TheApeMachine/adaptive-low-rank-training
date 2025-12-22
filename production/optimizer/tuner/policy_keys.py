"""Keying helpers for cache-policy caching."""

from __future__ import annotations

import torch

from production.selfopt_utils import device_sig

from production.optimizer.tuner.buckets import pow2_bucket


def _get_int_attr(o: object, name: str, default: int) -> int:
    v = getattr(o, name, None)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    return int(default)


def attn_mode_value(model_cfg: object) -> str:
    """Best-effort string value for Mode enums and legacy string modes."""
    mode_obj = getattr(model_cfg, "attn_mode", None)
    return str(getattr(mode_obj, "value", mode_obj) or "")


def policy_key(*, device: torch.device, model_cfg: object, batch_size: int, max_seq_len: int) -> str:
    """Stable key for caching a chosen cache-policy."""
    max_bucket = pow2_bucket(int(max_seq_len))
    sem_dim = _get_int_attr(model_cfg, "sem_dim", 0)
    geo_dim = _get_int_attr(model_cfg, "geo_dim", 0)
    attn_dim = _get_int_attr(model_cfg, "attn_dim", 0)
    n_head = _get_int_attr(model_cfg, "n_head", 1)
    dims = f"sem={sem_dim},geo={geo_dim},v={attn_dim},H={n_head}"
    return f"{device_sig(device)}|decoupled|max={max_bucket}|B={int(batch_size)}|{dims}"


