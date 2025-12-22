"""Keying helpers for cache-policy caching."""

from __future__ import annotations

import logging
from typing import cast

import torch

from production.selfopt_utils import device_sig

from production.optimizer.tuner.buckets import pow2_bucket

_LOG = logging.getLogger(__name__)

def _get_int_attr(o: object, name: str, default: int) -> int:
    """Fetch an attribute and coerce it to int for stable key construction.

    Behavior:
    - bool -> int(bool)
    - int -> returned as-is
    - float -> truncated via int(float)
    - str -> attempts int(str); on failure logs a warning and returns default
    - None -> returns default
    - other types -> logs a warning and returns default
    """
    v = cast(object | None, getattr(o, name, None))
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            _LOG.warning(
                "Expected %s.%s to be int-like; got non-numeric string %r. Using default=%d.",
                type(o).__name__,
                name,
                v,
                default,
            )
            return default
    if v is not None:
        _LOG.warning(
            "Expected %s.%s to be int-like; got %s (%r). Using default=%d.",
            type(o).__name__,
            name,
            type(v).__name__,
            v,
            default,
        )
    return default


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


