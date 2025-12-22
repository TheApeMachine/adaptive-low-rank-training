"""Keying helpers for decode-plan caching."""

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


def decode_plan_key(*, device: torch.device, attn: object, cache: object, prefix_len: int) -> str:
    """Stable key for caching decode plans."""
    bucket = pow2_bucket(int(prefix_len))
    try:
        k_sem = getattr(cache, "k_sem", None)
        k_geo = getattr(cache, "k_geo", None)
        v = getattr(cache, "v", None)
        k_sem_kind = getattr(k_sem, "kind", None)
        k_geo_kind = getattr(k_geo, "kind", None)
        v_kind = getattr(v, "kind", None)
        if isinstance(k_sem_kind, str) and isinstance(k_geo_kind, str) and isinstance(v_kind, str):
            ksig = f"ksem={k_sem_kind},kgeo={k_geo_kind},v={v_kind}"
        else:
            ksig = "kv=unknown"
    except (AttributeError, TypeError):
        ksig = "kv=unknown"
    try:
        H = _get_int_attr(attn, "H", 1)
        hd_sem = _get_int_attr(attn, "sem_head_dim", 0)
        hd_geo = _get_int_attr(attn, "geo_head_dim", 0)
        hd_v = _get_int_attr(attn, "v_head_dim", 0)
        dims = (
            f"H={int(H)},"
            f"hd_sem={int(hd_sem)},"
            f"hd_geo={int(hd_geo)},"
            f"hd_v={int(hd_v)}"
        )
    except (AttributeError, TypeError):
        dims = "dims=unknown"
    return f"{device_sig(device)}|{bucket}|{dims}|{ksig}"
