"""Memory budgeting helpers for cache-policy tuning."""

from __future__ import annotations

from production.optimizer.tuner.cache_estimates import estimate_decoupled_kvcache_bytes
from production.optimizer.tuner.cache_policy import KVCachePolicy
from production.optimizer.tuner.config import KVSelfOptConfig


def _get_int_attr(o: object, name: str, default: int) -> int:
    v = getattr(o, name, None)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    return int(default)


def policy_mem_bytes(
    *,
    model_cfg: object,
    batch_size: int,
    max_seq_len: int,
    policy: KVCachePolicy,
) -> int:
    """Estimate bytes for a decoupled KV cache under `policy`."""
    return estimate_decoupled_kvcache_bytes(
        n_layer=_get_int_attr(model_cfg, "n_layer", 0),
        batch_size=int(batch_size),
        max_seq_len=int(max_seq_len),
        sem_dim=_get_int_attr(model_cfg, "sem_dim", 0),
        geo_dim=_get_int_attr(model_cfg, "geo_dim", 0),
        v_dim=_get_int_attr(model_cfg, "attn_dim", 0),
        policy=policy,
    )


def budget_bytes(
    cfg: KVSelfOptConfig,
    *,
    model_cfg: object,
    batch_size: int,
    max_seq_len: int,
    base_policy: KVCachePolicy,
) -> int:
    """Compute budget bytes, defaulting to a conservative multiple of fp16-residual baseline."""
    if cfg.mem_budget_mb is not None:
        return int(float(cfg.mem_budget_mb) * 1024.0 * 1024.0)

    base0 = KVCachePolicy(
        k_sem_kind=base_policy.k_sem_kind,
        k_geo_kind=base_policy.k_geo_kind,
        v_kind=base_policy.v_kind,
        k_sem_qblock=base_policy.k_sem_qblock,
        k_geo_qblock=base_policy.k_geo_qblock,
        v_qblock=base_policy.v_qblock,
        residual_len=0,
    )
    base_bytes = policy_mem_bytes(
        model_cfg=model_cfg,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        policy=base0,
    )
    return int(base_bytes * (1.0 + float(cfg.mem_overhead_frac)))


