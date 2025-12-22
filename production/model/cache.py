"""
cache handles the construction of KV caches.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from production.kvcache_backend import DecoupledLayerKVCache, LayerKVCache, KVCacheTensorConfig

if TYPE_CHECKING:
    from .config import ModelConfig


def _normalize_attn_mode(mode: object) -> str:
    v = getattr(mode, "value", mode)
    # Treat `None` as "unset"; preserve falsy-but-meaningful values (0/False) by not using `v or ""`.
    s = "" if v is None else str(v).strip().lower()
    if s == "":
        return "bottleneck"
    if s in ("baseline", "standard", "base"):
        return "standard"
    if s in ("gqa", "bottleneck", "decoupled"):
        return s
    raise ValueError(
        f'Unknown attn_mode={v!r} (normalized={s!r}). Accepted aliases: ("standard"/"baseline"/"base"), ("gqa"/"bottleneck"/"decoupled").'
    )

class Cache:
    """Factory for KV cache construction."""
    @staticmethod
    def build(
        cfg: ModelConfig,
        batch_size: int,
        max_seq: int,
        device: torch.device,
        **tensor_cfgs: KVCacheTensorConfig
    ) -> list[DecoupledLayerKVCache | LayerKVCache]:
        """Build a list of KV caches (one per layer)."""
        return [
            Cache.build_layer(cfg, batch_size, max_seq, device, **tensor_cfgs)
            for _ in range(cfg.n_layer)
        ]

    @staticmethod
    def build_layer(
        cfg: ModelConfig,
        batch_size: int,
        max_seq: int,
        device: torch.device,
        **tensor_cfgs: KVCacheTensorConfig
    ) -> DecoupledLayerKVCache | LayerKVCache:
        """Build a single KV cache layer."""
        mode = _normalize_attn_mode(getattr(cfg, "attn_mode", "bottleneck"))
        if mode == "decoupled":
            return DecoupledLayerKVCache(
                batch_size=batch_size, max_seq_len=max_seq,
                k_sem_dim=cfg.sem_dim, k_geo_dim=cfg.geo_dim, v_dim=cfg.attn_dim,
                k_sem_cfg=tensor_cfgs.get("k_sem", KVCacheTensorConfig(kind="fp16", qblock=32)),
                k_geo_cfg=tensor_cfgs.get("k_geo", KVCacheTensorConfig(kind="fp16", qblock=32)),
                v_cfg=tensor_cfgs.get("v", KVCacheTensorConfig(kind="fp16", qblock=32)),
                device=device
            )

        # Standard or GQA
        kdim = (
            cfg.d_model if mode == "standard"
            else int((cfg.kv_head or cfg.n_head) * (cfg.attn_dim // cfg.n_head))
        )
        kv_cfg = tensor_cfgs.get("v", KVCacheTensorConfig(kind="fp16", qblock=32))
        return LayerKVCache(
            batch_size=batch_size, max_seq_len=max_seq,
            k_dim=kdim, v_dim=kdim,
            k_cfg=tensor_cfgs.get("k", kv_cfg), v_cfg=kv_cfg,
            device=device
        )
