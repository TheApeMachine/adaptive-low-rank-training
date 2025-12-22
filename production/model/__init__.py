"""Transformer model components (public API)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from production.kvcache_backend import (
    DecoupledLayerKVCache,
    KVCacheKind,
    KVCacheTensorConfig,
    LayerKVCache,
)
from production.runtime_tuning import (
    KVCachePolicy,
    KVCachePolicySelfOptimizer,
    KVDecodeSelfOptimizer,
    KVSelfOptConfig,
    estimate_decoupled_kvcache_bytes,
    load_token_ids_spec,
    policy_quality_reject_reasons,
    warn_policy_quality_reject,
)

if TYPE_CHECKING:
    # Import for type checkers only. At runtime we lazy-load to avoid import cycles:
    # `production.model.__init__` used to eagerly import `GPT`, while `gpt.py` pulls in other
    # submodules that (transitively) depend on the package module, forming a cycle.
    from production.model.config import ModelConfig
    from production.model.gpt import GPT

__all__ = [
    "DecoupledLayerKVCache",
    "GPT",
    "KVCacheKind",
    "KVCachePolicy",
    "KVCachePolicySelfOptimizer",
    "KVCacheTensorConfig",
    "KVDecodeSelfOptimizer",
    "KVSelfOptConfig",
    "LayerKVCache",
    "ModelConfig",
    "estimate_decoupled_kvcache_bytes",
    "load_token_ids_spec",
    "policy_quality_reject_reasons",
    "warn_policy_quality_reject",
]


def __getattr__(name: str) -> object:
    # Lazy exports to avoid import cycles at runtime and for pyright's import graph.
    if name == "GPT":
        from production.model.gpt import GPT
        return GPT
    if name == "ModelConfig":
        from production.model.config import ModelConfig
        return ModelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
