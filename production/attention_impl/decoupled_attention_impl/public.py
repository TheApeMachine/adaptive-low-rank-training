"""Public surface of the decoupled attention implementation.

Why this exists:
- `production/attention.py` re-exports a small set of symbols for benchmarking and tuning.
- This module defines the stable “API” while allowing the implementation to be split.
"""

from __future__ import annotations

from production.attention_impl.decoupled_attention_impl.helpers import (
    decoupled_qk_cat, decoupled_scores_f32, neg_inf
)
from production.attention_impl.decoupled_attention_impl.triton_runtime import (
    TRITON_AVAILABLE,
    triton_decoupled_q4q8q4_available,
)

from production.attention_impl.decoupled_attention_impl.attention_core import DecoupledBottleneckAttention

__all__ = [
    "DecoupledBottleneckAttention",
    "TRITON_AVAILABLE",
    "decoupled_qk_cat",
    "decoupled_scores_f32",
    "triton_decoupled_q4q8q4_available",
    "neg_inf",
]


