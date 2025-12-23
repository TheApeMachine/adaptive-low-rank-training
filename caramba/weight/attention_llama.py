"""
attention_llama provides Llama-compatible attention weights (GQA + RoPE).
"""

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import override

from caramba.operation.rope import RotaryEmbedding
from caramba.weight.dense import DenseWeight


class LlamaAttentionWeight(nn.Module):
    """Q/K/V/O projection weights for Llama-style GQA attention + RoPE config."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        rope_base: float,
        rope_dim: int,
        bias: bool,
    ) -> None:
        super().__init__()

        self._validate(d_model, n_heads, n_kv_heads, rope_dim)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope_dim = rope_dim

        def dense(d_in: int, d_out: int) -> DenseWeight:
            return DenseWeight(d_in, d_out, bias=bias)

        q_out = n_heads * self.head_dim
        kv_out = n_kv_heads * self.head_dim

        self.q_proj = dense(d_model, q_out)
        self.k_proj = dense(d_model, kv_out)
        self.v_proj = dense(d_model, kv_out)
        self.o_proj = dense(q_out, d_model)

        self.rope = RotaryEmbedding(rope_dim, base=rope_base)

    @staticmethod
    def _validate(d_model: int, n_heads: int, n_kv_heads: int, rope_dim: int) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0, got {n_kv_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        head_dim = d_model // n_heads
        if rope_dim <= 0:
            raise ValueError(f"rope_dim must be > 0, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim ({rope_dim}) must be <= head_dim ({head_dim})")
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass for the llama attention weight.
        """
        raise RuntimeError("LlamaAttentionWeight is a weight container; call Attention.forward.")
