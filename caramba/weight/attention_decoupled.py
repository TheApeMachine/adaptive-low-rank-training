"""
attention_decoupled provides decoupled bottleneck attention (DBA) weights.
"""

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import override

from caramba.operation.rope import RotaryEmbedding
from caramba.weight.dense import DenseWeight


class DecoupledAttentionWeight(nn.Module):
    """
    DecoupledAttentionWeight stores semantic/geometric QK projections plus V/O.

    RoPE is intended to be applied only to the geometric path.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        sem_dim: int,
        geo_dim: int,
        rope_base: float,
        rope_dim: int,
        bias: bool,
        gate: bool,
    ) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        self.n_heads: int = int(n_heads)
        self.n_kv_heads: int = int(n_kv_heads)
        self.sem_dim: int = int(sem_dim)
        self.geo_dim: int = int(geo_dim)

        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0, got {self.n_kv_heads}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads % n_kv_heads must be 0, got {self.n_heads} % {self.n_kv_heads}"
            )
        if self.sem_dim <= 0 or self.sem_dim % self.n_heads != 0:
            raise ValueError(
                f"sem_dim must be > 0 and divisible by n_heads, got sem_dim={self.sem_dim}, n_heads={self.n_heads}"
            )
        if self.geo_dim <= 0 or self.geo_dim % self.n_heads != 0:
            raise ValueError(
                f"geo_dim must be > 0 and divisible by n_heads, got geo_dim={self.geo_dim}, n_heads={self.n_heads}"
            )

        self.head_dim: int = self.d_model // self.n_heads
        self.sem_head_dim: int = self.sem_dim // self.n_heads
        self.geo_head_dim: int = self.geo_dim // self.n_heads

        rope_dim = int(rope_dim)
        if rope_dim <= 0:
            raise ValueError(f"rope_dim must be > 0, got {rope_dim}")
        if rope_dim > self.geo_head_dim:
            raise ValueError(
                f"rope_dim ({rope_dim}) must be <= geo_head_dim ({self.geo_head_dim})"
            )
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")

        k_sem_dim = self.n_kv_heads * self.sem_head_dim
        k_geo_dim = self.n_kv_heads * self.geo_head_dim

        self.q_sem: DenseWeight = DenseWeight(
            self.d_model,
            self.n_heads * self.sem_head_dim,
            bias=bool(bias),
        )
        self.k_sem: DenseWeight = DenseWeight(
            self.d_model,
            k_sem_dim,
            bias=bool(bias),
        )
        self.q_geo: DenseWeight = DenseWeight(
            self.d_model,
            self.n_heads * self.geo_head_dim,
            bias=bool(bias),
        )
        self.k_geo: DenseWeight = DenseWeight(
            self.d_model,
            k_geo_dim,
            bias=bool(bias),
        )

        self.v_proj: DenseWeight = DenseWeight(
            self.d_model,
            self.n_kv_heads * self.head_dim,
            bias=bool(bias),
        )
        self.o_proj: DenseWeight = DenseWeight(
            self.n_heads * self.head_dim,
            self.d_model,
            bias=bool(bias),
        )

        self.rope: RotaryEmbedding = RotaryEmbedding(
            rope_dim,
            base=float(rope_base),
        )

        self.gate_enabled: bool = bool(gate)
        self.gate_logit: nn.Parameter | None = (
            nn.Parameter(torch.zeros(self.n_heads))
            if self.gate_enabled
            else None
        )

    @override
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = (args, kwargs)
        raise RuntimeError(
            "DecoupledAttentionWeight is a weight container; call Attention.forward."
        )


