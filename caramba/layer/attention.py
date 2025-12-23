"""
attention provides an attention layer (operation + weight strategy).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import AttentionLayerConfig
from caramba.config.weight import (
    DecoupledAttentionWeightConfig,
    LlamaAttentionWeightConfig,
)
from caramba.operation.attention import AttentionOp
from caramba.operation.attention_math import decoupled_qk_cat
from caramba.operation.matmul import Matmul
from caramba.weight.attention_decoupled import DecoupledAttentionWeight
from caramba.weight.attention_llama import LlamaAttentionWeight


class Attention(nn.Module):
    """
    Attention provides a causal self-attention layer with pluggable weights.

    Distillation hooks can capture per-layer attention outputs by registering
    a forward hook on this module (post output-projection), or on `operation`
    (raw head outputs).
    """

    def __init__(self, config: AttentionLayerConfig) -> None:
        super().__init__()
        self.config: AttentionLayerConfig = config
        self.matmul: Matmul = Matmul()
        self.operation: AttentionOp = AttentionOp()
        self.is_causal: bool = bool(config.operation.is_causal)
        self.dropout_p: float = float(config.operation.dropout_p)

        w = config.weight
        if isinstance(w, LlamaAttentionWeightConfig):
            self.weight: nn.Module = LlamaAttentionWeight(
                d_model=int(w.d_model),
                n_heads=int(w.n_heads),
                n_kv_heads=int(w.n_kv_heads),
                rope_base=float(w.rope_base),
                rope_dim=int(w.rope_dim),
                bias=bool(w.bias),
            )
        elif isinstance(w, DecoupledAttentionWeightConfig):
            self.weight = DecoupledAttentionWeight(
                d_model=int(w.d_model),
                n_heads=int(w.n_heads),
                n_kv_heads=int(w.n_kv_heads),
                sem_dim=int(w.sem_dim),
                geo_dim=int(w.geo_dim),
                rope_base=float(w.rope_base),
                rope_dim=int(w.rope_dim),
                bias=bool(w.bias),
                gate=bool(w.gate),
            )
        else:
            raise ValueError(f"Unsupported attention weight config: {type(w)!r}")

    @override
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        pos_offset: int = 0,
    ) -> Tensor:
        """
        forward pass for attention.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {x.shape}")

        dropout_p = float(self.dropout_p) if self.training else 0.0

        match self.weight:
            case LlamaAttentionWeight() as w:
                return self._forward_llama(
                    x,
                    w=w,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    pos_offset=int(pos_offset),
                )
            case DecoupledAttentionWeight() as w:
                return self._forward_decoupled(
                    x,
                    w=w,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    pos_offset=int(pos_offset),
                )
            case _:
                raise RuntimeError(
                    f"Unexpected weight module type: {type(self.weight)!r}"
                )

    def _shape(
        self,
        x: Tensor,
        *,
        n_heads: int,
        head_dim: int,
    ) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, int(n_heads), int(head_dim)).transpose(1, 2).contiguous()

    def _merge(self, x: Tensor) -> Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, int(h) * int(d))

    def _forward_llama(
        self,
        x: Tensor,
        *,
        w: LlamaAttentionWeight,
        attn_mask: Tensor | None,
        dropout_p: float,
        pos_offset: int,
    ) -> Tensor:
        q = self.matmul.forward(x, weight=w.q_proj.weight, bias=w.q_proj.bias)
        k = self.matmul.forward(x, weight=w.k_proj.weight, bias=w.k_proj.bias)
        v = self.matmul.forward(x, weight=w.v_proj.weight, bias=w.v_proj.bias)

        qh = self._shape(q, n_heads=w.n_heads, head_dim=w.head_dim)
        kh = self._shape(k, n_heads=w.n_kv_heads, head_dim=w.head_dim)
        vh = self._shape(v, n_heads=w.n_kv_heads, head_dim=w.head_dim)

        qh = w.rope.forward(qh, pos_offset=int(pos_offset))
        kh = w.rope.forward(kh, pos_offset=int(pos_offset))

        out_h = self.operation.forward(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=bool(self.is_causal),
        )
        out = self._merge(out_h)
        return self.matmul.forward(out, weight=w.o_proj.weight, bias=w.o_proj.bias)

    def _forward_decoupled(
        self,
        x: Tensor,
        *,
        w: DecoupledAttentionWeight,
        attn_mask: Tensor | None,
        dropout_p: float,
        pos_offset: int,
    ) -> Tensor:
        q_sem = self.matmul.forward(x, weight=w.q_sem.weight, bias=w.q_sem.bias)
        k_sem = self.matmul.forward(x, weight=w.k_sem.weight, bias=w.k_sem.bias)
        q_geo = self.matmul.forward(x, weight=w.q_geo.weight, bias=w.q_geo.bias)
        k_geo = self.matmul.forward(x, weight=w.k_geo.weight, bias=w.k_geo.bias)
        v = self.matmul.forward(x, weight=w.v_proj.weight, bias=w.v_proj.bias)

        qsh = self._shape(q_sem, n_heads=w.n_heads, head_dim=w.sem_head_dim)
        ksh = self._shape(k_sem, n_heads=w.n_kv_heads, head_dim=w.sem_head_dim)
        qgh = self._shape(q_geo, n_heads=w.n_heads, head_dim=w.geo_head_dim)
        kgh = self._shape(k_geo, n_heads=w.n_kv_heads, head_dim=w.geo_head_dim)
        vh = self._shape(v, n_heads=w.n_kv_heads, head_dim=w.head_dim)

        qgh = w.rope.forward(qgh, pos_offset=int(pos_offset))
        kgh = w.rope.forward(kgh, pos_offset=int(pos_offset))

        if w.gate_logit is not None:
            gate = torch.sigmoid(w.gate_logit).view(1, -1, 1, 1).to(dtype=x.dtype)
            qsh = qsh * (2.0 * gate)
            qgh = qgh * (2.0 - 2.0 * gate)

        sem_scale = 1.0 / math.sqrt(float(w.sem_head_dim))
        geo_scale = 1.0 / math.sqrt(float(w.geo_head_dim))

        q_cat, k_cat = decoupled_qk_cat(
            q_sem=qsh,
            q_geo=qgh,
            k_sem=ksh,
            k_geo=kgh,
            sem_scale=float(sem_scale),
            geo_scale=float(geo_scale),
        )

        out_h = self.operation.forward(
            q_cat,
            k_cat,
            vh,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=bool(self.is_causal),
        )
        out = self._merge(out_h)
        return self.matmul.forward(out, weight=w.o_proj.weight, bias=w.o_proj.bias)


