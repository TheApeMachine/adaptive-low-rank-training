"""
attention provides scaled dot-product attention primitives.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override


def _sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
) -> Tensor:
    """
    Call torch's scaled-dot-product-attention with strict typing.
    """
    fn_obj = getattr(F, "scaled_dot_product_attention", None)
    if not callable(fn_obj):
        raise RuntimeError("scaled_dot_product_attention is not available")

    def _call(fn: Callable[..., object], /, *args: object, **kwargs: object) -> object:
        return fn(*args, **kwargs)

    out = _call(
        fn_obj,
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=float(dropout_p),
        is_causal=bool(is_causal),
    )
    if not isinstance(out, torch.Tensor):
        raise RuntimeError("scaled_dot_product_attention returned non-Tensor")
    return out


class AttentionOp(nn.Module):
    """
    AttentionOp computes attention outputs from projected Q/K/V.

    Supports GQA by repeating KV heads as needed.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        attn_mask: Tensor | None,
        dropout_p: float,
        is_causal: bool,
    ) -> Tensor:
        """
        q: (B, H, Tq, Dq)
        k: (B, H_kv, Tk, Dq)
        v: (B, H_kv, Tk, Dv)
        returns: (B, H, Tq, Dv)
        """
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                f"Expected rank-4 q/k/v, got q={q.shape}, k={k.shape}, v={v.shape}"
            )
        bq, hq, tq, dq = q.shape
        bk, hk, tk, dk = k.shape
        bv, hv, tv, dv = v.shape

        if int(bq) != int(bk) or int(bq) != int(bv):
            raise ValueError(
                f"Batch mismatch: q={q.shape}, k={k.shape}, v={v.shape}"
            )
        if int(tk) != int(tv) or int(hk) != int(hv):
            raise ValueError(
                f"K/V mismatch: k={k.shape}, v={v.shape}"
            )
        if int(dq) != int(dk):
            raise ValueError(
                f"Q/K head dim mismatch: q={q.shape}, k={k.shape}"
            )

        if int(hk) != int(hq):
            if int(hk) <= 0:
                raise ValueError(f"KV heads must be > 0, got {int(hk)}")
            if int(hq) % int(hk) != 0:
                raise ValueError(
                    f"GQA requires H % H_kv == 0, got H={int(hq)}, H_kv={int(hk)}"
                )
            group = int(hq) // int(hk)
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        if k.shape[1] != q.shape[1] or v.shape[1] != q.shape[1]:
            raise RuntimeError(
                f"GQA repeat failed: q={q.shape}, k={k.shape}, v={v.shape}"
            )
        if int(k.shape[2]) != int(tk) or int(v.shape[2]) != int(tv):
            raise RuntimeError("Unexpected token dim change during GQA repeat")

        if int(tq) <= 0 or int(tk) <= 0:
            raise ValueError(f"Empty seq_len: q={q.shape}, k={k.shape}")

        return _sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=bool(is_causal),
        )


