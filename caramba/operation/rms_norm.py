"""
rms_norm provides RMSNorm math primitives.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import override


class RMSNormOp(nn.Module):
    """
    RMSNormOp applies root-mean-square normalization.
    """
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, x: Tensor, *, weight: Tensor, eps: float) -> Tensor:
        """
        forward pass for RMSNorm.
        """
        if x.ndim < 1:
            raise ValueError(f"Expected x.ndim >= 1, got {x.shape}")
        if weight.ndim != 1:
            raise ValueError(f"Expected 1D weight, got {weight.shape}")
        if int(x.shape[-1]) != int(weight.shape[0]):
            raise ValueError(
                f"Expected x last dim {int(weight.shape[0])}, got {x.shape}"
            )

        x_f = x.float()
        inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        y = (x_f * inv_rms).to(dtype=x.dtype) * weight
        return y


