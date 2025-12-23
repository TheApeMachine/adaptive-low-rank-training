"""
layer_norm provides layer normalization math.
"""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import override


class LayerNormOp(nn.Module):
    """
    LayerNormOp applies layer normalization.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(
        self,
        x: Tensor,
        *,
        normalized_shape: tuple[int, ...],
        weight: Tensor | None,
        bias: Tensor | None,
        eps: float,
    ) -> Tensor:
        """
        forward pass for layer normalization.
        """
        return F.layer_norm(
            x,
            normalized_shape=normalized_shape,
            weight=weight,
            bias=bias,
            eps=float(eps),
        )


