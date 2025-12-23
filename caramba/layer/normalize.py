"""
normalize provides the normalize layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import LayerNormLayerConfig


class Normalize(nn.Module):
    """
    Normalize provides the normalize layer.
    """
    def __init__(self, config: LayerNormLayerConfig) -> None:
        super().__init__()
        self.config: LayerNormLayerConfig = config
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(
            config.d_model,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the normalize layer.
        """
        return self.layer_norm(x)