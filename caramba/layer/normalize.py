"""
normalize provides the normalize layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import LayerNormLayerConfig
from caramba.operation.layer_norm import LayerNormOp
from caramba.weight.layer_norm import LayerNormWeight


class Normalize(nn.Module):
    """
    Normalize provides the normalize layer.
    """
    def __init__(self, config: LayerNormLayerConfig) -> None:
        super().__init__()
        self.config: LayerNormLayerConfig = config
        self.operation: LayerNormOp = LayerNormOp()
        self.weight: LayerNormWeight = LayerNormWeight(
            config.weight.d_model,
            elementwise_affine=bool(config.weight.elementwise_affine),
        )
        self.eps: float = float(config.operation.eps)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the normalize layer.
        """
        return self.operation.forward(
            x,
            normalized_shape=(int(self.config.weight.d_model),),
            weight=self.weight.weight,
            bias=self.weight.bias,
            eps=self.eps,
        )