"""
rms_norm provides the RMSNorm layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import RMSNormLayerConfig
from caramba.operation.rms_norm import RMSNormOp
from caramba.weight.rms_norm import RMSNormWeight


class RMSNorm(nn.Module):
    """
    RMSNorm provides the RMSNorm layer.
    """
    def __init__(self, config: RMSNormLayerConfig) -> None:
        super().__init__()
        self.config: RMSNormLayerConfig = config
        self.operation: RMSNormOp = RMSNormOp()
        self.weight: RMSNormWeight = RMSNormWeight(int(config.weight.d_model))
        self.eps: float = float(config.operation.eps)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for RMSNorm.
        """
        return self.operation.forward(
            x,
            weight=self.weight.weight,
            eps=float(self.eps),
        )


