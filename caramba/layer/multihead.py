"""
multihead provides the multihead layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import MultiheadLayerConfig
from caramba.operation.multihead import MultiheadOp
from caramba.weight.multihead import MultiheadWeight


class Multihead(nn.Module):
    """
    Multihead provides the multihead layer.
    """
    def __init__(self, config: MultiheadLayerConfig) -> None:
        super().__init__()
        self.config: MultiheadLayerConfig = config
        self.operation: MultiheadOp = MultiheadOp()
        self.weight: MultiheadWeight = MultiheadWeight(
            d_model=int(config.weight.d_model),
            n_heads=int(config.weight.n_heads),
            dropout=float(config.weight.dropout),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the multihead layer.
        """
        return self.operation.forward(x, attn=self.weight.attn)
