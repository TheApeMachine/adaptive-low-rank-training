"""
recurrent provides the recurrent topology.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import RecurrentTopologyConfig


class Recurrent(nn.Module):
    """
    Recurrent provides a recurrent topology.
    """
    def __init__(self, config: RecurrentTopologyConfig) -> None:
        super().__init__()
        self.config: RecurrentTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList([])

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the recurrent topology.
        """
        for _ in range(int(self.config.repeat)):
            x = self.layers[0].forward(x)
            for layer in self.layers[1:]:
                residual = x
                x = layer.forward(x)
                x = residual + x

        return x