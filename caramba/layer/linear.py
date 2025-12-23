"""
linear provides the linear layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import LinearLayerConfig


class Linear(nn.Module):
    """
    Linear provides the linear layer.
    """

    def __init__(self, config: LinearLayerConfig) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(
            config.d_in,
            config.d_out,
            bias=config.bias,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the linear layer.
        """
        return self.linear(x)


