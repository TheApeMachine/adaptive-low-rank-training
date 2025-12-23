"""
sequential provides the sequential layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import LayerType, SequentialLayerConfig
from caramba.layer.dropout import Dropout
from caramba.layer.linear import Linear
from caramba.layer.multihead import Multihead
from caramba.layer.normalize import Normalize


class Sequential(nn.Module):
    """
    Sequential provides the sequential layer.
    """
    def __init__(self, config: SequentialLayerConfig) -> None:
        super().__init__()
        self.config: SequentialLayerConfig = config
        self.layers: nn.ModuleList = nn.ModuleList([])

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the sequential layer.
        """
        if len(self.layers) == 0:
            for layer in self.config.layers:
                match layer.type:
                    case LayerType.LINEAR:
                        self.layers.append(Linear(layer))
                    case LayerType.LAYER_NORM:
                        self.layers.append(Normalize(layer))
                    case LayerType.MULTIHEAD:
                        self.layers.append(Multihead(layer))
                    case LayerType.DROPOUT:
                        self.layers.append(Dropout(layer))
                    case _:
                        raise ValueError(f"Unsupported layer type: {layer.type}")

        for layer in self.layers:
            x = layer.forward(x)

        return x