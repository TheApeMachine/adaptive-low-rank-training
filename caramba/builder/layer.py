"""
layer provides layer modules.
"""
from __future__ import annotations

from torch import nn
from caramba.config.layer import LayerConfig, LayerType
from caramba.config.topology import TopologyConfig
from caramba.layer.linear import Linear
from caramba.layer.normalize import Normalize
from caramba.layer.multihead import Multihead
from caramba.layer.dropout import Dropout


def build(config: LayerConfig | TopologyConfig) -> nn.Module:
    """
    build builds a layer module from config.
    """
    out = None

    match config.type:
        case LayerType.LINEAR:
            out = Linear(config)
        case LayerType.LAYER_NORM:
            out = Normalize(config)
        case LayerType.MULTIHEAD:
            out = Multihead(config)
        case LayerType.DROPOUT:
            out = Dropout(config)

    if out is None:
        raise ValueError(f"Unsupported layer type: {config.type}")

    return out
