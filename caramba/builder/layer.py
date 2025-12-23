"""
layer provides layer modules.
"""
from __future__ import annotations

from torch import nn
from caramba.config.layer import LayerConfig, LayerType
from caramba.layer.attention import Attention
from caramba.layer.linear import Linear
from caramba.layer.normalize import Normalize
from caramba.layer.rms_norm import RMSNorm
from caramba.layer.swiglu import SwiGLU
from caramba.layer.multihead import Multihead
from caramba.layer.dropout import Dropout


def build(config: LayerConfig) -> nn.Module:
    """
    build builds a layer module from config.
    """
    out = None

    match config.type:
        case LayerType.LINEAR:
            out = Linear(config)
        case LayerType.LAYER_NORM:
            out = Normalize(config)
        case LayerType.RMS_NORM:
            out = RMSNorm(config)
        case LayerType.MULTIHEAD:
            out = Multihead(config)
        case LayerType.DROPOUT:
            out = Dropout(config)
        case LayerType.ATTENTION:
            out = Attention(config)
        case LayerType.SWIGLU:
            out = SwiGLU(config)

    if out is None:
        raise ValueError(f"Unsupported layer type: {config.type}")

    return out
