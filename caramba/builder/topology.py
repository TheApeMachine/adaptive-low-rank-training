"""
topology provides topology modules.
"""
from __future__ import annotations

from typing_extensions import TypeGuard
from torch import nn
from caramba.config.layer import LayerConfig, _LayerConfigBase
from caramba.config.topology import TopologyConfig, TopologyType, _TopologyConfigBase
from caramba.topology.nested import Nested
from caramba.topology.sequential import Sequential
from caramba.topology.parallel import Parallel
from caramba.topology.branching import Branching
from caramba.topology.cyclic import Cyclic
from caramba.topology.recurrent import Recurrent
from caramba.topology.residual import Residual
from caramba.topology.stacked import Stacked
from caramba.builder.layer import build as build_layer

def is_topology_config(config: object) -> TypeGuard[TopologyConfig]:
    """
    is_topology_config checks if config is a topology config instance.
    """
    return isinstance(config, _TopologyConfigBase)

def is_layer_config(config: object) -> TypeGuard[LayerConfig]:
    """
    is_layer_config checks if config is a layer config instance.
    """
    return isinstance(config, _LayerConfigBase)


def build(config: TopologyConfig) -> nn.Module:
    """
    build_topology builds a topology module from config.
    """
    out = None

    match config.type:
        case TopologyType.NESTED:
            out = Nested(config)
        case TopologyType.STACKED:
            out = Stacked(config)
        case TopologyType.RESIDUAL:
            out = Residual(config)
        case TopologyType.SEQUENTIAL:
            out = Sequential(config)
        case TopologyType.PARALLEL:
            out = Parallel(config)
        case TopologyType.BRANCHING:
            out = Branching(config)
        case TopologyType.CYCLIC:
            out = Cyclic(config)
        case TopologyType.RECURRENT:
            out = Recurrent(config)

    if out is None:
        raise ValueError(f"Unsupported topology type: {config.type}")

    for layer in out.config.layers:
        if is_topology_config(layer):
            out.layers.append(build(layer))
        elif is_layer_config(layer):
            out.layers.append(build_layer(layer))
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    return out