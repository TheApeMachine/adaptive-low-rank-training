"""
network provides the network configuration.
"""
from __future__ import annotations
import enum

from pydantic import BaseModel
from caramba.config.layer import LayerConfig


class NetworkType(str, enum.Enum):
    """
    NetworkType provides the network type.
    """
    STACKED = "stacked"
    RESIDUAL = "residual"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BRANCHING = "branching"
    CYCLIC = "cyclic"
    RECURRENT = "recurrent"
    CONVOLUTIONAL = "convolutional"
    POOLING = "pooling"
    NORMALIZATION = "normalization"


class NetworkConfig(BaseModel):
    """
    NetworkConfig provides the network configuration.
    """
    type: NetworkType
    layers: list[LayerConfig]
