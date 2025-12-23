"""
transformer provides the transformer model.
"""
from __future__ import annotations

from dataclasses import dataclass
from torch import nn, Tensor
from caramba.config.topology import TopologyConfig
from caramba.builder.topology import build
from caramba.compiler.lower import lower_topology
from caramba.compiler.validate import validate_topology


@dataclass(frozen=True, slots=True)
class _TransformerBlockSpec:
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    causal: bool


class Transformer(nn.Module):
    """
    Transformer provides the transformer model.
    """
    def __init__(self, config: TopologyConfig) -> None:
        super().__init__()
        lowered = lower_topology(config)
        validate_topology(lowered)
        self.config: TopologyConfig = lowered
        self.topology: nn.Module = build(lowered)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the transformer model.
        """
        return self.topology.forward(x)