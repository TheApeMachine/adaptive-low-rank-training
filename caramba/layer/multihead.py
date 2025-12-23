"""
multihead provides the multihead layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import MultiheadLayerConfig


class Multihead(nn.Module):
    """
    Multihead provides the multihead layer.
    """
    def __init__(self, config: MultiheadLayerConfig) -> None:
        super().__init__()
        self.config: MultiheadLayerConfig = config
        self.multihead: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the multihead layer.
        """
        return self.multihead(x, x, x, need_weights=False)[0]
