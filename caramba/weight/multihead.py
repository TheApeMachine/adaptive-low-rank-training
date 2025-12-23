"""
multihead provides multihead attention weight containers.
"""

from __future__ import annotations

from torch import nn


class MultiheadWeight(nn.Module):
    """
    MultiheadWeight stores the MultiheadAttention parameters.
    """

    def __init__(self, *, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )


