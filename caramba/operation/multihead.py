"""
multihead provides multihead attention compute over weight modules.
"""

from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override


class MultiheadOp(nn.Module):
    """
    MultiheadOp runs multihead attention using an injected weight module.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, x: Tensor, *, attn: nn.MultiheadAttention) -> Tensor:
        """
        forward pass for multihead attention.
        """
        return attn(x, x, x, need_weights=False)[0]


