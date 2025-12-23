"""
dropout provides dropout math.
"""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import override


class Drop(nn.Module):
    """
    Drop applies dropout with probability p.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p: float = float(p)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for dropout.
        """
        return F.dropout(x, p=self.p, training=self.training)


