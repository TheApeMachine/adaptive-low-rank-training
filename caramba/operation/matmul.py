"""
matmul provides linear projection math.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override


class Matmul(nn.Module):
    """
    Matmul applies y = xW^T + b.
    """
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(
        self,
        x: Tensor,
        *,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        """
        forward pass for matmul.
        """
        y: Tensor = x.matmul(weight.transpose(0, 1))

        if bias is not None:
            y = y + bias

        return y
