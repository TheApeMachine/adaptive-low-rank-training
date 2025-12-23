"""
linear provides the linear layer.
"""
from __future__ import annotations

from typing import TypeAlias, Annotated
from torch import Tensor


T: TypeAlias = Annotated[Tensor, "The input tensor type."]

class Linear:
    """
    Linear provides the linear layer.
    """
    def __call__(self, x: T) -> T: ...

    def forward(self, x: T) -> T: ...
