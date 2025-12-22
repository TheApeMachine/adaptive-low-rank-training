"""
base provides the core interface for all neural models.
"""
from __future__ import annotations
import torch.nn as nn
from typing_extensions import override


class BaseModel(nn.Module):
    """Deprecated shim: kept for backwards compatibility with older checkpoints/scripts."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        _ = args
        _ = kwargs
        raise NotImplementedError("BaseModel.forward must be overridden by subclasses")
