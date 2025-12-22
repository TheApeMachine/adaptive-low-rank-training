"""
base provides the core interface for all neural models.
"""
from __future__ import annotations
import warnings
import torch.nn as nn
from typing_extensions import override


class BaseModel(nn.Module):
    """Deprecated shim.

    Prefer using `torch.nn.Module` directly (or use `production.model.gpt.GPT` for the actual
    model implementation in this repo).
    """

    def __new__(cls, *args: object, **kwargs: object) -> BaseModel:
        _ = args
        _ = kwargs
        warnings.warn(
            "BaseModel is deprecated and will be removed in a future release; use torch.nn.Module directly (or production.model.gpt.GPT for the actual model).",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls)

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        _ = args
        _ = kwargs
        raise NotImplementedError("BaseModel.forward must be overridden by subclasses")
