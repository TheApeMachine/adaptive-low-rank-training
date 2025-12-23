"""
swiglu provides SwiGLU MLP weight containers.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.weight.dense import DenseWeight


class SwiGLUWeight(nn.Module):
    """
    SwiGLUWeight stores the gate/up/down projection weights.
    """
    def __init__(self, *, d_model: int, d_ff: int, bias: bool) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        self.d_ff: int = int(d_ff)
        self.bias: bool = bool(bias)

        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be > 0, got {self.d_ff}")

        self.w_gate: DenseWeight = DenseWeight(self.d_model, self.d_ff, bias=self.bias)
        self.w_up: DenseWeight = DenseWeight(self.d_model, self.d_ff, bias=self.bias)
        self.w_down: DenseWeight = DenseWeight(self.d_ff, self.d_model, bias=self.bias)

    @override
    def forward(self, *args: object, **kwargs: object) -> Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = (args, kwargs)
        raise RuntimeError(
            "SwiGLUWeight is a weight container; call SwiGLU.forward."
        )


