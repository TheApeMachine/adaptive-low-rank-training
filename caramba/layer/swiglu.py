"""
swiglu provides the SwiGLU MLP layer.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import SwiGLULayerConfig
from caramba.operation.matmul import Matmul
from caramba.weight.swiglu import SwiGLUWeight


class SwiGLU(nn.Module):
    """
    SwiGLU provides a SwiGLU MLP layer (SwiGLU(x) = W_down(silu(W_gate x) * W_up x)).
    """
    def __init__(self, config: SwiGLULayerConfig) -> None:
        super().__init__()
        self.config: SwiGLULayerConfig = config
        self.matmul: Matmul = Matmul()
        self.weight: SwiGLUWeight = SwiGLUWeight(
            d_model=int(config.weight.d_model),
            d_ff=int(config.weight.d_ff),
            bias=bool(config.weight.bias),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for SwiGLU.
        """
        gate = self.matmul.forward(
            x,
            weight=self.weight.w_gate.weight,
            bias=self.weight.w_gate.bias,
        )
        up = self.matmul.forward(
            x,
            weight=self.weight.w_up.weight,
            bias=self.weight.w_up.bias,
        )
        y = F.silu(gate) * up
        return self.matmul.forward(
            y,
            weight=self.weight.w_down.weight,
            bias=self.weight.w_down.bias,
        )


