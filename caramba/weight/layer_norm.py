"""
layer_norm provides layer norm weight containers.
"""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.init as init


class LayerNormWeight(nn.Module):
    """
    LayerNormWeight stores scale and bias for layer norm.
    """

    def __init__(
        self,
        d_model: int,
        *,
        elementwise_affine: bool,
    ) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        self.elementwise_affine: bool = bool(elementwise_affine)

        self.weight: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None

        if self.elementwise_affine:
            self.weight = nn.Parameter(Tensor(self.d_model))
            self.bias = nn.Parameter(Tensor(self.d_model))
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        reset normalization parameters.
        """
        if self.weight is not None:
            init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


