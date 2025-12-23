"""
dense provides dense weight containers.
"""

from __future__ import annotations

import math

from torch import Tensor, nn
import torch.nn.init as init


class DenseWeight(nn.Module):
    """
    DenseWeight stores a dense matrix and optional bias.
    """
    def __init__(self, d_in: int, d_out: int, *, bias: bool) -> None:
        super().__init__()
        self.d_in: int = int(d_in)
        self.d_out: int = int(d_out)

        self.weight: nn.Parameter = nn.Parameter(
            Tensor(self.d_out, self.d_in),
        )
        self.bias: nn.Parameter | None = (
            nn.Parameter(Tensor(self.d_out)) if bool(bias) else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        reset weight parameters.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)


