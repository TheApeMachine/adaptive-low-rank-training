"""
rms_norm provides RMSNorm weight containers.
"""
from __future__ import annotations

from torch import Tensor, nn
import torch.nn.init as init
from typing_extensions import override


class RMSNormWeight(nn.Module):
    """
    RMSNormWeight stores the RMSNorm scale.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        self.weight: nn.Parameter = nn.Parameter(Tensor(self.d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        reset RMSNorm parameters.
        """
        init.ones_(self.weight)

    @override
    def forward(self, *args: object, **kwargs: object) -> Tensor:
        """
        forward is intentionally unsupported for weight containers.
        """
        _ = (args, kwargs)
        raise RuntimeError(
            "RMSNormWeight is a weight container; call RMSNorm.forward."
        )


