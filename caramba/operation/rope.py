"""
rope provides rotary positional embedding (RoPE).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import override


class RotaryEmbedding(nn.Module):
    """
    RotaryEmbedding applies RoPE to the first rot_dim of the head dimension.
    """

    def __init__(self, rot_dim: int, *, base: float) -> None:
        super().__init__()
        rot_dim = int(rot_dim)
        if rot_dim <= 0:
            raise ValueError(f"rot_dim must be > 0, got {rot_dim}")
        if rot_dim % 2 != 0:
            raise ValueError(f"rot_dim must be even, got {rot_dim}")

        self.rot_dim: int = rot_dim
        inv_freq = 1.0 / (
            float(base)
            ** (
                torch.arange(0, rot_dim, 2, dtype=torch.float32)
                / float(rot_dim)
            )
        )
        self.inv_freq: Tensor = inv_freq

        self._cache: dict[
            tuple[str, str], tuple[Tensor, Tensor]
        ] = {}

    @staticmethod
    def _next_pow2(n: int) -> int:
        n = int(n)
        if n <= 0:
            return 0
        return 1 << (n - 1).bit_length()

    def _cos_sin(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        seq_len = int(seq_len)
        key = (str(device), str(dtype))
        cached = self._cache.get(key)
        if cached is None:
            cached_len = 0
            cos_cached = torch.empty(
                (0, self.rot_dim // 2),
                device=device,
                dtype=dtype,
            )
            sin_cached = torch.empty(
                (0, self.rot_dim // 2),
                device=device,
                dtype=dtype,
            )
        else:
            cos_cached, sin_cached = cached
            cached_len = int(cos_cached.size(0))

        if cached_len < seq_len:
            target_len = max(seq_len, self._next_pow2(seq_len))
            start = cached_len
            t = torch.arange(start, target_len, device=device, dtype=torch.float32)
            inv = self.inv_freq.to(device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv)
            cos_new = torch.cos(freqs).to(dtype=dtype)
            sin_new = torch.sin(freqs).to(dtype=dtype)
            cos_cached = torch.cat([cos_cached, cos_new], dim=0)
            sin_cached = torch.cat([sin_cached, sin_new], dim=0)
            self._cache[key] = (cos_cached, sin_cached)

        return cos_cached[:seq_len], sin_cached[:seq_len]

    @override
    def forward(self, x: Tensor, *, pos_offset: int) -> Tensor:
        """
        x: (B, H, T, D)
        """
        if x.ndim != 4:
            raise ValueError(f"RoPE expects rank-4 (B,H,T,D), got {x.shape}")
        _b, _h, t, d = x.shape
        rot = int(self.rot_dim)
        if rot > int(d):
            raise ValueError(f"rot_dim {rot} > head_dim {int(d)}")

        cos, sin = self._cos_sin(
            int(pos_offset) + int(t),
            device=x.device,
            dtype=x.dtype,
        )
        cos = cos[int(pos_offset) : int(pos_offset) + int(t)]
        sin = sin[int(pos_offset) : int(pos_offset) + int(t)]
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,rot/2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_rot = x[..., :rot]
        x_pass = x[..., rot:]

        x1 = x_rot[..., : rot // 2]
        x2 = x_rot[..., rot // 2 : rot]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2, x_pass], dim=-1)


