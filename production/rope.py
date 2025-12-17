from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    RoPE with cached cos/sin tables.

    NOTE:
      The original v29 implementation cached (cos, sin) keyed by (device, dtype, seq_len).
      During token-by-token decode that makes seq_len grow by 1 each step, which creates O(N)
      separate cache entries and can blow up memory.

      This version caches only per (device, dtype) and grows the table amortized (power-of-two)
      so decode stays O(N) memory and ~O(N) total trig work.
    """

    def __init__(self, rot_dim: int, base: float = 10000.0):
        super().__init__()
        if rot_dim % 2 != 0:
            raise ValueError(f"rot_dim must be even, got {rot_dim}")
        self.rot_dim = rot_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache: (device, dtype) -> (cos, sin) with shape (L_cached, rot_dim/2)
        self._cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _next_pow2(n: int) -> int:
        n = int(n)
        if n <= 0:
            return 0
        return 1 << (n - 1).bit_length()

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(seq_len)
        key = (str(device), str(dtype))
        cached = self._cache.get(key, None)
        if cached is None:
            cached_len = 0
            cos_cached = None
            sin_cached = None
        else:
            cos_cached, sin_cached = cached
            cached_len = int(cos_cached.size(0))

        if cached is None or cached_len < seq_len:
            target_len = self._next_pow2(seq_len)
            if target_len < seq_len:
                target_len = seq_len

            start = cached_len
            t = torch.arange(start, target_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            cos_new = torch.cos(freqs).to(dtype=dtype)
            sin_new = torch.sin(freqs).to(dtype=dtype)

            if cached is None:
                cos_all = cos_new
                sin_all = sin_new
            else:
                cos_all = torch.cat([cos_cached, cos_new], dim=0)  # type: ignore[arg-type]
                sin_all = torch.cat([sin_cached, sin_new], dim=0)  # type: ignore[arg-type]

            self._cache[key] = (cos_all, sin_all)
            cos_cached, sin_cached = cos_all, sin_all

        return cos_cached[:seq_len], sin_cached[:seq_len]

    def rotate(self, x: torch.Tensor, pos_offset: int) -> torch.Tensor:
        """
        x: (B,H,T,D)
        applies to first rot_dim of D
        """
        _B, _H, T, D = x.shape
        rot = self.rot_dim
        if rot > D:
            raise ValueError(f"rot_dim {rot} > head_dim {D}")
        cos, sin = self._cos_sin(pos_offset + T, x.device, x.dtype)
        cos = cos[pos_offset : pos_offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,rot/2)
        sin = sin[pos_offset : pos_offset + T].unsqueeze(0).unsqueeze(0)

        x_rot = x[..., :rot]
        x_pass = x[..., rot:]

        x1 = x_rot[..., : rot // 2]
        x2 = x_rot[..., rot // 2 : rot]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2, x_pass], dim=-1)


