"""Learning-rate scheduler helpers for training loops.

Why this exists:
- Production had multiple scheduler options; caramba needs the same flexibility.
- Schedulers should be configured via the manifest, not hard-coded in loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim import Optimizer


@dataclass(frozen=True)
class LRSchedulerConfig:
    """Configurable scheduler options for a training phase."""

    kind: str = "none"  # none|linear|cosine|constant
    total_steps: int = 0
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0


def build_lr_scheduler(optimizer: Optimizer, cfg: LRSchedulerConfig) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build a scheduler from a config.

    Uses LambdaLR for simplicity and to avoid missing scheduler variants.
    """

    kind = str(cfg.kind or "none").lower().strip()
    total = max(0, int(cfg.total_steps))
    warmup = max(0, int(cfg.warmup_steps))
    min_ratio = float(cfg.min_lr_ratio)
    min_ratio = max(0.0, min(1.0, min_ratio))

    if kind in ("none", "off", "disabled") or total <= 0:
        return None

    def lr_lambda(step: int) -> float:
        # step is 0-indexed in LambdaLR.
        s = max(0, int(step))
        if warmup > 0 and s < warmup:
            return max(1e-8, float(s + 1) / float(warmup))

        # Progress through decay region.
        decay_steps = max(1, total - warmup)
        t = min(decay_steps, max(0, s - warmup))
        frac = float(t) / float(decay_steps)

        if kind == "linear":
            return min_ratio + (1.0 - min_ratio) * (1.0 - frac)
        if kind == "cosine":
            return min_ratio + (1.0 - min_ratio) * (0.5 * (1.0 + math.cos(math.pi * frac)))
        if kind == "constant":
            return 1.0

        # Unknown kind: no-op.
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

