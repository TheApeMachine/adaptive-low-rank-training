"""Activation sizing helpers for memory-aware training.

Long-sequence training is often limited by activation memory, not parameters.
These helpers provide a cheap way to estimate activation tensor sizes so we
can make manifest-driven decisions (e.g., whether to checkpoint a subgraph).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def tensor_nbytes(t: Tensor) -> int:
    """Return the number of bytes consumed by a tensor's storage."""

    return int(t.numel() * t.element_size())


def tensors_nbytes(tensors: Iterable[Tensor]) -> int:
    """Return total bytes for a collection of tensors."""

    return int(sum(tensor_nbytes(t) for t in tensors))


def exceeds_activation_threshold(*, tensors: Iterable[Tensor], threshold_mb: float) -> bool:
    """Check if total activation bytes exceed a threshold in MB."""

    threshold = float(threshold_mb) * 1024.0 * 1024.0
    return float(tensors_nbytes(tensors)) >= threshold

