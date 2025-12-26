"""Shared utilities for topology implementations.

Common helpers that multiple topologies need, like extracting the
tensor output from layers that return (output, cache) tuples.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch import Tensor
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from caramba.runtime.activation import exceeds_activation_threshold


def unwrap_output(out: Tensor | tuple[Tensor, object]) -> Tensor:
    """Extract the tensor from a layer output.

    Many layers (especially AttentionLayer) return (output, cache) tuples.
    Topologies that don't track caches use this to get just the tensor.
    """
    if isinstance(out, tuple):
        return out[0]
    return out


def should_activation_checkpoint(*, x: Tensor, threshold_mb: float) -> bool:
    """Decide whether to activation-checkpoint based on a byte threshold."""

    if float(threshold_mb) <= 0.0:
        return True
    return exceeds_activation_threshold(tensors=[x], threshold_mb=float(threshold_mb))


def activation_checkpoint(fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Run a function under activation checkpointing with a stable default.

    Why this exists:
    - Prefer `use_reentrant=False` (PyTorch recommendation)
    - Fall back for older torch versions that don't accept the kwarg
    """

    try:
        return cast(Tensor, _torch_checkpoint(fn, x, use_reentrant=False))  # type: ignore[call-arg]
    except TypeError:
        return cast(Tensor, _torch_checkpoint(fn, x))
