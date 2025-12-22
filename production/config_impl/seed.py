"""RNG seeding helpers.

Why this exists:
- Runners want deterministic behavior across torch+cuda where possible.
- We keep imports local to avoid importing torch on module import.
"""

from __future__ import annotations

import random

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _call_best_effort(fn: object, *args: object, **kwargs: object) -> None:
    if not callable(fn):
        return
    try:
        _ = fn(*args, **kwargs)
    except (OSError, TypeError, ValueError, AttributeError, RuntimeError):
        return


def set_seed(seed: int) -> None:
    """Why: make training/sampling reproducible across Python and torch RNGs."""
    random.seed(int(seed))
    if torch is None:
        return
    torch_mod: object = torch
    _call_best_effort(getattr(torch_mod, "manual_seed", None), int(seed))
    cuda = getattr(torch_mod, "cuda", None)
    is_avail = getattr(cuda, "is_available", None)
    if callable(is_avail) and bool(is_avail()):
        _call_best_effort(getattr(cuda, "manual_seed_all", None), int(seed))


