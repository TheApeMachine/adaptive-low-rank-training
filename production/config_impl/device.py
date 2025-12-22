"""Device selection helpers.

Why this exists:
- We want a single, predictable place that decides the default device.
- Runners and CLIs can depend on this without importing torch at module import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

def pick_device(explicit: str | None = None) -> "torch.device":
    """
    pick_device chooses the best available accelerator unless the user overrides it.
    """
    # Local import: allows importing config/CLI without torch installed (parse-only, docs, etc).
    import torch

    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


