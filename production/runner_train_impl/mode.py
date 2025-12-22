"""Attention-mode parsing for training runner.

Why this exists:
- CLI/config flows historically used string values like "standard"/"decoupled".
- Keep a single normalization point so the runner doesn't diverge from model config.
"""

from __future__ import annotations



def mode_from_str(s: str | None) -> str:
    """Normalize legacy mode strings to canonical values ("standard","gqa","bottleneck","decoupled")."""
    v = str(s or "").strip().lower()
    if v in ("standard", "baseline", "base"):
        return "standard"
    if v in ("gqa", "bottleneck", "decoupled"):
        return v
    return "bottleneck"

