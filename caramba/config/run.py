"""
run provides the run module for the training loop.
"""
from __future__ import annotations

from pydantic import BaseModel
from caramba.config.mode import Mode


class Run(BaseModel):
    """
    Run provides the run module for the training loop.
    """
    id: str
    mode: Mode
    exp: str
    seed: int
    steps: int
    expected: dict[str, object]
