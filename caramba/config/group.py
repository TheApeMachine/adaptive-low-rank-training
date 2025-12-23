"""
group provides the group module for the training loop.
"""
from __future__ import annotations

from pydantic import BaseModel
from caramba.config.run import Run


class Group(BaseModel):
    """
    Group provides the group module for the training loop.
    """
    name: str
    description: str
    data: str
    runs: list[Run]
