"""
mode determines whether the run is a training run,
a sampling run, or a chat run.
"""
from __future__ import annotations
import enum


class Mode(enum.Enum):
    """
    Mode determines whether the run is a training run,
    a sampling run, or a chat run.
    """
    TRAIN = "train"
    SAMPLE = "sample"
    CHAT = "chat"