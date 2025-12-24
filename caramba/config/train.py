"""
train provides training configuration models.
"""
from __future__ import annotations

import enum
from pydantic import BaseModel, Field


class TrainPhase(str, enum.Enum):
    """
    TrainPhase provides the training phase.
    """
    BLOCKWISE = "blockwise"
    GLOBAL = "global"


class TrainConfig(BaseModel):
    """
    TrainConfig provides training parameters for a run.
    """
    phase: TrainPhase
    batch_size: int = Field(ge=1)
    block_size: int = Field(ge=1)
    lr: float = Field(gt=0.0)
    device: str = "cpu"
    dtype: str = "float32"
    teacher_ckpt: str | None = None
