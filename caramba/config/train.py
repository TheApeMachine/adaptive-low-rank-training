"""Training configuration for model upcycling.

Training happens in two phases: blockwise distillation (train each layer
individually to match the teacher) and global fine-tuning (train the whole
model on language modeling). This module defines the hyperparameters and
optimization settings for both phases.
"""
from __future__ import annotations

import enum

from pydantic import BaseModel

from caramba.config import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Probability,
)


class TrainPhase(str, enum.Enum):
    """Which phase of upcycling training to run.

    BLOCKWISE trains each attention layer individually to match the teacher.
    GLOBAL fine-tunes the whole model on next-token prediction.
    """

    BLOCKWISE = "blockwise"
    GLOBAL = "global"


class TrainConfig(BaseModel):
    """Hyperparameters and settings for a training run.

    The basics (batch_size, lr, device) are straightforward. The convergence_*
    fields enable adaptive training where each block trains until its loss
    drops below a target. The performance fields enable optimizations like
    teacher caching and mixed precision.
    """

    # Core training parameters
    phase: TrainPhase
    batch_size: PositiveInt
    block_size: PositiveInt
    lr: PositiveFloat
    device: str = "cpu"
    dtype: str = "float32"

    # Auto batch sizing: optionally scale batch size inversely with block_size.
    auto_batch_size: bool = False
    auto_batch_ref_block_size: PositiveInt = 512
    auto_batch_min: PositiveInt = 1

    # Teacher model settings
    teacher_ckpt: str | None = None
    teacher_rope_base: PositiveFloat | None = None
    teacher_rope_dim: PositiveInt | None = None

    # Convergence-based blockwise training: instead of fixed steps per block,
    # train each block until loss drops below target or patience runs out.
    convergence_target: PositiveFloat | None = None
    convergence_patience: PositiveInt = 50
    convergence_max_steps: PositiveInt = 5000

    # Performance optimizations
    cache_teacher_outputs: bool = True
    use_amp: bool = False
    amp_dtype: str = "float16"
    gradient_accumulation_steps: PositiveInt = 1
    num_workers: NonNegativeInt = 0
    pin_memory: bool = False
    compile_model: bool | str = False
    compile_mode: str = "reduce-overhead"

    # Activation checkpointing: controls when we recompute activations instead
    # of storing them to reduce peak memory usage on long sequences.
    activation_checkpointing: bool = False
    activation_checkpoint_threshold_mb: NonNegativeFloat = 0.0

    # LR scheduler (optional).
    scheduler: str = "none"  # none|linear|cosine|constant
    warmup_steps: NonNegativeInt = 0
    min_lr_ratio: Probability = 0.0
