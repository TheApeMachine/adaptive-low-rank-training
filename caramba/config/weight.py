"""
weight provides weight configuration models.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field


class WeightType(str, enum.Enum):
    """
    WeightType provides the weight type.
    """

    DENSE = "dense"
    NORM = "norm"
    RMS_NORM = "rms_norm"
    SWIGLU = "swiglu"
    MULTIHEAD = "multihead"
    LLAMA_ATTENTION = "llama_attention"
    DECOUPLED_ATTENTION = "decoupled_attention"


class _WeightConfigBase(BaseModel):
    """
    _WeightConfigBase provides the base type for weight configs.
    """


class DenseWeightConfig(_WeightConfigBase):
    """
    DenseWeightConfig provides the dense weight configuration.
    """

    type: Literal[WeightType.DENSE] = WeightType.DENSE
    d_in: int
    d_out: int
    bias: bool = True


class NormWeightConfig(_WeightConfigBase):
    """
    NormWeightConfig provides the normalization weight configuration.
    """

    type: Literal[WeightType.NORM] = WeightType.NORM
    d_model: int
    elementwise_affine: bool = True


class RMSNormWeightConfig(_WeightConfigBase):
    """
    RMSNormWeightConfig provides RMSNorm weight configuration.
    """
    type: Literal[WeightType.RMS_NORM] = WeightType.RMS_NORM
    d_model: int


class SwiGLUWeightConfig(_WeightConfigBase):
    """
    SwiGLUWeightConfig provides SwiGLU weight configuration.
    """
    type: Literal[WeightType.SWIGLU] = WeightType.SWIGLU
    d_model: int
    d_ff: int
    bias: bool = False


class MultiheadWeightConfig(_WeightConfigBase):
    """
    MultiheadWeightConfig provides the multihead attention weight configuration.
    """

    type: Literal[WeightType.MULTIHEAD] = WeightType.MULTIHEAD
    d_model: int
    n_heads: int
    dropout: float = 0.0


class LlamaAttentionWeightConfig(_WeightConfigBase):
    """
    LlamaAttentionWeightConfig provides Llama-compatible attention weights.
    """

    type: Literal[WeightType.LLAMA_ATTENTION] = WeightType.LLAMA_ATTENTION
    d_model: int
    n_heads: int
    n_kv_heads: int
    rope_base: float = 10000.0
    rope_dim: int
    bias: bool = False


class DecoupledAttentionWeightConfig(_WeightConfigBase):
    """
    DecoupledAttentionWeightConfig provides DBA attention weights.
    """

    type: Literal[WeightType.DECOUPLED_ATTENTION] = WeightType.DECOUPLED_ATTENTION
    d_model: int
    n_heads: int
    n_kv_heads: int
    sem_dim: int
    geo_dim: int
    rope_base: float = 10000.0
    rope_dim: int
    bias: bool = False
    gate: bool = True


WeightConfig: TypeAlias = Annotated[
    DenseWeightConfig
    | NormWeightConfig
    | RMSNormWeightConfig
    | SwiGLUWeightConfig
    | MultiheadWeightConfig
    | LlamaAttentionWeightConfig
    | DecoupledAttentionWeightConfig,
    Field(discriminator="type"),
]


