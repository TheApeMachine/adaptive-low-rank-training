"""
layer provides the layer configuration.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias
from pydantic import BaseModel, Field

from caramba.config.operation import (
    LayerNormOperationConfig,
    RMSNormOperationConfig,
    MatmulOperationConfig,
    MultiheadOperationConfig,
    DropoutOperationConfig,
    AttentionOperationConfig,
    SwiGLUOperationConfig,
)
from caramba.config.weight import (
    DecoupledAttentionWeightConfig,
    DenseWeightConfig,
    LlamaAttentionWeightConfig,
    MultiheadWeightConfig,
    NormWeightConfig,
    RMSNormWeightConfig,
    SwiGLUWeightConfig,
)


class LayerType(str, enum.Enum):
    """
    LayerType provides the layer type.
    """
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    LINEAR = "linear"
    MULTIHEAD = "multihead"
    DROPOUT = "dropout"
    ATTENTION = "attention"
    SWIGLU = "swiglu"


class _LayerConfigBase(BaseModel):
    pass


class LinearLayerConfig(_LayerConfigBase):
    """
    LinearLayerConfig provides the linear layer configuration.
    """
    type: Literal[LayerType.LINEAR] = LayerType.LINEAR
    operation: MatmulOperationConfig
    weight: DenseWeightConfig


class LayerNormLayerConfig(_LayerConfigBase):
    """
    LayerNormLayerConfig provides the layer normalization layer configuration.
    """
    type: Literal[LayerType.LAYER_NORM] = LayerType.LAYER_NORM
    operation: LayerNormOperationConfig
    weight: NormWeightConfig


class RMSNormLayerConfig(_LayerConfigBase):
    """
    RMSNormLayerConfig provides RMSNorm layer configuration.
    """
    type: Literal[LayerType.RMS_NORM] = LayerType.RMS_NORM
    operation: RMSNormOperationConfig
    weight: RMSNormWeightConfig


class MultiheadLayerConfig(_LayerConfigBase):
    """
    MultiheadLayerConfig provides the multihead layer configuration.
    """
    type: Literal[LayerType.MULTIHEAD] = LayerType.MULTIHEAD
    operation: MultiheadOperationConfig
    weight: MultiheadWeightConfig


class DropoutLayerConfig(_LayerConfigBase):
    """
    DropoutLayerConfig provides the dropout layer configuration.
    """
    type: Literal[LayerType.DROPOUT] = LayerType.DROPOUT
    operation: DropoutOperationConfig


class AttentionLayerConfig(_LayerConfigBase):
    """
    AttentionLayerConfig provides attention layer configuration.
    """

    type: Literal[LayerType.ATTENTION] = LayerType.ATTENTION
    operation: AttentionOperationConfig
    weight: LlamaAttentionWeightConfig | DecoupledAttentionWeightConfig


class SwiGLULayerConfig(_LayerConfigBase):
    """
    SwiGLULayerConfig provides SwiGLU MLP configuration.
    """
    type: Literal[LayerType.SWIGLU] = LayerType.SWIGLU
    operation: SwiGLUOperationConfig
    weight: SwiGLUWeightConfig


LayerConfig: TypeAlias = Annotated[
    LinearLayerConfig
    | LayerNormLayerConfig
    | RMSNormLayerConfig
    | MultiheadLayerConfig
    | DropoutLayerConfig
    | AttentionLayerConfig
    | SwiGLULayerConfig,
    Field(discriminator="type"),
]
