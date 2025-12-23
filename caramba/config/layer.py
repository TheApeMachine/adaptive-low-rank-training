"""
layer provides the layer configuration.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias
from pydantic import BaseModel, Field

from caramba.config.operation import (
    LayerNormOperationConfig,
    MatmulOperationConfig,
    MultiheadOperationConfig,
    DropoutOperationConfig,
    AttentionOperationConfig,
)
from caramba.config.weight import (
    DecoupledAttentionWeightConfig,
    DenseWeightConfig,
    LlamaAttentionWeightConfig,
    MultiheadWeightConfig,
    NormWeightConfig,
)


class LayerType(str, enum.Enum):
    """
    LayerType provides the layer type.
    """
    LAYER_NORM = "layer_norm"
    LINEAR = "linear"
    MULTIHEAD = "multihead"
    DROPOUT = "dropout"
    ATTENTION = "attention"


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


LayerConfig: TypeAlias = Annotated[
    LinearLayerConfig
    | LayerNormLayerConfig
    | MultiheadLayerConfig
    | DropoutLayerConfig
    | AttentionLayerConfig,
    Field(discriminator="type"),
]
