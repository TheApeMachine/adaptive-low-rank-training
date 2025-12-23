"""
operation provides operation configuration models.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field


class OperationType(str, enum.Enum):
    """
    OperationType provides the operation type.
    """

    MATMUL = "matmul"
    LAYER_NORM = "layer_norm"
    DROPOUT = "dropout"
    MULTIHEAD = "multihead"
    ATTENTION = "attention"


class _OperationConfigBase(BaseModel):
    """
    _OperationConfigBase provides the base type for operation configs.
    """


class MatmulOperationConfig(_OperationConfigBase):
    """
    MatmulOperationConfig provides the matmul operation configuration.
    """

    type: Literal[OperationType.MATMUL] = OperationType.MATMUL


class LayerNormOperationConfig(_OperationConfigBase):
    """
    LayerNormOperationConfig provides the layer norm operation configuration.
    """

    type: Literal[OperationType.LAYER_NORM] = OperationType.LAYER_NORM
    eps: float = 1e-5


class DropoutOperationConfig(_OperationConfigBase):
    """
    DropoutOperationConfig provides the dropout operation configuration.
    """

    type: Literal[OperationType.DROPOUT] = OperationType.DROPOUT
    p: float = 0.0


class MultiheadOperationConfig(_OperationConfigBase):
    """
    MultiheadOperationConfig provides the multihead operation configuration.
    """

    type: Literal[OperationType.MULTIHEAD] = OperationType.MULTIHEAD


class AttentionOperationConfig(_OperationConfigBase):
    """
    AttentionOperationConfig provides attention operation configuration.
    """

    type: Literal[OperationType.ATTENTION] = OperationType.ATTENTION
    is_causal: bool = True
    dropout_p: float = 0.0


OperationConfig: TypeAlias = Annotated[
    MatmulOperationConfig
    | LayerNormOperationConfig
    | DropoutOperationConfig
    | MultiheadOperationConfig
    | AttentionOperationConfig,
    Field(discriminator="type"),
]


