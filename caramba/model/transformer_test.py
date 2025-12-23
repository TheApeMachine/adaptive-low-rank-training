"""
transformer_test provides tests to validate the
transformer model.
"""
from __future__ import annotations

import unittest
import torch
from caramba.model.transformer import Transformer
from caramba.config.topology import StackedTopologyConfig
from caramba.config.layer import (
    LinearLayerConfig,
    LayerNormLayerConfig,
    MultiheadLayerConfig,
    DropoutLayerConfig,
    LayerType,
)
from caramba.config.operation import (
    DropoutOperationConfig,
    LayerNormOperationConfig,
    MatmulOperationConfig,
    MultiheadOperationConfig,
)
from caramba.config.weight import DenseWeightConfig, MultiheadWeightConfig, NormWeightConfig

class TransformerTest(unittest.TestCase):
    """
    TransformerTest provides tests to validate the
    transformer model.
    """
    def test_forward(self) -> None:
        """
        test the forward pass of the transformer model.
        """
        transformer = Transformer(StackedTopologyConfig(layers=[
            LinearLayerConfig(
                type=LayerType.LINEAR,
                operation=MatmulOperationConfig(),
                weight=DenseWeightConfig(
                    d_in=128,
                    d_out=128,
                    bias=True,
                ),
            ),
            LayerNormLayerConfig(
                type=LayerType.LAYER_NORM,
                operation=LayerNormOperationConfig(eps=1e-5),
                weight=NormWeightConfig(
                    d_model=128,
                    elementwise_affine=True,
                ),
            ),
            MultiheadLayerConfig(
                type=LayerType.MULTIHEAD,
                operation=MultiheadOperationConfig(),
                weight=MultiheadWeightConfig(
                    d_model=128,
                    n_heads=4,
                    dropout=0.1,
                ),
            ),
            DropoutLayerConfig(
                type=LayerType.DROPOUT,
                operation=DropoutOperationConfig(p=0.1),
            ),
        ]))

        x: torch.Tensor = torch.randn(1, 10, 128)
        self.assertEqual(transformer(x).shape, (1, 10, 128))