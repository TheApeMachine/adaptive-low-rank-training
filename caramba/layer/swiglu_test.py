"""
swiglu_test provides tests for SwiGLU.
"""
from __future__ import annotations

import unittest
import torch

from caramba.config.layer import LayerType, SwiGLULayerConfig
from caramba.config.operation import SwiGLUOperationConfig
from caramba.config.weight import SwiGLUWeightConfig, WeightType
from caramba.layer.swiglu import SwiGLU


class SwiGLUTest(unittest.TestCase):
    """
    SwiGLUTest provides tests for SwiGLU.
    """
    def test_forward_shape(self) -> None:
        """
        test SwiGLU output shape.
        """
        cfg = SwiGLULayerConfig(
            type=LayerType.SWIGLU,
            operation=SwiGLUOperationConfig(),
            weight=SwiGLUWeightConfig(type=WeightType.SWIGLU, d_model=8, d_ff=16),
        )
        layer = SwiGLU(cfg)
        x = torch.randn(2, 3, 8)
        y = layer.forward(x)
        self.assertEqual(tuple(y.shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()


