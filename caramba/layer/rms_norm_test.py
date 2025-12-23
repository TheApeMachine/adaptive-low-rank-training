"""
rms_norm_test provides tests for RMSNorm.
"""
from __future__ import annotations

import unittest
import torch

from caramba.config.layer import LayerType, RMSNormLayerConfig
from caramba.config.operation import RMSNormOperationConfig
from caramba.config.weight import RMSNormWeightConfig, WeightType
from caramba.layer.rms_norm import RMSNorm


class RMSNormTest(unittest.TestCase):
    """
    RMSNormTest provides tests for RMSNorm.
    """
    def test_forward_shape(self) -> None:
        """
        test RMSNorm output shape.
        """
        cfg = RMSNormLayerConfig(
            type=LayerType.RMS_NORM,
            operation=RMSNormOperationConfig(eps=1e-5),
            weight=RMSNormWeightConfig(type=WeightType.RMS_NORM, d_model=8),
        )
        layer = RMSNorm(cfg)
        x = torch.randn(2, 3, 8)
        y = layer.forward(x)
        self.assertEqual(tuple(y.shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()


