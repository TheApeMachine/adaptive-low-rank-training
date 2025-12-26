from __future__ import annotations

import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(
        f"torch is required for these tests but is not available: {e}"
    )

from caramba.config.layer import AttentionLayerConfig, AttentionMode, LayerType
from caramba.layer.attention import AttentionLayer


class TestCarambaAttention(unittest.TestCase):
    def test_gqa_shapes(self) -> None:
        torch.manual_seed(0)

        cfg = AttentionLayerConfig(
            type=LayerType.ATTENTION,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            mode=AttentionMode.GQA,
            rope_enabled=True,
            rope_base=10_000.0,
            bias=False,
            dropout_p=0.0,
            is_causal=True,
        )

        layer = AttentionLayer(cfg)

        x = torch.randn(2, 5, 16)
        y, cache = layer(x)

        self.assertEqual(tuple(y.shape), (2, 5, 16))
        self.assertIsNone(cache)

    def test_decoupled_rope_only_geo_shapes(self) -> None:
        torch.manual_seed(0)

        cfg = AttentionLayerConfig(
            type=LayerType.ATTENTION,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            mode=AttentionMode.DECOUPLED,
            sem_dim=8,
            geo_dim=8,
            rope_enabled=True,
            rope_base=10_000.0,
            decoupled_gate=True,
            bias=False,
            dropout_p=0.0,
            is_causal=True,
        )

        layer = AttentionLayer(cfg)
        # RoPE should be attached to the geometric path in decoupled mode.
        self.assertIsNotNone(layer.rotary_geo)

        x = torch.randn(2, 5, 16)
        y, cache = layer(x)

        self.assertEqual(tuple(y.shape), (2, 5, 16))
        self.assertIsNone(cache)


if __name__ == "__main__":
    unittest.main()


