import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.attention import DecoupledBottleneckAttention
from production.model import ModelConfig


class TestAttentionConfigValidation(unittest.TestCase):
    def test_standard_requires_d_model_divisible_by_n_head(self) -> None:
        cfg = ModelConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=8,
            d_model=30,  # not divisible by 8
            d_ff=64,
            embed_dim=30,
            attn_mode="standard",
            attn_dim=30,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
        )
        with self.assertRaisesRegex(ValueError, r"d_model.*divisible"):
            _ = DecoupledBottleneckAttention(cfg)

    def test_gqa_rejects_tie_qk(self) -> None:
        cfg = ModelConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=4,
            kv_head=2,
            d_model=16,
            d_ff=32,
            embed_dim=16,
            attn_mode="gqa",
            attn_dim=16,
            sem_dim=8,
            geo_dim=8,
            rope=False,
            tie_qk=True,
            learned_temp=False,
            dropout=0.0,
        )
        with self.assertRaisesRegex(ValueError, r"tie_qk.*gqa"):
            _ = DecoupledBottleneckAttention(cfg)

    def test_decoupled_requires_sem_dim_divisible_by_n_head(self) -> None:
        cfg = ModelConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=4,
            d_model=16,
            d_ff=32,
            embed_dim=16,
            attn_mode="decoupled",
            attn_dim=16,
            sem_dim=18,  # not divisible by 4
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
        )
        with self.assertRaisesRegex(ValueError, r"sem_dim.*divisible"):
            _ = DecoupledBottleneckAttention(cfg)

    def test_bottleneck_rope_requires_even_head_dim(self) -> None:
        # head_dim = attn_dim / n_head = 3 -> odd, RoPE requires even.
        cfg = ModelConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=3,
            d_model=12,
            d_ff=24,
            embed_dim=12,
            attn_mode="bottleneck",
            attn_dim=9,
            sem_dim=6,
            geo_dim=6,
            rope=True,
            learned_temp=False,
            dropout=0.0,
        )
        with self.assertRaisesRegex(ValueError, r"RoPE.*even head dim"):
            _ = DecoupledBottleneckAttention(cfg)


if __name__ == "__main__":
    unittest.main()

