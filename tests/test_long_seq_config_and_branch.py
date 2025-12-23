import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from unittest import mock

from production.attention_impl.decoupled_attention_impl.attention_core import DecoupledBottleneckAttention
from production.model import ModelConfig


class TestLongSeqConfigAndBranch(unittest.TestCase):
    def test_model_config_from_dict_includes_long_seq_fields(self) -> None:
        cfg = ModelConfig.from_dict(
            {
                "vocab_size": 128,
                "block_size": 32,
                "n_layer": 1,
                "n_head": 4,
                "d_model": 32,
                "d_ff": 64,
                "embed_dim": 32,
                "attn_mode": "decoupled",
                "attn_dim": 32,
                "sem_dim": 16,
                "geo_dim": 16,
                "rope": False,
                "learned_temp": False,
                "dropout": 0.0,
                "null_attn": False,
                "tie_qk": True,
                "train_long_seq_enabled": False,
                "train_long_seq_threshold": 123,
                "train_long_seq_mem_block": 9,
                "train_long_seq_local_window": 456,
                "train_long_seq_q_chunk": 7,
                "train_long_seq_mem_summarizer": "conv",
            }
        )
        self.assertFalse(cfg.train_long_seq_enabled)
        self.assertEqual(cfg.train_long_seq_threshold, 123)
        self.assertEqual(cfg.train_long_seq_mem_block, 9)
        self.assertEqual(cfg.train_long_seq_local_window, 456)
        self.assertEqual(cfg.train_long_seq_q_chunk, 7)
        self.assertEqual(cfg.train_long_seq_mem_summarizer, "conv")

    def test_long_seq_branch_uses_chunked_sdpa_when_enabled(self) -> None:
        cfg = ModelConfig(
            vocab_size=128,
            block_size=32,
            n_layer=1,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode="decoupled",
            attn_dim=32,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=False,
            tie_qk=True,
            train_long_seq_enabled=True,
            train_long_seq_threshold=0,
            train_long_seq_mem_block=2,
            train_long_seq_local_window=4,
            train_long_seq_q_chunk=2,
        )

        attn = DecoupledBottleneckAttention(cfg).train()
        with mock.patch.object(attn, "_sdp", wraps=attn._sdp) as mocked_sdp:
            x = torch.randn(2, 7, cfg.d_model)
            y, cache = attn(x, attn_mask=None, cache=None, pos_offset=0)
            self.assertIsNone(cache)
            self.assertEqual(tuple(y.shape), (2, 7, cfg.d_model))
            self.assertGreater(int(mocked_sdp.call_count), 1)

    def test_learned_mem_summarizers_match_mean_at_init(self) -> None:
        cfg = ModelConfig(
            vocab_size=128,
            block_size=32,
            n_layer=1,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode="decoupled",
            attn_dim=32,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=False,
            tie_qk=True,
            train_long_seq_enabled=True,
            train_long_seq_threshold=0,
            train_long_seq_mem_block=2,
            train_long_seq_local_window=4,
            train_long_seq_q_chunk=2,
        )
        torch.manual_seed(123)
        x = torch.randn(2, 7, cfg.d_model)

        torch.manual_seed(0)
        cfg.train_long_seq_mem_summarizer = "mean"
        attn_mean = DecoupledBottleneckAttention(cfg).train()
        y_mean, _ = attn_mean(x, attn_mask=None, cache=None, pos_offset=0)

        torch.manual_seed(0)
        cfg.train_long_seq_mem_summarizer = "linear"
        attn_linear = DecoupledBottleneckAttention(cfg).train()
        y_linear, _ = attn_linear(x, attn_mask=None, cache=None, pos_offset=0)

        torch.manual_seed(0)
        cfg.train_long_seq_mem_summarizer = "conv"
        attn_conv = DecoupledBottleneckAttention(cfg).train()
        y_conv, _ = attn_conv(x, attn_mask=None, cache=None, pos_offset=0)

        self.assertTrue(torch.allclose(y_mean, y_linear, atol=1e-6))
        self.assertTrue(torch.allclose(y_mean, y_conv, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
