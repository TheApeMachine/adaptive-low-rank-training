import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from unittest import mock

from production.attention_impl.decoupled_attention_impl.attention_core import DecoupledBottleneckAttention
from production.model import ModelConfig


class TestDecoupledOutProjCallCount(unittest.TestCase):
    def test_out_proj_called_once_for_null_prefill(self) -> None:
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
            null_attn=True,
            tie_qk=True,
        )
        attn = DecoupledBottleneckAttention(cfg).eval()

        with mock.patch.object(attn.out_proj, "forward", wraps=attn.out_proj.forward) as mocked_forward:
            x = torch.randn(2, 7, cfg.d_model)
            y, cache = attn(x, attn_mask=None, cache=None, pos_offset=0)
            self.assertIsNone(cache)
            self.assertEqual(int(mocked_forward.call_count), 1)
            self.assertEqual(tuple(y.shape), (2, 7, cfg.d_model))


if __name__ == "__main__":
    unittest.main()

