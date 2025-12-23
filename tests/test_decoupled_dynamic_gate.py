import unittest
from typing import cast

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    SKIP_TORCH = True
else:
    SKIP_TORCH = False

if not SKIP_TORCH:
    from production.attention_impl.decoupled_attention_impl.attention_core import DecoupledBottleneckAttention
    from production.model import ModelConfig


@unittest.skipIf(SKIP_TORCH, "torch is required for these tests")
class TestDecoupledDynamicGate(unittest.TestCase):
    def _cfg(self) -> ModelConfig:
        return ModelConfig(
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
            decoupled_gate=True,
            decoupled_gate_dynamic=True,
        )

    def test_gate_is_neutral_at_init(self) -> None:
        attn = DecoupledBottleneckAttention(self._cfg()).eval()
        x = torch.randn(2, 5, attn.cfg.d_model, dtype=torch.float32)
        g = attn._decoupled_gate(x)
        self.assertIsNotNone(g)
        g_t = cast(torch.Tensor, g)
        self.assertTrue(torch.allclose(g_t, g_t.new_full(g_t.shape, 0.5), rtol=0.0, atol=0.0))

    def test_gate_is_token_local(self) -> None:
        attn = DecoupledBottleneckAttention(self._cfg()).eval()
        proj = attn.decoupled_gate_proj
        self.assertIsNotNone(proj)
        proj_t = cast(torch.nn.Linear, proj)
        with torch.no_grad():
            proj_t.weight.zero_()
            proj_t.weight[0, 0] = 1.0

        x1 = torch.zeros(1, 3, attn.cfg.d_model, dtype=torch.float32)
        x2 = x1.clone()
        x2[0, 1, 0] = 1.0

        g1 = attn._decoupled_gate(x1)
        g2 = attn._decoupled_gate(x2)
        self.assertIsNotNone(g1)
        self.assertIsNotNone(g2)
        g1_t = cast(torch.Tensor, g1)
        g2_t = cast(torch.Tensor, g2)

        # gate shape: [batch, heads, seq_len, features]
        self.assertTrue(torch.allclose(g1_t[:, :, 0, :], g2_t[:, :, 0, :], rtol=0.0, atol=0.0))
        self.assertTrue(torch.allclose(g1_t[:, :, 2, :], g2_t[:, :, 2, :], rtol=0.0, atol=0.0))
        self.assertFalse(torch.allclose(g1_t[:, 0:1, 1, :], g2_t[:, 0:1, 1, :], rtol=0.0, atol=0.0))
        self.assertTrue(torch.allclose(g1_t[:, 1:, 1, :], g2_t[:, 1:, 1, :], rtol=0.0, atol=0.0))


if __name__ == "__main__":
    unittest.main()

