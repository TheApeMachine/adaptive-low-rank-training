import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.model import GPT, ModelConfig


class TestAttentionForwardSmoke(unittest.TestCase):
    def _cfg(self, *, attn_mode: str, null_attn: bool) -> ModelConfig:
        return ModelConfig(
            vocab_size=128,
            block_size=32,
            n_layer=2,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode=str(attn_mode),
            attn_dim=32,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=bool(null_attn),
            tie_qk=True,
        )

    def test_forward_standard_shape(self) -> None:
        m = GPT(self._cfg(attn_mode="standard", null_attn=False)).eval()
        x = torch.randint(0, 128, (2, 8), dtype=torch.long)
        logits, caches = m(x)
        self.assertEqual(tuple(logits.shape), (2, 8, 128))
        self.assertIsNone(caches)

    def test_forward_decoupled_shape_no_null(self) -> None:
        m = GPT(self._cfg(attn_mode="decoupled", null_attn=False)).eval()
        x = torch.randint(0, 128, (2, 8), dtype=torch.long)
        logits, caches = m(x)
        self.assertEqual(tuple(logits.shape), (2, 8, 128))
        self.assertIsNone(caches)
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_forward_decoupled_shape_with_null(self) -> None:
        # Null-attn is intentionally not the flagship path, but it should remain correct.
        m = GPT(self._cfg(attn_mode="decoupled", null_attn=True)).eval()
        x = torch.randint(0, 128, (2, 8), dtype=torch.long)
        logits, caches = m(x)
        self.assertEqual(tuple(logits.shape), (2, 8, 128))
        self.assertIsNone(caches)
        self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()


