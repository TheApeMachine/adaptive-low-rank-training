import unittest
from unittest.mock import patch

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.model import GPT, ModelConfig


class TestGQAKVCacheDim(unittest.TestCase):
    def _tiny_cfg(self, *, mode: str) -> ModelConfig:
        # Keep dims divisible: attn_dim % n_head == 0.
        return ModelConfig(
            vocab_size=64,
            block_size=32,
            n_layer=2,
            n_head=4,
            d_model=16,
            d_ff=32,
            embed_dim=16,
            attn_mode=str(mode),
            attn_dim=16,
            kv_head=2 if mode == "gqa" else None,
            sem_dim=8,
            geo_dim=8,
            rope=False,
            learned_temp=False,
            dropout=0.0,
        )

    def test_gqa_uses_kv_head_dim_not_attn_dim(self) -> None:
        cfg = self._tiny_cfg(mode="gqa")
        m = GPT(cfg).eval()
        prompt = torch.zeros((1, 4), dtype=torch.long)

        created = []

        # Wrap the real LayerKVCache constructor to capture k_dim/v_dim used by GPT.generate.
        import production.model as model_mod

        real_cls = model_mod.LayerKVCache

        def _wrap(*args, **kwargs):
            created.append((int(kwargs["k_dim"]), int(kwargs["v_dim"])))
            return real_cls(*args, **kwargs)

        with patch("production.model.LayerKVCache", side_effect=_wrap):
            _ = m.generate(prompt, max_new_tokens=0, kv_cache="fp16", kv_qblock=32, kv_residual=0, self_opt=None)

        self.assertEqual(len(created), int(cfg.n_layer))
        # head_dim = attn_dim / n_head = 4; kv_dim = kv_head * head_dim = 2 * 4 = 8.
        self.assertTrue(all(kd == 8 and vd == 8 for kd, vd in created))

    def test_standard_uses_d_model_dim(self) -> None:
        cfg = self._tiny_cfg(mode="standard")
        m = GPT(cfg).eval()
        prompt = torch.zeros((1, 4), dtype=torch.long)

        created = []
        import production.model as model_mod

        real_cls = model_mod.LayerKVCache

        def _wrap(*args, **kwargs):
            created.append((int(kwargs["k_dim"]), int(kwargs["v_dim"])))
            return real_cls(*args, **kwargs)

        with patch("production.model.LayerKVCache", side_effect=_wrap):
            _ = m.generate(prompt, max_new_tokens=0, kv_cache="fp16", kv_qblock=32, kv_residual=0, self_opt=None)

        self.assertEqual(len(created), int(cfg.n_layer))
        self.assertTrue(all(kd == int(cfg.d_model) and vd == int(cfg.d_model) for kd, vd in created))

    def test_bottleneck_uses_attn_dim_dim(self) -> None:
        cfg = self._tiny_cfg(mode="bottleneck")
        m = GPT(cfg).eval()
        prompt = torch.zeros((1, 4), dtype=torch.long)

        created = []
        import production.model as model_mod

        real_cls = model_mod.LayerKVCache

        def _wrap(*args, **kwargs):
            created.append((int(kwargs["k_dim"]), int(kwargs["v_dim"])))
            return real_cls(*args, **kwargs)

        with patch("production.model.LayerKVCache", side_effect=_wrap):
            _ = m.generate(prompt, max_new_tokens=0, kv_cache="fp16", kv_qblock=32, kv_residual=0, self_opt=None)

        self.assertEqual(len(created), int(cfg.n_layer))
        self.assertTrue(all(kd == int(cfg.attn_dim) and vd == int(cfg.attn_dim) for kd, vd in created))


if __name__ == "__main__":
    unittest.main()


