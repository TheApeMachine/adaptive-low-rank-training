import argparse
import unittest

try:
    import torch
except Exception as e:
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.config import apply_exp_preset
from production.model import GPT, ModelConfig
from production.model.diffusion_head import DIFFUSERS_AVAILABLE


class TestDiffusionHeadPresetAndGating(unittest.TestCase):
    def test_preset_sets_diffusion_flags(self) -> None:
        args = argparse.Namespace(exp="fun_diffusion_head", d_model=256)
        apply_exp_preset(args)
        self.assertTrue(bool(getattr(args, "diffusion_head", False)))
        self.assertEqual(str(getattr(args, "diffusion_head_scheduler", "")), "ddim")

    def test_gpt_init_diffusion_enabled_is_dependency_gated(self) -> None:
        cfg = ModelConfig(device=torch.device("cpu"))
        cfg.vocab_size = 128
        cfg.block_size = 8
        cfg.n_layer = 2
        cfg.d_model = 64
        cfg.n_head = 4
        cfg.d_ff = 256
        cfg.embed_dim = 64
        cfg.attn_mode = "standard"
        cfg.attn_dim = 64
        cfg.sem_dim = 64
        cfg.geo_dim = 0
        cfg.dropout = 0.0

        cfg.diffusion_head = True
        if not DIFFUSERS_AVAILABLE:
            with self.assertRaises(RuntimeError):
                _ = GPT(cfg)
        else:
            m = GPT(cfg)
            self.assertIsNotNone(getattr(m, "diffusion_head", None))


if __name__ == "__main__":
    unittest.main()


