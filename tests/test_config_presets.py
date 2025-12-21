import argparse
import unittest

from production.config import (
    apply_exp_preset,
    infer_dataset_tokens_from_path,
    infer_layers_from_out_dir,
)


class TestIntentInference(unittest.TestCase):
    def test_infer_layers_from_out_dir(self) -> None:
        self.assertEqual(infer_layers_from_out_dir("runs/a100_fw1b_l22_baseline_s1337"), 22)
        self.assertEqual(infer_layers_from_out_dir("runs/mac_fw100m_l12_decoupled_s1337"), 12)
        self.assertEqual(infer_layers_from_out_dir("runs/x_layers24_y"), 24)
        self.assertIsNone(infer_layers_from_out_dir("runs/no_layers_tag"))

    def test_infer_dataset_tokens_from_path(self) -> None:
        self.assertEqual(infer_dataset_tokens_from_path("fineweb_20b.npy"), 20_000_000_000)
        self.assertEqual(infer_dataset_tokens_from_path("fineweb_100m.npy"), 100_000_000)
        self.assertIsNone(infer_dataset_tokens_from_path("fineweb.npy"))


class TestExperimentPresetDerivedDims(unittest.TestCase):
    def test_apply_exp_preset_standard_sets_attn_dim(self) -> None:
        args = argparse.Namespace(exp="paper_baseline", d_model=768)
        apply_exp_preset(args)
        self.assertEqual(args.attn_mode, "standard")
        self.assertEqual(int(args.attn_dim), 768)

    def test_apply_exp_preset_decoupled_derives_dims(self) -> None:
        # Uses derived formula (no size tables)
        args = argparse.Namespace(exp="paper_decoupled", d_model=2048, n_head=16)
        apply_exp_preset(args)
        self.assertEqual(args.attn_mode, "decoupled")
        self.assertTrue(int(args.attn_dim) > 0)
        self.assertTrue(int(args.sem_dim) > 0)
        self.assertTrue(int(args.geo_dim) > 0)
        self.assertEqual(int(args.sem_dim) + int(args.geo_dim), int(args.attn_dim))
        self.assertEqual(int(args.attn_dim) % int(args.n_head), 0)
        self.assertEqual(int(args.sem_dim) % int(args.n_head), 0)
        self.assertEqual(int(args.geo_dim) % int(args.n_head), 0)

    def test_apply_exp_preset_decoupled_small_heads_is_divisible(self) -> None:
        # Regression test: previously d_model=768,n_head=6 derived attn_dim=160, sem_dim=64, geo_dim=96
        # which violates decoupled's per-head divisibility requirements.
        args = argparse.Namespace(exp="paper_decoupled", d_model=768, n_head=6)
        apply_exp_preset(args)
        self.assertEqual(args.attn_mode, "decoupled")
        self.assertTrue(int(args.attn_dim) > 0)
        self.assertTrue(int(args.sem_dim) > 0)
        self.assertTrue(int(args.geo_dim) > 0)
        self.assertEqual(int(args.sem_dim) + int(args.geo_dim), int(args.attn_dim))
        self.assertEqual(int(args.attn_dim) % int(args.n_head), 0)
        self.assertEqual(int(args.sem_dim) % int(args.n_head), 0)
        self.assertEqual(int(args.geo_dim) % int(args.n_head), 0)


if __name__ == "__main__":
    unittest.main()
