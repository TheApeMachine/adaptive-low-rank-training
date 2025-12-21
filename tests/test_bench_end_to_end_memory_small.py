import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.model import ModelConfig


class TestBenchEndToEndMemorySmall(unittest.TestCase):
    def test_bench_decomposition_runs_on_cpu_with_tiny_context(self) -> None:
        # This validates the paper-critical decomposition plumbing without allocating huge caches.
        from production.bench_end_to_end_memory import main as bench_main

        cfg = ModelConfig(
            vocab_size=64,
            block_size=64,
            n_layer=2,
            n_head=2,
            # Important: make d_model > attn_dim so the architecture-only factor is > 1.0.
            # (standard KV uses d_model; decoupled KV uses attn_dim/sem+geo)
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode="decoupled",
            attn_dim=16,
            sem_dim=8,
            geo_dim=8,
            rope=False,
            learned_temp=False,
            dropout=0.0,
        )

        with tempfile.TemporaryDirectory() as td:
            td = str(td)
            ckpt = Path(td) / "ckpt.pt"
            out = Path(td) / "mem.json"

            torch.save({"config": asdict(cfg)}, str(ckpt))

            rc = bench_main(
                [
                    "--ckpt",
                    str(ckpt),
                    "--device",
                    "cpu",
                    "--context-len",
                    "32",
                    "--batch-size",
                    "1",
                    "--out",
                    str(out),
                    "--mode",
                    "decoupled",
                    "--decompose",
                    "--baseline-mode",
                    "standard",
                    "--policy",
                    "ksem=q4_0@32,kgeo=q8_0@32,v=q4_0@32,resid=8",
                ]
            )
            self.assertEqual(int(rc), 0)

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("decomposition", payload)
            dec = payload["decomposition"]
            self.assertIn("estimate_bytes", dec)
            est = dec["estimate_bytes"]
            self.assertGreater(float(est["ratio_arch_standard_over_decoupled_fp16"]), 1.0)
            self.assertGreater(float(est["ratio_quant_decoupled_fp16_over_candidate"]), 1.0)
            self.assertGreater(float(est["ratio_e2e_standard_over_candidate"]), 1.0)


if __name__ == "__main__":
    unittest.main()


