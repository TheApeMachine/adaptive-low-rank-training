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


class TestBenchModesSmall(unittest.TestCase):
    def _write_ckpt(self, td: Path, *, attn_mode: str, kv_head=None) -> Path:
        cfg = ModelConfig(
            vocab_size=64,
            block_size=64,
            n_layer=2,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode=str(attn_mode),
            attn_dim=32,
            kv_head=kv_head,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=False,
            tie_qk=True,
        )
        p = td / f"{attn_mode}.pt"
        torch.save({"config": asdict(cfg)}, str(p))
        return p

    def _run(self, ckpt: Path, out: Path, *, mode: str) -> dict:
        from production.bench_end_to_end_memory import main as bench_main

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
                str(mode),
                "--kv-kind",
                "fp16",
            ]
        )
        self.assertEqual(int(rc), 0)
        return json.loads(out.read_text(encoding="utf-8"))

    def test_standard_mode_outputs_selected_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ck = self._write_ckpt(td, attn_mode="standard")
            out = td / "out.json"
            payload = self._run(ck, out, mode="standard")
            self.assertIn("measured", payload)
            self.assertIn("selected", payload["measured"])

    def test_gqa_mode_outputs_selected_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ck = self._write_ckpt(td, attn_mode="gqa", kv_head=2)
            out = td / "out.json"
            payload = self._run(ck, out, mode="gqa")
            self.assertIn("measured", payload)
            self.assertIn("selected", payload["measured"])

    def test_bottleneck_mode_outputs_selected_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ck = self._write_ckpt(td, attn_mode="bottleneck")
            out = td / "out.json"
            payload = self._run(ck, out, mode="bottleneck")
            self.assertIn("measured", payload)
            self.assertIn("selected", payload["measured"])


if __name__ == "__main__":
    unittest.main()


