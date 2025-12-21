import os
import tempfile
import unittest
from pathlib import Path


class TestSelfOptContract(unittest.TestCase):
    def test_no_env_controls_exist(self) -> None:
        # Contract: no env toggles for core optimization behavior.
        forbidden = {
            "EXPERIMENTS_NO_AMP",
            "EXPERIMENTS_NO_COMPILE",
            "EXPERIMENTS_FORCE_FP32",
        }
        for k in forbidden:
            self.assertNotIn(k, os.environ, msg=f"{k} should not be used as a control surface")

    def test_decision_log_is_written_on_train(self) -> None:
        # Run a tiny end-to-end train for 1 step and confirm selfopt_decisions.jsonl exists.
        try:
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"numpy required: {e}")

        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            toks = (np.arange(0, 2048, dtype=np.int64) % 64).astype(np.int64)
            data_path = td / "tokens.npy"
            np.save(str(data_path), toks)

            out_dir = td / "run"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "main.py"),
                "--mode",
                "train",
                "--out-dir",
                str(out_dir),
                "--data",
                str(data_path),
                "--exp",
                "paper_baseline",
                "--size",
                "1m",
                "--steps",
                "1",
            ]
            p = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, msg=p.stdout + "\n" + p.stderr)

            log_path = out_dir / "selfopt_decisions.jsonl"
            self.assertTrue(log_path.exists(), msg="selfopt_decisions.jsonl should be created for every training run")
            raw = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(raw), 1)


