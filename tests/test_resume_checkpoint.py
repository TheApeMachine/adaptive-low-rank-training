import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"numpy is required for these tests but is not available: {e}")

import subprocess
import sys


class TestResumeCheckpoint(unittest.TestCase):
    def test_resume_advances_opt_step(self) -> None:
        raise unittest.SkipTest(
            "Legacy main.py CLI checkpoint contract is no longer supported in the caramba entrypoint. "
            "Checkpoint/resume behavior is covered by caramba/trainer/upcycle_checkpoint_test.py."
        )
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            toks = (np.arange(0, 512, dtype=np.int64) % 64).astype(np.int64)
            data_path = td / "tokens.npy"
            np.save(str(data_path), toks)

            out_dir = td / "run"
            out_dir.mkdir(parents=True, exist_ok=True)

            # First: run 2 optimizer steps and save each step.
            cmd1 = [
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
                "2",
            ]
            p1 = subprocess.run(cmd1, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
            self.assertEqual(p1.returncode, 0, msg=p1.stdout + "\n" + p1.stderr)

            last = out_dir / "last.pt"
            self.assertTrue(last.exists())

            # Second: resume and run up to opt_step=4.
            cmd2 = cmd1.copy()
            # Update total steps and add resume.
            cmd2[cmd2.index("--steps") + 1] = "4"
            cmd2.append("--resume")

            p2 = subprocess.run(cmd2, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
            self.assertEqual(p2.returncode, 0, msg=p2.stdout + "\n" + p2.stderr)

            import torch  # type: ignore

            ck = torch.load(str(last), map_location="cpu")
            self.assertIn("opt", ck)
            self.assertIn("opt_step", ck)
            self.assertEqual(int(ck["opt_step"]), 4)

    # Note: the production CLI is intentionally minimal; we do not expose a
    # `--layers` override or config-mismatch override flag in the paper-focused flow.


if __name__ == "__main__":
    unittest.main()
