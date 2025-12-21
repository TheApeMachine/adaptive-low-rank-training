import io
import unittest
from contextlib import redirect_stderr

from production.cli import parse_args


class TestCLIParseArgs(unittest.TestCase):
    def test_minimal_parse_accepts_intent_flags(self) -> None:
        args = parse_args(["--mode", "train", "--out-dir", "runs/tiny_baseline_test", "--exp", "paper_baseline"])
        self.assertEqual(args.mode, "train")
        self.assertEqual(args.exp, "paper_baseline")

    def test_minimal_parse_rejects_expert_only_flags_with_hint(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf), self.assertRaises(SystemExit):
            _ = parse_args(["--attn-mode", "standard"])
        err = buf.getvalue()
        self.assertIn("Hint: this project uses an intent-first CLI", err)
        self.assertIn("high-level intent", err)

    def test_rejects_removed_expert_flag(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf), self.assertRaises(SystemExit):
            _ = parse_args(["--expert"])
        err = buf.getvalue()
        self.assertIn("unrecognized arguments", err)

    def test_rejects_legacy_performance_flags(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf), self.assertRaises(SystemExit):
            _ = parse_args(["--batch-size", "4"])
        err = buf.getvalue()
        self.assertIn("unrecognized arguments", err)

    # NOTE: we do NOT require parse_args() to populate runtime fields.
    # The runner derives/tunes them at runtime (self-optimizing contract).


if __name__ == "__main__":
    unittest.main()
