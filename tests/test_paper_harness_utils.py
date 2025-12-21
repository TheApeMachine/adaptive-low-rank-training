import unittest

from run_paper_manifest import _normalize_instrument_level, _parse_print_config, _validate_expected


class TestPaperHarnessUtils(unittest.TestCase):
    def test_parse_print_config_extracts_json_from_mixed_output(self) -> None:
        out = "\n".join(
            [
                "some log line",
                "{",
                '  "attn_mode": "decoupled",',
                '  "null_attn": false',
                "}",
                "tail log",
            ]
        )
        cfg = _parse_print_config(out)
        self.assertEqual(cfg["attn_mode"], "decoupled")
        self.assertEqual(cfg["null_attn"], False)

    def test_validate_expected_reports_mismatches(self) -> None:
        cfg = {"attn_mode": "standard", "null_attn": False}
        errs = _validate_expected(cfg, {"attn_mode": "decoupled", "null_attn": False})
        self.assertTrue(any("attn_mode" in e for e in errs))
        self.assertFalse(any("null_attn" in e for e in errs))

    def test_validate_expected_reports_missing_keys(self) -> None:
        cfg = {"attn_mode": "standard"}
        errs = _validate_expected(cfg, {"null_attn": False})
        self.assertTrue(any("missing key" in e for e in errs))

    def test_normalize_instrument_level(self) -> None:
        self.assertEqual(_normalize_instrument_level("full"), "rich")
        self.assertEqual(_normalize_instrument_level("rich"), "rich")
        self.assertEqual(_normalize_instrument_level("medium"), "basic")
        self.assertEqual(_normalize_instrument_level("basic"), "basic")
        self.assertEqual(_normalize_instrument_level("off"), "off")
        self.assertEqual(_normalize_instrument_level("auto"), "auto")
        self.assertEqual(_normalize_instrument_level("UNKNOWN"), "rich")


if __name__ == "__main__":
    unittest.main()


