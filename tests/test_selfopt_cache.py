import json
import tempfile
import unittest
from pathlib import Path

from production.selfopt_cache import get_cache_entry, load_selfopt_cache, set_cache_entry


class TestSelfOptCache(unittest.TestCase):
    def test_load_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "missing.json"
            self.assertEqual(load_selfopt_cache(str(p)), {})

    def test_set_preserves_other_sections(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cache.json"
            p.write_text(json.dumps({"decode_plans": {"a": 1}, "cache_policies": {"b": 2}}), encoding="utf-8")
            set_cache_entry(str(p), section="train_plans", key="k", value={"x": 3})
            root = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(root["decode_plans"]["a"], 1)
            self.assertEqual(root["cache_policies"]["b"], 2)
            self.assertEqual(root["train_plans"]["k"]["x"], 3)

    def test_get_returns_none_for_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cache.json"
            p.write_text(json.dumps({"train_plans": {"k": {"x": 1}}}), encoding="utf-8")
            self.assertIsNone(get_cache_entry(str(p), section="train_plans", key="missing"))
            self.assertIsNone(get_cache_entry(str(p), section="missing_section", key="k"))


if __name__ == "__main__":
    unittest.main()

