from __future__ import annotations

from pathlib import Path

from caramba.config.kvcache import KVCacheKind
from caramba.infer.cache_plan import load_cached_entry, load_cached_kind, save_cached_kind, should_probe_entry


def test_cache_plan_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cache_plans.json"
    save_cached_kind(path, key="abc", kind=KVCacheKind.Q8_0, tps=123.0, source="bench")
    k = load_cached_kind(path, key="abc")
    assert k == KVCacheKind.Q8_0
    entry = load_cached_entry(path, key="abc")
    assert entry is not None
    assert entry["cache_kind"] == "q8_0"
    assert float(entry["tps"]) == 123.0


def test_cache_plan_should_probe_when_missing_ts() -> None:
    assert should_probe_entry({"cache_kind": "fp16"}, interval_sec=3600)

