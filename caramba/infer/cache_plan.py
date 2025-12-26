"""Persistence for KV-cache auto-selection decisions.

The "auto" cache kind can run a micro-benchmark to select a quantization policy.
That is useful, but re-running it every time wastes compute. This module
stores the chosen policy keyed by a stable signature of (model + settings),
so subsequent runs can reuse the decision instantly.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from caramba.config.kvcache import KVCacheKind
from caramba.runtime.plan import make_plan_key


def _stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_plan_payload(
    *,
    model: object,
    batch_size: int,
    max_seq_len: int,
    qblock: int,
    residual_len: int,
    budget_mb: float | None,
    quality_max_delta_nll: float | None,
    quality_max_ppl_ratio: float | None,
    quality_max_mean_kl: float | None,
    quality_prompt_len: int,
    quality_decode_steps: int,
    auto_benchmark: bool,
    auto_bench_steps: int,
    auto_bench_prompt_len: int,
) -> dict[str, Any]:
    """Build the payload used to key a cached cache-kind decision.

    Why this exists:
    - We want cache reuse across runs, but only when the (model + settings)
      signature matches.
    """

    model_name = type(model).__module__ + "." + type(model).__qualname__
    return {
        "model": model_name,
        "batch_size": int(batch_size),
        "max_seq_len": int(max_seq_len),
        "qblock": int(qblock),
        "residual_len": int(residual_len),
        "budget_mb": None if budget_mb is None else float(budget_mb),
        "quality": {
            "max_delta_nll": None if quality_max_delta_nll is None else float(quality_max_delta_nll),
            "max_ppl_ratio": None if quality_max_ppl_ratio is None else float(quality_max_ppl_ratio),
            "max_mean_kl": None if quality_max_mean_kl is None else float(quality_max_mean_kl),
            "prompt_len": int(quality_prompt_len),
            "decode_steps": int(quality_decode_steps),
        },
        "auto_benchmark": bool(auto_benchmark),
        "auto_bench_steps": int(auto_bench_steps),
        "auto_bench_prompt_len": int(auto_bench_prompt_len),
    }


def cache_plan_key(payload: dict[str, Any]) -> str:
    """Make a stable key for the cache plan payload."""
    return str(make_plan_key(payload))


def load_cached_kind(path: Path, *, key: str) -> KVCacheKind | None:
    """Load a cached cache kind from disk."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    plans = blob.get("plans", None)
    if not isinstance(plans, dict):
        return None
    entry = plans.get(str(key), None)
    if not isinstance(entry, dict):
        return None
    kind = entry.get("cache_kind", None)
    if not isinstance(kind, str):
        return None
    try:
        return KVCacheKind(kind)
    except Exception:
        return None


def load_cached_entry(path: Path, *, key: str) -> dict[str, Any] | None:
    """Load a cached plan entry (kind + metadata) from disk."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    plans = blob.get("plans", None)
    if not isinstance(plans, dict):
        return None
    entry = plans.get(str(key), None)
    if not isinstance(entry, dict):
        return None
    return entry


def should_probe_entry(entry: dict[str, Any], *, interval_sec: int) -> bool:
    """Decide whether a cached entry is stale and should be re-probed."""
    interval = max(0, int(interval_sec))
    if interval <= 0:
        return True
    ts = entry.get("ts", None)
    if ts is None:
        return True
    try:
        ts_f = float(ts)
    except Exception:
        return True
    now = float(time.time())
    return (now - ts_f) >= float(interval)


def save_cached_kind(
    path: Path,
    *,
    key: str,
    kind: KVCacheKind,
    tps: float | None = None,
    source: str | None = None,
) -> None:
    """Persist a cache kind decision under a key.

    Stores a timestamp so callers can periodically re-probe and refresh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any]
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            blob = {}
    except Exception:
        blob = {}
    plans = blob.get("plans", None)
    if not isinstance(plans, dict):
        plans = {}
    entry: dict[str, Any] = {"cache_kind": str(kind.value), "ts": float(time.time())}
    if tps is not None:
        try:
            entry["tps"] = float(tps)
        except Exception:
            pass
    if source is not None:
        entry["source"] = str(source)
    plans[str(key)] = entry
    blob["version"] = 1
    blob["plans"] = plans
    path.write_text(_stable_json(blob) + "\n", encoding="utf-8")

