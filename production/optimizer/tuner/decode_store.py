"""Persistence for decode-plan tuning results."""

from __future__ import annotations

from dataclasses import asdict

from production.selfopt_cache import as_str_object_dict, get_cache_entry, set_cache_entry

from production.optimizer.tuner.decode_plan import KVDecodePlan


def _as_int(o: object, default: int) -> int:
    if isinstance(o, bool):
        return int(o)
    if isinstance(o, int):
        return int(o)
    if isinstance(o, float):
        return int(o)
    if isinstance(o, str):
        try:
            return int(o)
        except ValueError:
            return int(default)
    return int(default)


def _as_str(o: object, default: str) -> str:
    if isinstance(o, str):
        return o
    return str(default)


class DecodePlanStore:
    """Persist decode plans as a single JSON entry (per cache path)."""

    def __init__(self, cache_path: str | None, *, verbose: bool) -> None:
        self.cache_path: str | None = cache_path
        self.verbose: bool = bool(verbose)

    def load(self) -> dict[str, KVDecodePlan]:
        """Load all cached plans from disk (best-effort)."""
        if not self.cache_path:
            return {}
        try:
            raw = get_cache_entry(self.cache_path, section="decode_plans", key="__all__")
            raw_map = as_str_object_dict(raw)
            if raw_map is None:
                return {}
            out: dict[str, KVDecodePlan] = {}
            # `raw` is untyped JSON-ish data; parse defensively so typing stays strict.
            for k, v in raw_map.items():
                v_map = as_str_object_dict(v)
                if v_map is None:
                    continue

                fused = _as_str(v_map.get("fused", "none"), "none")
                decode_block = _as_int(v_map.get("decode_block", 0), 0)

                out[str(k)] = KVDecodePlan(
                    fused=str(fused),
                    decode_block=int(decode_block),
                    block_n=_as_int(v_map.get("block_n", 128), 128),
                    num_warps_1pass=_as_int(v_map.get("num_warps_1pass", 4), 4),
                    num_stages_1pass=_as_int(v_map.get("num_stages_1pass", 2), 2),
                    num_warps_part=_as_int(v_map.get("num_warps_part", 4), 4),
                    num_stages_part=_as_int(v_map.get("num_stages_part", 2), 2),
                    num_warps_reduce=_as_int(v_map.get("num_warps_reduce", 1), 1),
                    num_stages_reduce=_as_int(v_map.get("num_stages_reduce", 1), 1),
                )
            return out
        except (TypeError, OSError, ValueError):
            if self.verbose:
                print(f"[selfopt] Failed to load cache '{self.cache_path}'")
            return {}

    def save(self, plans: dict[str, KVDecodePlan]) -> None:
        """Save all plans to disk (best-effort)."""
        if not self.cache_path:
            return
        try:
            payload = {k: asdict(v) for k, v in plans.items()}
            set_cache_entry(str(self.cache_path), section="decode_plans", key="__all__", value=payload)
        except (OSError, ValueError, TypeError):
            if self.verbose:
                print(f"[selfopt] Failed to save cache '{self.cache_path}'")


