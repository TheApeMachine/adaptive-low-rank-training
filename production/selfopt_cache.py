from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Callable


def _json_loads_obj(text: str) -> object:
    # `json.loads` is typed as returning `Any` in stubs; isolate it behind an `object` boundary.
    return json.loads(text)  # pyright: ignore[reportAny]


def load_selfopt_cache(path: str | None) -> dict[str, object]:
    p = str(path or "").strip()
    if not p or not os.path.exists(p):
        return {}
    try:
        obj = _json_loads_obj(Path(p).read_text(encoding="utf-8"))
        d = as_str_object_dict(obj)
        return {} if d is None else d
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def update_selfopt_cache(path: str, update_fn: Callable[[dict[str, object]], None]) -> None:
    p = str(path).strip()
    if not p:
        raise ValueError("cache path is empty")
    root = load_selfopt_cache(p)
    update_fn(root)
    try:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    # Selfopt plans may contain torch dtypes/devices; make cache writes best-effort.
    try:
        _ = Path(p).write_text(json.dumps(root, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass


def as_str_object_dict(x: object) -> dict[str, object] | None:
    """Best-effort conversion of a mapping-like object into dict[str, object]."""
    if not isinstance(x, dict):
        return None
    # Dict payloads from JSON / dynamic sources are intentionally treated as `object` values.
    return {str(k): v for k, v in x.items()}  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]


def as_object_pair2(x: object) -> tuple[object, object] | None:
    """Best-effort extraction of a length-2 pair from a list/tuple-like object."""
    x_list = as_object_list(x)
    if x_list is not None and len(x_list) == 2:
        return (x_list[0], x_list[1])
    return None


def as_object_list(x: object) -> list[object] | None:
    """Best-effort extraction of a list as list[object] (helps strict typing in callers)."""
    if not isinstance(x, list):
        return None
    return [v for v in x]  # pyright: ignore[reportUnknownVariableType]


def get_cache_entry(path: str | None, *, section: str, key: str) -> object | None:
    p = str(path or "").strip()
    if not p:
        return None
    root = load_selfopt_cache(p)
    sec = root.get(str(section), None)
    sec_d = as_str_object_dict(sec)
    if sec_d is None:
        return None
    return sec_d.get(str(key))


def set_cache_entry(path: str, *, section: str, key: str, value: object) -> None:
    def _upd(root: dict[str, object]) -> None:
        sec_name = str(section)
        sec = root.get(sec_name, None)
        sec_d = as_str_object_dict(sec) or {}
        sec_d[str(key)] = value
        root[sec_name] = sec_d

    update_selfopt_cache(path, _upd)
