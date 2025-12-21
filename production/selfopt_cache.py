from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def load_selfopt_cache(path: Optional[str]) -> Dict[str, Any]:
    p = str(path or "").strip()
    if not p or not os.path.exists(p):
        return {}
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, dict) else {}
    except Exception:
        return {}


def update_selfopt_cache(path: str, update_fn: Callable[[Dict[str, Any]], None]) -> None:
    p = str(path).strip()
    if not p:
        raise ValueError("cache path is empty")
    root = load_selfopt_cache(p)
    update_fn(root)
    try:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    Path(p).write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_cache_entry(path: Optional[str], *, section: str, key: str) -> Optional[Any]:
    p = str(path or "").strip()
    if not p:
        return None
    root = load_selfopt_cache(p)
    sec = root.get(str(section), None)
    if not isinstance(sec, dict):
        return None
    return sec.get(str(key))


def set_cache_entry(path: str, *, section: str, key: str, value: Any) -> None:
    def _upd(root: Dict[str, Any]) -> None:
        sec_name = str(section)
        sec = root.get(sec_name, None)
        if not isinstance(sec, dict):
            sec = {}
        sec[str(key)] = value
        root[sec_name] = sec

    update_selfopt_cache(path, _upd)

