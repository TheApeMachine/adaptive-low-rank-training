from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def append_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    """Append a single JSON record to a .jsonl file (best-effort)."""
    if not path:
        return
    try:
        rec = dict(record)
        rec.setdefault("ts", float(time.time()))
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        with open(str(path), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    except Exception:
        # Logging must never break training.
        return


