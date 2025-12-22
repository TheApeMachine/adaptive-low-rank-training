"""Best-effort structured logging for self-optimization + training runtime."""

from __future__ import annotations

import json
import os
import time


def append_jsonl(path: str | None, record: dict[str, object]) -> None:
    """Append a single JSON record to a .jsonl file (best-effort)."""
    if not path:
        return
    try:
        rec = dict(record)
        _ = rec.setdefault("ts", float(time.time()))
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        with open(str(path), "a", encoding="utf-8") as f:
            _ = f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        # Logging must never break training.
        return


def _call_best_effort(fn: object, *args: object, **kwargs: object) -> object | None:
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except (OSError, TypeError, ValueError, AttributeError, RuntimeError):
        return None


class SelfOptLogger:
    """Unified event logger used by the runner and self-optimization components.

    Goals:
    - One callsite API for emitting structured events
    - Optional JSONL persistence (best-effort)
    - Optional forwarding to `RunLogger` (TensorBoard/W&B/plots)
    - Optional human-readable echo (best-effort)
    """

    def __init__(
        self,
        *,
        jsonl_path: str | None = None,
        run_logger: object | None = None,
        echo: bool = True,
    ) -> None:
        self.jsonl_path: str | None = str(jsonl_path) if jsonl_path else None
        self.run_logger: object | None = run_logger
        self.echo: bool = bool(echo)

    def log(self, event: dict[str, object], *, msg: str | None = None, echo: bool | None = None) -> None:
        """Log an event, optionally echoing a human-readable message."""
        try:
            # Always forward structured events first (so a failing print doesn't drop metrics).
            if self.run_logger is not None:
                try:
                    log_fn = getattr(self.run_logger, "log", None)
                    _ = _call_best_effort(log_fn, event)
                except (OSError, TypeError, ValueError, AttributeError, RuntimeError):
                    pass
            append_jsonl(self.jsonl_path, event)
        finally:
            do_echo = self.echo if echo is None else bool(echo)
            if do_echo and msg:
                try:
                    print(str(msg), flush=True)
                except (OSError, TypeError, ValueError):
                    pass

    def finalize(self, *args: object, **kwargs: object) -> None:
        if self.run_logger is None:
            return
        try:
            fin = getattr(self.run_logger, "finalize", None)
            _ = _call_best_effort(fin, *args, **kwargs)
        except (OSError, TypeError, ValueError, AttributeError, RuntimeError):
            pass

    def close(self) -> None:
        if self.run_logger is None:
            return
        try:
            close_fn = getattr(self.run_logger, "close", None)
            _ = _call_best_effort(close_fn)
        except (OSError, TypeError, ValueError, AttributeError, RuntimeError):
            pass


