"""Structured JSONL logger for experiments.

Console logs are great for humans but hard to analyze programmatically.
This module writes machine-readable JSONL events (one JSON object per line)
so training and verification metrics can be plotted, compared, and audited.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caramba.instrumentation.hdf5_store import H5Store
from caramba.instrumentation.utils import coerce_jsonable, dumps_json, now_s


@dataclass(frozen=True, slots=True)
class JSONLEvent:
    """Single JSONL event record.

    Events are intentionally flat and schema-light: downstream consumers can
    evolve without requiring strict versioning.
    """

    type: str
    ts: float
    run_id: str | None
    phase: str | None
    step: int | None
    data: dict[str, Any]


class RunLogger:
    """Append-only JSONL writer for training/verification metrics.

    Why this exists:
    - Training loops should emit structured metrics for later analysis.
    - Logging must be best-effort: failures should never crash training.
    - JSONL is easy to stream, tail, and parse incrementally.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        filename: str = "train.jsonl",
        hdf5_filename: str = "train.h5",
        enable_hdf5: bool = True,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / filename
        self._fh: Any | None = None
        self.h5 = H5Store(
            self.out_dir / hdf5_filename,
            enabled=bool(self.enabled and enable_hdf5),
        )

        if not self.enabled:
            return

        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we can't create dirs, disable logging but don't crash.
            self.enabled = False
            return

        try:
            # Line-buffered text append.
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        except Exception:
            self.enabled = False
            self._fh = None

    def _write_line(self, line: str) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            self._fh.write(line + "\n")
        except Exception:
            # Best-effort: disable after failure to avoid repeated exceptions.
            self.enabled = False
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def log_event(
        self,
        *,
        type: str,
        data: dict[str, Any],
        run_id: str | None = None,
        phase: str | None = None,
        step: int | None = None,
    ) -> None:
        """Write an arbitrary JSONL event."""

        if not self.enabled:
            return

        ev = {
            "type": str(type),
            "ts": float(now_s()),
            "pid": int(os.getpid()),
            "run_id": None if run_id is None else str(run_id),
            "phase": None if phase is None else str(phase),
            "step": None if step is None else int(step),
            "data": coerce_jsonable(data),
        }
        self._write_line(dumps_json(ev))

    def log_metrics(
        self,
        *,
        run_id: str,
        phase: str,
        step: int,
        metrics: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Convenience wrapper for metric events."""

        payload: dict[str, Any] = {"metrics": metrics}
        if extra:
            payload["extra"] = extra
        self.log_event(type="metrics", data=payload, run_id=run_id, phase=phase, step=step)

    def close(self) -> None:
        """Close the underlying file handle."""

        if self._fh is None:
            # Still close HDF5 store if it exists.
            self.h5.close()
            return
        try:
            self._fh.flush()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None
        self.h5.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def h5_write_step(self, *, step: int, arrays: dict[str, object]) -> None:
        """Write dense arrays to the HDF5 store under this step.

        This is a best-effort hook used by future instrumentation:
        activations, gradients, histograms, etc.
        """

        self.h5.write_step(int(step), arrays)
