"""TensorBoard writer for scalar and histogram logging.

TensorBoard is a lightweight way to visualize training progress locally.
This module provides an optional integration that:
- Uses `torch.utils.tensorboard.SummaryWriter` when available
- Degrades gracefully when tensorboard isn't installed
- Never crashes training due to logging failures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _try_summary_writer() -> object | None:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
    except Exception:
        return None
    return SummaryWriter


@dataclass
class TensorBoardWriter:
    """Best-effort TensorBoard logger."""

    out_dir: Path
    enabled: bool = True
    log_every: int = 10

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self._writer: Any | None = None
        if not self.enabled:
            return

        Writer = _try_summary_writer()
        if Writer is None:
            self.enabled = False
            return

        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.enabled = False
            return

        try:
            # We keep Writer as `object` to avoid hard dependency typing.
            self._writer = Writer(str(self.out_dir))  # type: ignore[operator]
        except Exception:
            self.enabled = False
            self._writer = None

    def should_log(self, step: int) -> bool:
        """Decide whether to log at this step."""

        if not self.enabled:
            return False
        le = int(self.log_every)
        return True if le <= 1 else (int(step) % le == 0)

    def log_scalars(self, *, prefix: str, step: int, scalars: dict[str, float]) -> None:
        """Log a dict of scalar metrics."""

        if not self.enabled or self._writer is None:
            return
        if not self.should_log(step):
            return

        try:
            for k, v in scalars.items():
                self._writer.add_scalar(f"{prefix}/{k}", float(v), int(step))
        except Exception:
            self.enabled = False

    def log_histogram(self, *, name: str, step: int, values: object) -> None:
        """Log a histogram (tensor/array) if supported."""

        if not self.enabled or self._writer is None:
            return
        if not self.should_log(step):
            return

        try:
            self._writer.add_histogram(str(name), values, int(step))
        except Exception:
            self.enabled = False

    def flush(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.flush()
        except Exception:
            pass
        try:
            self._writer.close()
        except Exception:
            pass
        self._writer = None

