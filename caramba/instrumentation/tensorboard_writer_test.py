from __future__ import annotations

from pathlib import Path

from caramba.instrumentation.tensorboard_writer import TensorBoardWriter


def test_tensorboard_writer_best_effort_no_crash(tmp_path: Path) -> None:
    tb = TensorBoardWriter(tmp_path / "tb", enabled=True, log_every=1)
    # Even if tensorboard isn't installed, these should not raise.
    tb.log_scalars(prefix="train", step=1, scalars={"loss": 1.0})
    tb.log_histogram(name="weights", step=1, values=[0.0, 1.0])
    tb.flush()
    tb.close()

