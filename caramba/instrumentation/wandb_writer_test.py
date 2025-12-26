from __future__ import annotations

from pathlib import Path

from caramba.instrumentation.wandb_writer import WandBWriter


def test_wandb_writer_best_effort_no_crash(tmp_path: Path) -> None:
    w = WandBWriter(
        tmp_path / "wandb",
        enabled=True,
        project="",
        entity=None,
        mode="offline",
        run_name="test",
        group="g",
    )
    # Even if wandb isn't installed, these should not raise.
    w.log_scalars(prefix="train", step=1, scalars={"loss": 1.0})
    w.close()

