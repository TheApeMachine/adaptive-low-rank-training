from __future__ import annotations

from caramba.instrumentation.live_plotter import LivePlotter


def test_live_plotter_best_effort_no_crash() -> None:
    lp = LivePlotter(enabled=True, title="test", plot_every=1)
    # Even if matplotlib isn't installed, this should not raise.
    lp.update(step=1, scalars={"loss": 1.0})
    lp.close()

