"""Training and inference instrumentation utilities.

Instrumentation provides structured logging and optional integrations that help
make experiments reproducible and debuggable. The goal is to capture the same
signals we print to console, but in machine-readable formats for analysis.
"""

from __future__ import annotations

from caramba.instrumentation.hdf5_store import H5Store
from caramba.instrumentation.analysis import generate_analysis_png
from caramba.instrumentation.live_plotter import LivePlotter
from caramba.instrumentation.run_logger import RunLogger
from caramba.instrumentation.tensorboard_writer import TensorBoardWriter
from caramba.instrumentation.wandb_writer import WandBWriter

__all__ = [
    "H5Store",
    "generate_analysis_png",
    "LivePlotter",
    "RunLogger",
    "TensorBoardWriter",
    "WandBWriter",
]

