from __future__ import annotations

import torch

from caramba.config.layer import LinearLayerConfig
from caramba.config.topology import StackedTopologyConfig


def test_activation_checkpointing_forward_backward() -> None:
    cfg = StackedTopologyConfig(
        layers=[
            LinearLayerConfig(d_in=8, d_out=8),
            LinearLayerConfig(d_in=8, d_out=8),
        ],
        repeat=1,
    )
    topo = cfg.build()
    # Enable checkpointing to exercise the wrapper path.
    topo.activation_checkpointing = True  # type: ignore[attr-defined]
    topo.activation_checkpoint_threshold_mb = 0.0  # type: ignore[attr-defined]

    x = torch.randn(2, 4, 8, requires_grad=True)
    y = topo(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None

