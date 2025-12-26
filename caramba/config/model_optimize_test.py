from __future__ import annotations

from caramba.config.embedder import TokenEmbedderConfig
from caramba.config.layer import AttentionLayerConfig, LayerType, SwiGLULayerConfig
from caramba.config.model import ModelConfig, ModelType
from caramba.config.topology import StackedTopologyConfig


def test_model_config_optimize_scales_common_transformer() -> None:
    cfg = ModelConfig(
        type=ModelType.GPT,
        embedder=TokenEmbedderConfig(vocab_size=32000, d_model=256),
        topology=StackedTopologyConfig(
            layers=[
                AttentionLayerConfig(type=LayerType.ATTENTION, d_model=256, n_heads=4),
                SwiGLULayerConfig(type=LayerType.SWIGLU, d_model=256, d_ff=1024),
            ],
            repeat=4,
        ),
        target_params=100_000_000,
    )
    opt = cfg.optimize()
    assert opt is not cfg
    assert isinstance(opt.embedder, TokenEmbedderConfig)
    assert opt.embedder.d_model % 64 == 0
    assert opt.topology.repeat >= 4
