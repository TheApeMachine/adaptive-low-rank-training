"""Model configuration: the complete specification for a model.

A model config ties together:
- Embedder: how to convert tokens to vectors
- Topology: the layer structure
- Optional diffusion head for hybrid generation
"""
from __future__ import annotations

import enum

from pydantic import Field

from caramba.config import Config
from caramba.config.diffusion import DiffusionHeadConfig
from caramba.config.embedder import EmbedderConfig, NoEmbedderConfig
from caramba.config.embedder import TokenEmbedderConfig
from caramba.config.layer import (
    AttentionLayerConfig,
    LayerNormLayerConfig,
    LinearLayerConfig,
    RMSNormLayerConfig,
    SwiGLULayerConfig,
)
from caramba.config.topology import TopologyConfig
from caramba.config.topology import (
    BranchingTopologyConfig,
    CyclicTopologyConfig,
    NestedTopologyConfig,
    ParallelTopologyConfig,
    RecurrentTopologyConfig,
    ResidualTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
)
from caramba.config.topology import NodeConfig

import math


class ModelType(str, enum.Enum):
    """Type of model architecture."""

    TRANSFORMER = "TransformerModel"
    GPT = "GPTModel"
    VIT = "ViTModel"
    MLP = "MLPModel"

    @classmethod
    def from_str(cls, s: str) -> "ModelType":
        """Convert a string to a ModelType."""
        return cls(s)

    @staticmethod
    def module_name() -> str:
        """Return the Python module containing model implementations."""
        return "caramba.model"


class ModelConfig(Config):
    """Complete specification for a model architecture.

    Combines embedder, topology, and optional diffusion head into a
    single config that can build the full model.
    """

    type: ModelType
    embedder: EmbedderConfig = Field(default_factory=NoEmbedderConfig)
    topology: TopologyConfig
    diffusion_head: DiffusionHeadConfig = Field(default_factory=DiffusionHeadConfig)
    tied_embeddings: bool = True

    # Optional self-optimization target: approximate parameter budget.
    target_params: int | None = None
    block_size: int | None = None

    def optimize(self) -> "ModelConfig":
        """Derive a reasonable transformer size from target_params.

        This is intentionally conservative and only scales common transformer
        patterns (stacked/residual topologies with attention + MLP blocks).
        """

        if self.target_params is None:
            return self

        target = int(self.target_params)
        if target <= 0:
            return self

        # Determine vocab size + starting depth estimate.
        vocab_size = 0
        if isinstance(self.embedder, TokenEmbedderConfig):
            vocab_size = int(self.embedder.vocab_size)
        # Entropy-ish signals: vocab_size and context length affect optimal depth/width tradeoffs.
        block_size = int(self.block_size) if self.block_size is not None else 2048
        ctx_factor = max(0.5, min(2.0, math.sqrt(max(1.0, math.log2(float(block_size)) / 11.0))))
        n_layer_guess = max(
            4,
            min(
                80,
                int(round((math.sqrt(float(target)) / 1000.0) * ctx_factor)),
            ),
        )

        # Solve for d_model in: a*d^2 + b*d - target = 0, where:
        # a ≈ 16*n_layer (attention + MLP), b ≈ vocab_size (embeddings).
        a = float(16 * n_layer_guess)
        b = float(vocab_size)
        disc = b * b + 4.0 * a * float(target)
        d = int(max(64, round((-b + math.sqrt(disc)) / (2.0 * a))))

        # Round d_model to a multiple of 64 for clean head dimensions.
        d_model = int(max(64, (d // 64) * 64))
        n_heads = max(1, d_model // 64)
        # Auto-derive n_kv_heads: use more aggressive GQA when context/depth ratio is high.
        ctx_per_layer = float(block_size) / float(max(1, n_layer_guess))
        if ctx_per_layer >= 512:
            n_kv_heads = max(1, n_heads // 4)
        elif ctx_per_layer >= 256:
            n_kv_heads = max(1, n_heads // 2)
        else:
            n_kv_heads = n_heads
        d_ff = int(4 * d_model)

        # Apply to a deep copy of the config.
        cfg = self.model_copy(deep=True)
        cfg.target_params = int(target)

        if isinstance(cfg.embedder, TokenEmbedderConfig):
            cfg.embedder.d_model = int(d_model)

        def scale_node(node: NodeConfig) -> NodeConfig:
            if isinstance(node, AttentionLayerConfig):
                node.d_model = int(d_model)
                node.n_heads = int(n_heads)
                node.n_kv_heads = int(n_kv_heads)
                return node
            if isinstance(node, SwiGLULayerConfig):
                node.d_model = int(d_model)
                node.d_ff = int(d_ff)
                return node
            if isinstance(node, (LayerNormLayerConfig, RMSNormLayerConfig)):
                node.d_model = int(d_model)
                return node
            if isinstance(node, LinearLayerConfig):
                # Only adjust if it looks like a residual-stream projection.
                if int(node.d_in) == int(node.d_out):
                    node.d_in = int(d_model)
                    node.d_out = int(d_model)
                return node
            if isinstance(
                node,
                (
                    NestedTopologyConfig,
                    StackedTopologyConfig,
                    ResidualTopologyConfig,
                    SequentialTopologyConfig,
                    ParallelTopologyConfig,
                    BranchingTopologyConfig,
                    CyclicTopologyConfig,
                    RecurrentTopologyConfig,
                ),
            ):
                node.layers = [scale_node(x) for x in list(node.layers)]  # type: ignore[assignment]
                if isinstance(node, (StackedTopologyConfig, ResidualTopologyConfig, SequentialTopologyConfig)):
                    node.repeat = int(n_layer_guess)
                return node
            return node

        cfg.topology = scale_node(cfg.topology)  # type: ignore[assignment]
        return cfg
