"""
embedder provides the embedder module.
"""
from __future__ import annotations

from torch import Tensor, nn
from caramba.config.embedder import EmbedderConfig, EmbedderType


class Embedder(nn.Module):
    """
    Embedder provides a pluggable embedding stage.
    """
    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__()
        self.config: EmbedderConfig = config

        if config.type == EmbedderType.NONE:
            raise ValueError("Embedder type cannot be NONE.")

        self.token_embedding: nn.Embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the embedder.
        """
        match self.config.type:
            case EmbedderType.TOKEN:
                return self.token_embedding(
                    x.to(dtype=self.token_embedding.weight.dtype).long(),
                )
            case _:
                raise ValueError(f"Unknown embedder type: {self.config.type}")
