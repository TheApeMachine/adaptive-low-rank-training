"""
weight provides weight containers and strategies.
"""

from __future__ import annotations

from caramba.weight.attention_decoupled import DecoupledAttentionWeight
from caramba.weight.attention_llama import LlamaAttentionWeight
from caramba.weight.dense import DenseWeight
from caramba.weight.layer_norm import LayerNormWeight
from caramba.weight.multihead import MultiheadWeight

__all__ = [
    "DecoupledAttentionWeight",
    "DenseWeight",
    "LayerNormWeight",
    "LlamaAttentionWeight",
    "MultiheadWeight",
]


