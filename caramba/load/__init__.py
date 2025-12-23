"""
load provides input/output utilities such as checkpoint loaders.
"""
from __future__ import annotations

from caramba.load.llama_loader import (
    init_decoupled_from_llama_attention,
    init_decoupled_from_qkvo,
    load_state_dict_mapped,
    load_torch_state_dict,
)

__all__ = [
    "init_decoupled_from_llama_attention",
    "init_decoupled_from_qkvo",
    "load_state_dict_mapped",
    "load_torch_state_dict",
]


