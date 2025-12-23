"""
compiler provides lowering passes that turn configs into canonical forms.
"""
from __future__ import annotations

from caramba.compiler.lower import lower_manifest

__all__ = [
    "lower_manifest",
]
