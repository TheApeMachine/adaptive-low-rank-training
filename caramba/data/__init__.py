"""Dataset utilities for training data loading.

Training requires feeding the model sequences of tokens. This package
provides Dataset implementations that load preprocessed token data
and serve it in the (input, target) pairs needed for language modeling.
"""
from __future__ import annotations

from caramba.data.auto import build_token_dataset
from caramba.data.npy import NpyDataset
from caramba.data.text_tokens import TextTokensDataset

__all__ = ["NpyDataset", "TextTokensDataset", "build_token_dataset"]
