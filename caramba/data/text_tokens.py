"""Text token dataset for whitespace-separated integer token IDs.

This is a compatibility loader for legacy `.tokens` files that store token
IDs as ASCII integers separated by whitespace. For large-scale training,
prefer binary formats like `.npy` (mmap) to reduce parsing overhead.
"""

from __future__ import annotations

import importlib
import importlib.util

import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import override


def _require_numpy() -> object:
    if importlib.util.find_spec("numpy") is None:
        raise ImportError("numpy is required for TextTokensDataset")
    return importlib.import_module("numpy")


def _np_attr(np_mod: object, name: str) -> object:
    return getattr(np_mod, name, None)


def _np_call(np_mod: object, name: str, *args: object, **kwargs: object) -> object:
    fn = _np_attr(np_mod, name)
    if not callable(fn):
        raise AttributeError(f"numpy.{name} is not callable")
    return fn(*args, **kwargs)


class TextTokensDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset that loads whitespace-separated token IDs from a text file.

    Why this exists:
    - Some existing corpora are stored as `.tokens` text files.
    - Caramba should be able to train from those without a conversion step.
    """

    def __init__(self, path: str, *, block_size: int, dtype: str = "int32") -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        np_mod = _require_numpy()
        int_dt = _np_call(np_mod, "dtype", str(dtype))
        # numpy.fromfile with sep parses text without an intermediate Python string.
        arr_obj = _np_call(np_mod, "fromfile", str(path), dtype=int_dt, sep=" ")
        reshape = getattr(arr_obj, "reshape", None)
        if callable(reshape):
            arr_obj = reshape(-1)
        t = torch.from_numpy(arr_obj).to(dtype=torch.long)
        if len(t) <= int(block_size):
            raise ValueError(
                f"block_size must be smaller than data length, got "
                f"block_size={block_size}, len={len(t)}"
            )
        self.tokens = t
        self.block_size = int(block_size)

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        block = self.tokens[idx : idx + self.block_size + 1]
        x = block[:-1]
        y = block[1:]
        return x, y

