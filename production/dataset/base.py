"""
base holds the base class for all datasets.
"""

from __future__ import annotations

# stdlib
from numbers import Integral

# Python 3.12+ has `typing.override`; older runtimes should use typing_extensions.
try:  # pragma: no cover
    from typing import override  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing_extensions import override

import torch


_TokenBlock = torch.Tensor  # expected shape: (block_size,) or (block_size+1,)


class Dataset(torch.utils.data.Dataset[_TokenBlock]):
    """
    Dataset is a base class that provides a base implementation for a dataset.
    """

    data: torch.Tensor
    block_size: int

    def __init__(self, data: torch.Tensor, block_size: int):
        if isinstance(block_size, bool) or not isinstance(block_size, Integral):
            raise ValueError(
                f"Invalid block_size={block_size!r} (type={type(block_size).__name__}); block_size must be a positive integer and <= len(data) ({len(data)})."
            )

        bs = int(block_size)
        data_len = len(data)
        if bs <= 0 or bs > data_len:
            raise ValueError(
                f"Invalid block_size={bs}; block_size must be in [1, len(data)] but len(data)={data_len}."
            )

        self.data = data
        self.block_size = bs

    def __len__(self) -> int:
        return len(self.data) - self.block_size + 1

    @override
    def __getitem__(self, idx: int) -> _TokenBlock:
        return self.data[idx : idx + self.block_size]
