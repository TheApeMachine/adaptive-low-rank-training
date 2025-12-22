"""
base holds the base class for all datasets.
"""

from __future__ import annotations

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
        self.data = data
        self.block_size = int(block_size)

    def __len__(self) -> int:
        return len(self.data) - self.block_size + 1

    @override
    def __getitem__(self, idx: int) -> _TokenBlock:
        return self.data[idx : idx + self.block_size]
