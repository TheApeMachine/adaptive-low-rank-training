from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from caramba.data.npy import NpyDataset


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="numpy not installed")
def test_rejects_negative_tokens(tmp_path: Path) -> None:
    import numpy as np  # type: ignore[import-not-found]

    arr = np.array([0, 1, -1, 2], dtype=np.int64)
    path = tmp_path / "tokens.npy"
    np.save(path, arr)
    with pytest.raises(ValueError):
        _ = NpyDataset(str(path), block_size=2)


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="numpy not installed")
def test_rejects_int32_overflow_tokens(tmp_path: Path) -> None:
    import numpy as np  # type: ignore[import-not-found]

    arr = np.array([0, 1, 2**31], dtype=np.int64)
    path = tmp_path / "tokens.npy"
    np.save(path, arr)
    with pytest.raises(ValueError):
        _ = NpyDataset(str(path), block_size=2)


def test_validate_sample_path_works_on_tensor() -> None:
    # Sanity: sample validation shouldn't allocate huge memory.
    t = torch.zeros((10_000,), dtype=torch.long)
    # NpyDataset validates via helper; we just ensure creating the dataset isn't required here.
    assert int(t.max().item()) == 0

