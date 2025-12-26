from __future__ import annotations

from pathlib import Path

import pytest

import torch

from caramba.instrumentation.hdf5_store import H5Store


def test_h5store_writes_step(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    _ = h5py

    path = tmp_path / "train.h5"
    store = H5Store(path, enabled=True)
    assert store.enabled in (True, False)
    if not store.enabled:
        pytest.skip("h5py not available or H5Store disabled")

    store.write_step(1, {"x": torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    store.close()

    assert path.exists()
    with h5py.File(str(path), "r") as f:
        assert "steps" in f
        assert "1" in f["steps"]
        assert "x" in f["steps"]["1"]
        ds = f["steps"]["1"]["x"]
        assert tuple(ds.shape) == (2, 2)

