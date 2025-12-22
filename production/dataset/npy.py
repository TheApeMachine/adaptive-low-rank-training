"""
npy holds the dataset for npy files.
"""

from __future__ import annotations

import importlib
import importlib.util
import torch

from production.dataset.base import Dataset


def _require_numpy() -> object:
    if importlib.util.find_spec("numpy") is None:
        raise ImportError("numpy is required for NPYDataset")
    return importlib.import_module("numpy")


def _np_attr(np_mod: object, name: str) -> object:
    return getattr(np_mod, name, None)


def _np_call(np_mod: object, name: str, *args: object, **kwargs: object) -> object:
    fn = _np_attr(np_mod, name)
    if not callable(fn):
        raise AttributeError(f"numpy.{name} is not callable")
    return fn(*args, **kwargs)


def _torch_from_numpy(o: object) -> torch.Tensor:
    fn = getattr(torch, "from_numpy", None)
    if not callable(fn):
        raise AttributeError("torch.from_numpy is required for NPYDataset")
    t_obj = fn(o)
    if not isinstance(t_obj, torch.Tensor):
        raise TypeError("torch.from_numpy did not return torch.Tensor")
    return t_obj


class NPYDataset(Dataset):
    """
    NPYDataset is a dataset for npy files.
    """

    def __init__(self, path: str, block_size: int):
        np_mod = _require_numpy()
        arr_obj = _np_call(np_mod, "load", str(path), mmap_mode="r")
        ndarray_t = _np_attr(np_mod, "ndarray")
        if not isinstance(ndarray_t, type) or not isinstance(arr_obj, ndarray_t):
            raise TypeError("Expected numpy.load to return a numpy ndarray/memmap")
        reshape = getattr(arr_obj, "reshape", None)
        if not callable(reshape):
            raise TypeError("Expected numpy array to support .reshape(...)")
        arr = reshape(-1)
        t = _torch_from_numpy(arr).to(dtype=torch.long)
        super().__init__(t, int(block_size))
