"""Token-ID spec loader for calibration / verification."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

from production.selfopt_cache import as_object_list



def _require_numpy() -> object:
    if importlib.util.find_spec("numpy") is None:
        raise ImportError("numpy is required to load .npy token specs")
    return importlib.import_module("numpy")


def _np_attr(np_mod: object, name: str) -> object:
    return getattr(np_mod, name, None)


def _np_call(np_mod: object, name: str, *args: object, **kwargs: object) -> object:
    fn = _np_attr(np_mod, name)
    if not callable(fn):
        raise AttributeError(f"numpy.{name} is not callable")
    return fn(*args, **kwargs)


def load_token_ids_spec(spec: str) -> list[int]:
    """Load token IDs from either:
    - a path to a file containing whitespace-separated ints
    - a path to a .npy file (np.load)
    - an inline whitespace-separated string of ints
    """
    s = str(spec)
    p = Path(s)
    if os.path.exists(s):
        if p.suffix == ".npy":
            np_mod = _require_numpy()
            # We convert the loaded array to a Python list[int] below, so using
            # mmap_mode="r" would be redundant (materialization happens anyway).
            arr = _np_call(np_mod, "load", str(p))
            asarray = _np_attr(np_mod, "asarray")
            if not callable(asarray):
                raise AttributeError("numpy.asarray is required")
            arr2 = asarray(arr)
            reshape = getattr(arr2, "reshape", None)
            if callable(reshape):
                arr2 = reshape(-1)
            int64_t = _np_attr(np_mod, "int64")
            dtype = getattr(arr2, "dtype", None)
            if dtype is not None and int64_t is not None and dtype != int64_t:
                astype = getattr(arr2, "astype", None)
                if callable(astype):
                    arr2 = astype(int64_t, copy=False)
            tolist = getattr(arr2, "tolist", None)
            if callable(tolist):
                lst = tolist()
                lst2 = as_object_list(lst)
                if lst2 is not None:
                    def _as_int(o: object) -> int:
                        if isinstance(o, bool):
                            return int(o)
                        if isinstance(o, int):
                            return int(o)
                        if isinstance(o, float):
                            if not o.is_integer():
                                raise TypeError(
                                    f"token id float must be a whole number, got {o!r}"
                                )
                            return int(o)
                        if isinstance(o, str):
                            return int(o.strip())
                        raise TypeError(
                            f"token id must be int-like, got {o!r} (type {type(o).__name__})"
                        )

                    return [_as_int(x) for x in lst2]
            raise TypeError("Unable to convert .npy token spec to list[int]")
        raw = p.read_text(encoding="utf-8", errors="ignore")
        return [int(t) for t in raw.strip().split() if t.strip()]
    return [int(t) for t in s.strip().split() if t.strip()]
