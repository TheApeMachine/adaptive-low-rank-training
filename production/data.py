from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import numbers
import threading
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import torch

from production.selfopt_cache import as_str_object_dict


@dataclass
class TokenView:
    """A lightweight slice/view over a 1D token container (torch.Tensor or numpy array/memmap)."""

    data: object
    start: int
    end: int

    def __len__(self) -> int:
        return int(self.end - self.start)


# Embedding indices are ultimately handled with 32-bit integer indexing on some backends.
# We enforce a conservative *exclusive* upper bound to avoid edge-case overflows in downstream ops.
_INT32_MAX_EXCLUSIVE = 2_147_483_647  # 2**31 - 1 (exclusive)
_INT32_MIN_INCLUSIVE = 0

# Track whether a loaded token array/tensor is safe to cast to int32 for embedding indices.
# Keyed by id(tokens_any) because TokenView holds a reference to the loaded object.
_TOKENS_INT32_SAFE: dict[int, bool] = {}
_TOKENS_INT32_SAFE_LOCK = threading.Lock()


@runtime_checkable
class _Sized(Protocol):
    def __len__(self) -> int: ...


@runtime_checkable
class _Indexable(Protocol):
    def __getitem__(self, idx: object) -> object: ...


def _require_numpy() -> object:
    if importlib.util.find_spec("numpy") is None:
        raise ImportError("numpy is required for training data loading (npy/bin/text).")
    return importlib.import_module("numpy")


def _np_attr(np_mod: object, name: str) -> object:
    return getattr(np_mod, name, None)


def _np_call(np_mod: object, name: str, *args: object, **kwargs: object) -> object:
    fn = _np_attr(np_mod, name)
    if not callable(fn):
        raise AttributeError(f"numpy.{name} is not callable")
    return fn(*args, **kwargs)


def _call1(fn: object, arg: object, *, what: str) -> object:
    if not callable(fn):
        raise AttributeError(f"{what} is required but not callable")
    return fn(arg)


def _as_int(o: object) -> int:
    if isinstance(o, bool):
        return int(o)
    # numpy integer scalars and other integer-like objects
    if isinstance(o, numbers.Integral):
        return int(o)
    if isinstance(o, float):
        return int(o)
    if isinstance(o, str):
        return int(o.strip())
    raise TypeError(f"Expected an int-like value, got {type(o).__name__}: {o!r}")


def _token_id_min_max(tokens_any: object) -> tuple[int, int]:
    """Return (min_id, max_id) for torch.Tensor or numpy array/memmap."""
    if isinstance(tokens_any, torch.Tensor):
        if tokens_any.numel() == 0:
            raise ValueError("Token dataset is empty; expected at least 1 token id.")
        # Ensure integer comparisons behave as expected.
        t = tokens_any.view(-1).to(dtype=torch.long)
        return int(t.min().item()), int(t.max().item())

    if not isinstance(tokens_any, _Sized):
        raise TypeError("Token dataset must support len() for non-torch backends.")
    if len(tokens_any) == 0:
        raise ValueError("Token dataset is empty; expected at least 1 token id.")
    # numpy min/max scan once over the dataset (ndarray or memmap).
    np_mod = _require_numpy()
    mn_obj = _np_call(np_mod, "min", tokens_any)
    mx_obj = _np_call(np_mod, "max", tokens_any)
    try:
        return _as_int(mn_obj), _as_int(mx_obj)
    except (TypeError, ValueError) as e:
        msg = (
            "Failed to convert numpy min/max results to int token ids. "
            f"mn_obj={mn_obj!r} (type={type(mn_obj).__name__}), "
            f"mx_obj={mx_obj!r} (type={type(mx_obj).__name__}), "
            f"tokens_any={tokens_any!r} (type={type(tokens_any).__name__})."
        )
        raise ValueError(msg) from e


def _validate_token_ids_int32_range(*, tokens_any: object, source: Path) -> None:
    """Fail fast if any token id would overflow/underflow an int32 embedding index."""
    mn, mx = _token_id_min_max(tokens_any)
    # Enforce ids in [0, 2**31-1) (exclusive upper bound).
    if mn < _INT32_MIN_INCLUSIVE or mx >= _INT32_MAX_EXCLUSIVE:
        msg = (
            "Invalid token ids for int32 embedding indices. "
            f"Expected ids in [{_INT32_MIN_INCLUSIVE}, {_INT32_MAX_EXCLUSIVE - 1}] "
            f"but found min={mn}, max={mx} in dataset {str(source)!r}. "
            "Fix the dataset/tokenizer or regenerate tokens before training."
        )
        raise ValueError(msg)


def _record_int32_safety(*, tokens_any: object, source: Path) -> None:
    """
    Record whether token ids fit in int32 embedding indices.

    Note: callers should validate tokens separately if they want a hard failure on invalid IDs.
    """
    mn, mx = _token_id_min_max(tokens_any)
    safe = (mn >= _INT32_MIN_INCLUSIVE) and (mx < _INT32_MAX_EXCLUSIVE)
    with _TOKENS_INT32_SAFE_LOCK:
        _TOKENS_INT32_SAFE[id(tokens_any)] = bool(safe)
    if not safe:
        msg = (
            "[warn] token ids exceed int32 range for embedding indices; "
            f"min={mn}, max={mx} in dataset {str(source)!r}. "
            "Will keep inputs as int64 to avoid overflow."
        )
        print(msg)


def infer_data_format(path: Path, data_format: str) -> str:
    fmt = str(data_format)
    if fmt != "auto":
        return fmt
    suf = path.suffix.lower()
    if suf == ".npy":
        return "npy"
    if suf == ".bin":
        return "bin"
    if suf == ".pt":
        return "pt"
    return "text"


def _torch_load_obj(path: Path) -> object:
    return cast(object, torch.load(str(path), map_location="cpu"))


def load_tokens_any(*, path: Path, fmt: str, data_dtype: str) -> object:
    """Load tokens as either numpy array/memmap or torch tensor, prioritizing binary formats."""
    if fmt == "npy":
        np_mod = _require_numpy()
        arr = _np_call(np_mod, "load", str(path), mmap_mode="r")
        ndim = getattr(arr, "ndim", None)
        if isinstance(ndim, int) and ndim != 1:
            reshape = getattr(arr, "reshape", None)
            if callable(reshape):
                arr = reshape(-1)
        _validate_token_ids_int32_range(tokens_any=arr, source=path)
        _record_int32_safety(tokens_any=arr, source=path)
        return arr

    if fmt == "bin":
        np_mod = _require_numpy()
        dtype_fn = _np_attr(np_mod, "dtype")
        memmap_fn = _np_attr(np_mod, "memmap")
        if not callable(dtype_fn) or not callable(memmap_fn):
            raise AttributeError("numpy.dtype/numpy.memmap are required for .bin loading")
        dt = dtype_fn(str(data_dtype))
        arr = memmap_fn(str(path), dtype=dt, mode="r")
        ndim = getattr(arr, "ndim", None)
        if isinstance(ndim, int) and ndim != 1:
            reshape = getattr(arr, "reshape", None)
            if callable(reshape):
                arr = reshape(-1)
        _validate_token_ids_int32_range(tokens_any=arr, source=path)
        _record_int32_safety(tokens_any=arr, source=path)
        return arr

    if fmt == "pt":
        t_obj = _torch_load_obj(path)
        if isinstance(t_obj, torch.Tensor):
            t = t_obj
        else:
            d = as_str_object_dict(t_obj)
            tok_obj = d["tokens"] if d is not None and "tokens" in d else None
            if not isinstance(tok_obj, torch.Tensor):
                raise ValueError("pt data must be a 1D torch.Tensor or dict with 'tokens'")
            t = tok_obj
        t = t.view(-1).to(torch.long)
        _validate_token_ids_int32_range(tokens_any=t, source=path)
        _record_int32_safety(tokens_any=t, source=path)
        return t

    if fmt == "text":
        # Legacy: whitespace-separated integer IDs.
        # NOTE: This reads the file into RAM; for real scale prefer .npy/.bin.
        raw = path.read_text(encoding="utf-8")
        np_mod = _require_numpy()
        fromstring_fn = _np_attr(np_mod, "fromstring")
        int64_t = _np_attr(np_mod, "int64")
        if not callable(fromstring_fn) or int64_t is None:
            raise AttributeError("numpy.fromstring/numpy.int64 are required for text loading")
        arr = fromstring_fn(raw.strip(), dtype=int64_t, sep=" ")
        _validate_token_ids_int32_range(tokens_any=arr, source=path)
        _record_int32_safety(tokens_any=arr, source=path)
        return arr

    raise ValueError(f"Unknown data format: {fmt}")


def split_train_val(tokens_any: object, *, val_frac: float) -> tuple[TokenView, TokenView]:
    if isinstance(tokens_any, torch.Tensor):
        n_total = int(tokens_any.numel())
    elif isinstance(tokens_any, _Sized):
        n_total = int(len(tokens_any))
    else:
        raise TypeError("Token dataset must be torch.Tensor or sized numpy-like container.")
    n_train = int((1.0 - float(val_frac)) * n_total)
    n_train = max(min(n_train, n_total - 2), 2)
    return TokenView(tokens_any, 0, n_train), TokenView(tokens_any, n_train, n_total)


def determine_vocab_size(
    *,
    tokens_any: object,
    vocab_size: int | None,
    tokenizer: str,
) -> int:
    if vocab_size is not None:
        return int(vocab_size)
    if tokenizer == "tiktoken":
        return 50257
    print("[warn] --vocab-size not provided; scanning dataset for max token id (can be very slow on big memmaps).")
    _, mx = _token_id_min_max(tokens_any)
    return int(mx) + 1


def get_batch_any(
    view: TokenView,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    _offs_cache_t: dict[int, torch.Tensor] | None = None,
    _offs_cache_np: dict[int, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized batch sampler for torch tensors and numpy arrays/memmaps."""
    max_start = len(view) - int(block_size) - 1
    if max_start <= 0:
        raise ValueError(f"Not enough tokens in split: len={len(view)} block={block_size}")

    if _offs_cache_t is None:
        _offs_cache_t = {}
    if _offs_cache_np is None:
        _offs_cache_np = {}

    offs_t = _offs_cache_t.get(int(block_size))
    if offs_t is None or offs_t.numel() != int(block_size):
        offs_t = torch.arange(int(block_size), dtype=torch.long)
        _offs_cache_t[int(block_size)] = offs_t

    ix = torch.randint(0, max_start, (int(batch_size),), device="cpu", dtype=torch.long)

    if isinstance(view.data, torch.Tensor):
        base = (int(view.start) + ix).unsqueeze(1)
        idx = base + offs_t.unsqueeze(0)
        x_raw = view.data[idx]
        # Keep inputs compact for memory efficiency: embedding accepts int32/int64 indices;
        # using int32 saves ~50% memory vs int64. If any token id exceeds 2**31-1 (or is
        # negative), avoid the int32 cast to prevent overflow.
        with _TOKENS_INT32_SAFE_LOCK:
            int32_safe = _TOKENS_INT32_SAFE.get(id(view.data))
        if int32_safe is False:
            x = x_raw.to(torch.long)
        elif int32_safe is True:
            x = x_raw.to(torch.int32)
        else:
            mn = int(x_raw.min().item())
            mx = int(x_raw.max().item())
            if mn < _INT32_MIN_INCLUSIVE or mx >= _INT32_MAX_EXCLUSIVE:
                msg = (
                    "[warn] batch contains token ids outside int32 range; "
                    f"min={mn}, max={mx}. Keeping inputs as int64 to avoid overflow."
                )
                print(msg)
                x = x_raw.to(torch.long)
            else:
                x = x_raw.to(torch.int32)
        y = view.data[idx + 1].to(torch.long)
        if device.type == "cuda":
            x = x.pin_memory()
            y = y.pin_memory()
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        return x.to(device), y.to(device)

    np_mod = _require_numpy()
    if not isinstance(view.data, _Indexable):
        raise TypeError("Non-torch token container must support __getitem__.")

    offs_np = _offs_cache_np.get(int(block_size))
    shape0: int | None = None
    if offs_np is not None:
        shape_obj = getattr(offs_np, "shape", None)
        if isinstance(shape_obj, tuple) and shape_obj and isinstance(shape_obj[0], numbers.Integral):
            shape0 = int(shape_obj[0])
    if offs_np is None or shape0 != int(block_size):
        int64_t = _np_attr(np_mod, "int64")
        offs_np = _np_call(np_mod, "arange", int(block_size), dtype=int64_t)
        _offs_cache_np[int(block_size)] = offs_np

    ix_np0 = ix.numpy()
    astype_fn = getattr(ix_np0, "astype", None)
    int64_t = _np_attr(np_mod, "int64")
    if not callable(astype_fn):
        raise TypeError("Expected torch.Tensor.numpy() result to support .astype(...)")
    ix_np = astype_fn(int64_t, copy=False)
    reshape_ix = getattr(ix_np, "reshape", None)
    reshape_offs = getattr(offs_np, "reshape", None)
    if not callable(reshape_ix) or not callable(reshape_offs):
        raise TypeError("Expected numpy arrays to support .reshape(...)")
    ix_col = reshape_ix(-1, 1)
    offs_row = reshape_offs(1, -1)
    tmp = _np_call(np_mod, "add", ix_col, offs_row)
    idx_np = _np_call(np_mod, "add", tmp, int(view.start))
    astype_idx = getattr(idx_np, "astype", None)
    if callable(astype_idx):
        idx_np = astype_idx(int64_t, copy=False)

    # Keep inputs compact for memory efficiency: embedding accepts int32/int64 indices; using
    # int32 saves ~50% memory vs int64. If any token id exceeds 2**31-1 (or is negative),
    # avoid the int32 cast to prevent overflow.
    with _TOKENS_INT32_SAFE_LOCK:
        int32_safe = _TOKENS_INT32_SAFE.get(id(view.data))
    if int32_safe is False:
        x_np = _np_call(np_mod, "asarray", view.data[idx_np], dtype=int64_t)
    elif int32_safe is True:
        int32_t = _np_attr(np_mod, "int32")
        x_np = _np_call(np_mod, "asarray", view.data[idx_np], dtype=int32_t)
    else:
        x64 = _np_call(np_mod, "asarray", view.data[idx_np], dtype=int64_t)
        mn = _as_int(_np_call(np_mod, "min", x64))
        mx = _as_int(_np_call(np_mod, "max", x64))
        if mn < _INT32_MIN_INCLUSIVE or mx >= _INT32_MAX_EXCLUSIVE:
            msg = (
                "[warn] batch contains token ids outside int32 range; "
                f"min={mn}, max={mx}. Keeping inputs as int64 to avoid overflow."
            )
            print(msg)
            x_np = x64
        else:
            int32_t = _np_attr(np_mod, "int32")
            astype2 = getattr(x64, "astype", None)
            if callable(astype2):
                x_np = astype2(int32_t, copy=False)
            else:
                x_np = x64

    idx_np_p1 = _np_call(np_mod, "add", idx_np, 1)
    y_np = _np_call(np_mod, "asarray", view.data[idx_np_p1], dtype=int64_t)

    def _torch_from_numpy(o: object) -> torch.Tensor:
        fn = getattr(torch, "from_numpy", None)
        t_obj = _call1(fn, o, what="torch.from_numpy")
        if not isinstance(t_obj, torch.Tensor):
            raise TypeError("torch.from_numpy did not return torch.Tensor")
        return t_obj

    x = _torch_from_numpy(x_np)
    y = _torch_from_numpy(y_np)
    if device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x.to(device), y.to(device)

