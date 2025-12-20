from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None


@dataclass
class TokenView:
    """A lightweight slice/view over a 1D token container (torch.Tensor or numpy array/memmap)."""

    data: Any
    start: int
    end: int

    def __len__(self) -> int:
        return int(self.end - self.start)


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


def load_tokens_any(*, path: Path, fmt: str, data_dtype: str) -> Any:
    """Load tokens as either numpy array/memmap or torch tensor, prioritizing binary formats."""
    if fmt in ("npy", "bin", "text") and _np is None:
        raise ImportError("numpy is required for training data loading (npy/bin/text).")

    if fmt == "npy":
        arr = _np.load(str(path), mmap_mode="r")  # type: ignore[union-attr]
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    if fmt == "bin":
        dt = _np.dtype(str(data_dtype))  # type: ignore[union-attr]
        arr = _np.memmap(str(path), dtype=dt, mode="r")  # type: ignore[union-attr]
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    if fmt == "pt":
        t = torch.load(str(path), map_location="cpu")
        if isinstance(t, dict) and "tokens" in t:
            t = t["tokens"]
        if not isinstance(t, torch.Tensor):
            raise ValueError("pt data must be a 1D torch.Tensor or dict with 'tokens'")
        return t.view(-1).to(torch.long)

    if fmt == "text":
        # Legacy: whitespace-separated integer IDs.
        # NOTE: This reads the file into RAM; for real scale prefer .npy/.bin.
        raw = path.read_text(encoding="utf-8")
        arr = _np.fromstring(raw.strip(), dtype=_np.int64, sep=" ")  # type: ignore[union-attr]
        return arr

    raise ValueError(f"Unknown data format: {fmt}")


def split_train_val(tokens_any: Any, *, val_frac: float) -> Tuple[TokenView, TokenView]:
    n_total = int(tokens_any.numel()) if isinstance(tokens_any, torch.Tensor) else int(len(tokens_any))
    n_train = int((1.0 - float(val_frac)) * n_total)
    n_train = max(min(n_train, n_total - 2), 2)
    return TokenView(tokens_any, 0, n_train), TokenView(tokens_any, n_train, n_total)


def determine_vocab_size(
    *,
    tokens_any: Any,
    vocab_size: Optional[int],
    tokenizer: str,
) -> int:
    if vocab_size is not None:
        return int(vocab_size)
    if tokenizer == "tiktoken":
        return 50257
    if isinstance(tokens_any, torch.Tensor):
        return int(tokens_any.max().item()) + 1
    if _np is None:
        raise ImportError("numpy required to scan vocab size for numpy/memmap datasets.")
    print("[warn] --vocab-size not provided; scanning dataset for max token id (can be very slow on big memmaps).")
    return int(_np.max(tokens_any)) + 1  # type: ignore[arg-type, union-attr]


def get_batch_any(
    view: TokenView,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    _offs_cache_t: Optional[Dict[int, torch.Tensor]] = None,
    _offs_cache_np: Optional[Dict[int, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        x = view.data[idx].to(torch.int32)
        y = view.data[idx + 1]
        if device.type == "cuda":
            x = x.pin_memory()
            y = y.pin_memory()
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        return x.to(device), y.to(device)

    if _np is None:
        raise ImportError("numpy required for numpy/memmap batch sampling.")

    offs_np = _offs_cache_np.get(int(block_size))
    if offs_np is None or int(getattr(offs_np, "shape", [0])[0]) != int(block_size):
        offs_np = _np.arange(int(block_size), dtype=_np.int64)  # type: ignore[union-attr]
        _offs_cache_np[int(block_size)] = offs_np

    ix_np = ix.numpy().astype(_np.int64, copy=False)  # type: ignore[union-attr]
    idx_np = (int(view.start) + ix_np[:, None] + offs_np[None, :]).astype(_np.int64, copy=False)

    # Keep inputs compact: embedding accepts int32 indices, targets remain int64 for cross_entropy.
    x_np = _np.asarray(view.data[idx_np], dtype=_np.int32)  # type: ignore[union-attr]
    y_np = _np.asarray(view.data[idx_np + 1], dtype=_np.int64)  # type: ignore[union-attr]
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    if device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x.to(device), y.to(device)

