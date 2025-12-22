"""Dataset loading for training runner.

Why this exists:
- Training wants views + a stable batching function; the underlying storage can vary.
- Keeping dataset I/O separate reduces noise in the training loop and helps with profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sized

import torch

from production.run_config import TrainConfig


@dataclass(frozen=True)
class DatasetState:
    """Why: bundle together the minimal dataset objects the runner needs."""

    train_view: object
    val_view: object
    vocab_size: int
    n_total_tokens: int
    fmt: str


def load_dataset(run_cfg: TrainConfig) -> DatasetState:
    """Why: resolve format, load tokens (mmap when possible), split, and infer vocab if needed."""
    import production.data as data_mod  # pylint: disable=import-outside-toplevel

    def _as_int(o: object, default: int) -> int:
        if isinstance(o, bool):
            return int(o)
        if isinstance(o, int):
            return int(o)
        if isinstance(o, float):
            return int(o)
        if isinstance(o, str):
            try:
                return int(o.strip())
            except ValueError:
                return int(default)
        return int(default)

    def _load_tokens_obj(*, path: Path, fmt: str, data_dtype: str) -> object:
        # `load_tokens_any` is typed loosely; isolate behind object boundary.
        return data_mod.load_tokens_any(path=path, fmt=fmt, data_dtype=data_dtype)

    data_path = Path(str(run_cfg.data))
    fmt = data_mod.infer_data_format(data_path, str(run_cfg.data_format))
    tokens = _load_tokens_obj(path=data_path, fmt=str(fmt), data_dtype=str(run_cfg.data_dtype))

    vocab = run_cfg.vocab_size
    if vocab is None:
        vocab = data_mod.determine_vocab_size(
            tokens_any=tokens,
            vocab_size=None,
            tokenizer=str(run_cfg.tokenizer),
        )
    vocab_i = _as_int(vocab, 0)

    if isinstance(tokens, torch.Tensor):
        n_total = int(tokens.numel())
    elif isinstance(tokens, Sized):
        n_total = int(len(tokens))
    else:
        raise TypeError(f"unsupported tokens container type: {type(tokens).__name__}")

    train_view, val_view = data_mod.split_train_val(tokens, val_frac=float(run_cfg.val_frac))

    return DatasetState(
        train_view=train_view,
        val_view=val_view,
        vocab_size=vocab_i,
        n_total_tokens=int(n_total),
        fmt=str(fmt),
    )


