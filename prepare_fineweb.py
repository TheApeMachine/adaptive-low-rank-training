#!/usr/bin/env python3
"""
prepare_fineweb.py

Download and prepare FineWeb-Edu dataset for training.

This script downloads a subset of the FineWeb-Edu dataset from Hugging Face
and tokenizes it using tiktoken (GPT-2 encoding).

Usage:
    python3.12 prepare_fineweb.py --tokens 100M    # 100 million tokens
    python3.12 prepare_fineweb.py --tokens 1B     # 1 billion tokens
    python3.12 prepare_fineweb.py --tokens 100M --output fineweb_100m.tokens
    python3.12 prepare_fineweb.py --tokens 100M --output fineweb_100m.npy

Requirements:
    pip install datasets tiktoken tqdm numpy
"""

import argparse
import os
import gc
import importlib
import importlib.util
from collections.abc import Iterable
from typing import Protocol, runtime_checkable
import numpy as np

HAS_DATASETS: bool = importlib.util.find_spec("datasets") is not None
HAS_TIKTOKEN: bool = importlib.util.find_spec("tiktoken") is not None
HAS_TQDM: bool = importlib.util.find_spec("tqdm") is not None


def _as_str_object_dict(x: object) -> dict[str, object] | None:
    if not isinstance(x, dict):
        return None
    # JSON/datasets payloads are untyped; isolate unknown key/value types here.
    return {str(k): v for k, v in x.items()}  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]


def _as_iterable(x: object) -> Iterable[object] | None:
    return x if isinstance(x, Iterable) else None


@runtime_checkable
class _TiktokenEncoder(Protocol):
    n_vocab: int

    def encode_ordinary(self, text: str) -> list[int]: ...


@runtime_checkable
class _TextWriter(Protocol):
    def write(self, s: str, /) -> int: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def _load_dataset_stream(*, repo: str, name: str | None) -> object:
    if not HAS_DATASETS:
        raise ImportError("'datasets' not installed. Run: pip install datasets")
    try:
        datasets = importlib.import_module("datasets")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("'datasets' not installed. Run: pip install datasets") from exc
    ld = getattr(datasets, "load_dataset", None)
    if not callable(ld):
        raise ImportError("datasets.load_dataset is not available")
    if name is None:
        return ld(repo, split="train", streaming=True)
    return ld(repo, name=name, split="train", streaming=True)


def _get_tiktoken_encoder() -> _TiktokenEncoder:
    if not HAS_TIKTOKEN:
        raise ImportError("'tiktoken' not installed. Run: pip install tiktoken")
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("'tiktoken' not installed. Run: pip install tiktoken") from exc
    get_enc = getattr(tiktoken, "get_encoding", None)
    if not callable(get_enc):
        raise ImportError("tiktoken.get_encoding is not available")
    enc_obj: object = get_enc("gpt2")
    if not isinstance(enc_obj, _TiktokenEncoder):
        raise TypeError("tiktoken encoder does not match expected interface")
    return enc_obj


def _pbar_new(*, total: int, desc: str) -> object | None:
    if not HAS_TQDM:
        return None
    try:
        tqdm_mod = importlib.import_module("tqdm")
    except ImportError:
        return None
    tqdm_ctor = getattr(tqdm_mod, "tqdm", None)
    if not callable(tqdm_ctor):
        return None
    return tqdm_ctor(total=int(total), unit="tok", desc=str(desc))


def _pbar_update(pbar: object | None, n: int) -> None:
    if pbar is None:
        return
    upd = getattr(pbar, "update", None)
    if callable(upd):
        _ = upd(int(n))


def _pbar_close(pbar: object | None) -> None:
    if pbar is None:
        return
    close_fn = getattr(pbar, "close", None)
    if callable(close_fn):
        _ = close_fn()


def _np_load_obj(path: str, *, mmap_mode: str | None = None) -> object:
    load_fn = getattr(np, "load", None)
    if not callable(load_fn):
        raise ImportError("numpy.load not available")
    if mmap_mode is None:
        return load_fn(path)
    return load_fn(path, mmap_mode=mmap_mode)


def parse_token_count(s: str) -> int:
    """Parse token count string like '100M' or '1B'."""
    s = s.upper().strip()
    if s.endswith('B'):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(s)


def download_and_tokenize(
    target_tokens: int,
    output_path: str,
    sample_name: str = "sample-10BT",  # FineWeb-Edu 10BT sample
    output_format: str = "auto",
):
    """
    Download FineWeb-Edu and tokenize to target token count.

    Args:
        target_tokens: Number of tokens to collect
        output_path: Output file path
        sample_name: Which FineWeb-Edu sample to use
        output_format: 'text', 'npy', or 'auto'
    """
    if not HAS_DATASETS or not HAS_TIKTOKEN:
        print("Missing dependencies. Please install: pip install datasets tiktoken")
        return False

    # Determine format if auto
    if output_format == "auto":
        output_format = "npy" if output_path.endswith(".npy") else "text"

    print(f"Target: {target_tokens:,} tokens")
    print(f"Output: {output_path} (format={output_format})")
    print(f"Dataset: HuggingFaceFW/fineweb-edu ({sample_name})")
    print("-" * 60)

    # Initialize tokenizer
    try:
        enc = _get_tiktoken_encoder()
    except (ImportError, TypeError, ValueError) as exc:
        print(f"Missing or incompatible tokenizer: {exc}")
        return False

    # Stream dataset
    print("Loading dataset (streaming mode)...")
    try:
        dataset = _load_dataset_stream(repo="HuggingFaceFW/fineweb-edu", name=sample_name)
    except (ImportError, RuntimeError, OSError, ValueError, TypeError) as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative: fineweb-edu-score-2 subset...")
        try:
            dataset = _load_dataset_stream(repo="HuggingFaceFW/fineweb-edu-score-2", name=None)
        except (ImportError, RuntimeError, OSError, ValueError, TypeError) as e2:
            print(f"Error loading alternative: {e2}")
            return False

    # Collect tokens
    total_tokens = 0
    doc_count = 0
    mm_writer: np.memmap | None = None
    txt_writer: _TextWriter | None = None
    write_pos = 0

    print("Tokenizing documents...")
    pbar = _pbar_new(total=target_tokens, desc="Tokens")

    try:
        # Streaming writers:
        # - npy: write directly into a memmap-backed .npy (no giant Python list)
        # - text: append chunks of space-separated ints
        if output_format == "npy":
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            # `numpy.lib.format.open_memmap` typing is incomplete; treat it as an object boundary.
            open_memmap = getattr(np.lib.format, "open_memmap", None)
            if not callable(open_memmap):
                raise ImportError("numpy.lib.format.open_memmap not available")
            mm_obj: object = open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(int(target_tokens),))
            if not isinstance(mm_obj, np.memmap):
                raise TypeError("open_memmap did not return numpy.memmap")
            mm_writer = mm_obj
            write_pos = 0
        else:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            txt_writer = open(output_path, "w", encoding="utf-8")

        it = _as_iterable(dataset)
        if it is None:
            raise TypeError("streaming dataset is not iterable")
        for doc_obj in it:
            # Get text content
            doc = _as_str_object_dict(doc_obj)
            if doc is None:
                continue
            text_obj = doc.get("text", "")
            text = str(text_obj) if isinstance(text_obj, str) else ""
            if not text:
                continue

            # Tokenize
            tokens = enc.encode_ordinary(text)
            if not tokens:
                continue

            new_tokens = len(tokens)
            remaining = int(target_tokens) - int(total_tokens)
            take = int(new_tokens if new_tokens <= remaining else remaining)
            if take <= 0:
                break

            # Write
            if output_format == "npy":
                if mm_writer is None:
                    raise RuntimeError("memmap writer not initialized")
                # Clip into uint16 range (GPT-2 vocab fits; this is defensive).
                arr = np.asarray(tokens[:take], dtype=np.int64)
                if arr.size > 0:
                    arr = np.clip(arr, 0, 65535).astype(np.uint16, copy=False)
                    mm_writer[write_pos : write_pos + int(arr.size)] = arr
                    write_pos += int(arr.size)
            else:
                if txt_writer is None:
                    raise RuntimeError("text writer not initialized")
                # Write as space-separated integers (streaming)
                # Avoid huge join by chunking.
                chunk_size = 100_000
                for i in range(0, take, chunk_size):
                    chunk = tokens[i : i + chunk_size]
                    _ = txt_writer.write(" ".join(map(str, chunk)))
                    if i + chunk_size < take:
                        _ = txt_writer.write(" ")
                _ = txt_writer.write(" ")

            total_tokens += take
            doc_count += 1

            _pbar_update(pbar, take)

            # Check if we have enough
            if total_tokens >= target_tokens:
                break

            # Progress update every 1000 docs
            if doc_count % 1000 == 0 and not HAS_TQDM:
                print(f"  {doc_count:,} docs | {total_tokens:,} tokens ({100*total_tokens/target_tokens:.1f}%)")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving collected tokens...")

    # Explicit cleanup to avoid Bad file descriptor errors from background threads
    del dataset
    _ = gc.collect()

    _pbar_close(pbar)

    # Finalize writer
    if output_format == "npy":
        # If we didn't reach target_tokens (e.g., interruption), rewrite to exact length.
        # Note: shrinking a .npy in-place isn't supported; we create a truncated copy.
        if total_tokens < target_tokens:
            print(f"[warn] Collected {total_tokens:,} < target {target_tokens:,} tokens; writing truncated file.")
            tmp_path = output_path + ".partial.npy"
            open_memmap = getattr(np.lib.format, "open_memmap", None)
            if not callable(open_memmap):
                raise ImportError("numpy.lib.format.open_memmap not available")
            mm_in_obj: object = _np_load_obj(output_path, mmap_mode="r")
            if not isinstance(mm_in_obj, np.memmap):
                raise TypeError("np.load(..., mmap_mode='r') did not return numpy.memmap")
            mm_out_obj: object = open_memmap(tmp_path, mode="w+", dtype=np.uint16, shape=(int(total_tokens),))
            if not isinstance(mm_out_obj, np.memmap):
                raise TypeError("open_memmap did not return numpy.memmap")
            mm_out_obj[:] = mm_in_obj[: int(total_tokens)]
            del mm_out_obj
            del mm_in_obj
            os.replace(tmp_path, output_path)
    else:
        if txt_writer is not None:
            try:
                txt_writer.flush()
            except (OSError, ValueError, AttributeError, TypeError):
                pass
            try:
                txt_writer.close()
            except (OSError, ValueError, AttributeError, TypeError):
                pass

    print(f"\nCollected {int(total_tokens):,} tokens from {doc_count:,} documents")
    print(f"Saved to {output_path}")

    # Verify
    file_size = os.path.getsize(output_path)
    print(f"Done! File size: {file_size / 1e6:.1f} MB")

    # Write metadata
    meta_path = output_path + ".meta"
    with open(meta_path, "w", encoding="utf-8") as f:
        _ = f.write(f"tokens: {int(total_tokens)}\n")
        _ = f.write(f"documents: {doc_count}\n")
        _ = f.write("encoding: gpt2\n")
        _ = f.write(f"vocab_size: {int(enc.n_vocab)}\n")
        _ = f.write("source: HuggingFaceFW/fineweb-edu\n")
        _ = f.write(f"format: {output_format}\n")
    print(f"Metadata saved to {meta_path}")

    return True


def verify_existing(path: str) -> int | None:
    """Check if file exists and return token count."""
    if not os.path.exists(path):
        return None

    meta_path = path + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("tokens:"):
                    return int(line.split(":")[1].strip())

    # Count tokens in file (slow)
    print(f"Counting tokens in {path}...")
    if path.endswith(".npy"):
        try:
            # Use mmap_mode='r' to peek without full load
            arr_obj = _np_load_obj(path, mmap_mode="r")
            if isinstance(arr_obj, np.ndarray) and len(arr_obj.shape) >= 1:
                return int(arr_obj.shape[0])
            return None
        except (OSError, ValueError, TypeError) as e:
            print(f"Error reading npy: {e}")
            return None
    else:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            tokens = len(content.split())
        return tokens


def main() -> int | None:
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset")
    _ = parser.add_argument(
        "--tokens",
        type=str,
        default="100M",
        help="Number of tokens to collect (e.g., 100M, 1B)",
    )
    _ = parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: fineweb_{tokens}.tokens)",
    )
    _ = parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    _ = parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "text", "npy"],
        help="Output format (text or npy). Auto infers from output filename.",
    )

    ns = parser.parse_args()
    args = _as_str_object_dict(ns.__dict__) or {}
    tokens_s = str(args.get("tokens", "100M"))
    target_tokens = parse_token_count(tokens_s)

    # Determine format and default path
    fmt = str(args.get("format", "auto"))

    out_opt = args.get("output", None)
    output_raw = str(out_opt) if isinstance(out_opt, str) else ""
    if output_raw:
        output_path = output_raw
        if fmt == "auto":
            fmt = "npy" if output_path.endswith(".npy") else "text"
    else:
        # Format nicely: 100M, 1B, etc.
        if target_tokens >= 1_000_000_000:
            suffix = f"{target_tokens // 1_000_000_000}B"
        elif target_tokens >= 1_000_000:
            suffix = f"{target_tokens // 1_000_000}M"
        else:
            suffix = f"{target_tokens // 1000}K"

        if fmt == "auto":
            # Default to text for backward compat, unless user wants npy.
            # But wait, user query is specifically about npy.
            # I will let user specify via --output x.npy or --format npy
            fmt = "text"

        ext = "npy" if fmt == "npy" else "tokens"
        output_path = f"fineweb_{suffix.lower()}.{ext}"

    # Check if already exists
    if not bool(args.get("force", False)):
        existing = verify_existing(output_path)
        if existing is not None:
            print(f"File {output_path} already exists with {existing:,} tokens")
            if existing >= target_tokens:
                print("Sufficient tokens already available. Use --force to re-download.")
                return
            else:
                print(f"Need {target_tokens - existing:,} more tokens. Re-downloading...")

    # Download and tokenize
    success = download_and_tokenize(target_tokens, output_path, output_format=fmt)

    if success:
        print("\n" + "=" * 60)
        print(f"SUCCESS! Dataset ready at: {output_path}")
        print("=" * 60)
        print("\nTo train, run:")
        print("  python3.12 v28_transformer_decoupled_bottleneck_instrumented.py \\")
        print(f"      --data {output_path} \\")
        if fmt == "npy":
            print("      --data-format npy \\")
        print("      --out-dir runs/paper_experiment \\")
        print("      --tokenizer tiktoken \\")
        print("      --block 1024 \\")
        print("      --instrument medium")
    else:
        print("\nFailed to prepare dataset")
        return 1


if __name__ == "__main__":
    _ = main()
