"""
Small helpers for device memory/allocator hygiene and measurement.
"""

from __future__ import annotations

import torch


def device_synchronize(dev: torch.device) -> None:
    """Best-effort device synchronize (CUDA/MPS)."""
    try:
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(dev)
        elif dev.type == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except (AttributeError, RuntimeError, TypeError):
        pass


def empty_device_cache(dev: torch.device) -> None:
    """Best-effort cache empty (CUDA/MPS). Does not free live tensors."""
    try:
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif dev.type == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except (AttributeError, RuntimeError, TypeError):
        pass


def reset_peak_memory_stats(dev: torch.device) -> None:
    """Reset CUDA peak memory stats if available (no-op on MPS/CPU)."""
    try:
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(dev)
    except (AttributeError, RuntimeError, TypeError):
        pass


def get_device_mem_stats(dev: torch.device, *, include_cuda_peaks: bool = True) -> dict[str, float]:
    """Return a device-specific snapshot of memory counters in bytes.

    - CUDA: allocated/reserved (+ optional peaks)
    - MPS: current_allocated_memory/driver_allocated_memory
    """
    out: dict[str, float] = {}
    try:
        if dev.type == "cuda" and torch.cuda.is_available():
            out["cuda_mem_alloc_bytes"] = float(torch.cuda.memory_allocated(dev))
            out["cuda_mem_reserved_bytes"] = float(torch.cuda.memory_reserved(dev))
            if include_cuda_peaks:
                out["cuda_mem_peak_alloc_bytes"] = float(torch.cuda.max_memory_allocated(dev))
                out["cuda_mem_peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(dev))
        elif dev.type == "mps":
            if hasattr(torch, "mps"):
                if hasattr(torch.mps, "current_allocated_memory"):
                    out["mps_mem_alloc_bytes"] = float(torch.mps.current_allocated_memory())
                if hasattr(torch.mps, "driver_allocated_memory"):
                    out["mps_mem_driver_bytes"] = float(torch.mps.driver_allocated_memory())
    except (AttributeError, RuntimeError, TypeError):
        pass
    return out


def diff_mem_stats(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    """Key-wise (after-before) for memory stat dicts."""
    out: dict[str, float] = {}
    keys = set(before.keys()) | set(after.keys())
    for k in sorted(keys):
        out[k] = float(after.get(k, 0.0) - before.get(k, 0.0))
    return out


def pick_primary_mem_key(dev: torch.device) -> str | None:
    """Return the primary key to use for ratio computations (best effort)."""
    if dev.type == "cuda" and torch.cuda.is_available():
        return "cuda_mem_reserved_bytes"
    if dev.type == "mps":
        return "mps_mem_driver_bytes"
    return None


