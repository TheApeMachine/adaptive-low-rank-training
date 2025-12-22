"""Micro-benchmarks for training tuning."""

from __future__ import annotations

import time
from contextlib import nullcontext
from collections.abc import Callable
from typing import Protocol, cast

import torch
import torch.nn.functional as F

from production.memory_utils import device_synchronize, empty_device_cache


class _CudaGraphStepMarker(Protocol):
    """Typed `cudagraph_mark_step_begin()` callable surface (CUDA-only, optional)."""

    def __call__(self) -> None: ...

class _TrainBenchModel(Protocol):
    """Minimal callable model interface for typed bench loops."""

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, object]: ...


def _maybe_cudagraph_step_begin(*, device: torch.device) -> None:
    """
    Fix for CUDA graphs + torch.compile repeated invocation.

    On some torch versions/backends, compiled graphs require an explicit step
    boundary to avoid "accessing tensor output of CUDAGraphs that has been
    overwritten by a subsequent run".
    """
    if device.type != "cuda":
        return
    torch_obj: object = torch
    compiler_obj: object = getattr(torch_obj, "compiler", None)
    mark_obj: object = getattr(compiler_obj, "cudagraph_mark_step_begin", None)
    if callable(mark_obj):
        cast(_CudaGraphStepMarker, mark_obj)()


def bench_train_tok_s(
    *,
    device: torch.device,
    model: torch.nn.Module,
    get_batch: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    warmup: int,
    iters: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Benchmark training throughput in tokens/sec for a fixed (bs, ga, seq_len)."""
    batch_size = int(max(1, batch_size))
    grad_accum = int(max(1, grad_accum))
    seq_len = int(max(1, seq_len))
    warmup = int(max(0, warmup))
    iters = int(max(1, iters))

    if amp_enabled:
        if device.type == "cpu":
            cast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
        else:
            cast_ctx = torch.autocast(device.type, dtype=amp_dtype)
    else:
        cast_ctx = nullcontext()

    empty_device_cache(device)
    device_synchronize(device)

    _ = model.train()
    model_call = cast(_TrainBenchModel, model)

    # Warmup
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        for _m in range(grad_accum):
            xb, yb = get_batch(batch_size, seq_len)
            _maybe_cudagraph_step_begin(device=device)
            with cast_ctx:
                logits, _cache = model_call(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            torch.autograd.backward(loss / grad_accum)
    device_synchronize(device)

    # Timed
    tok = 0
    t0 = time.perf_counter()
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        for _m in range(grad_accum):
            xb, yb = get_batch(batch_size, seq_len)
            _maybe_cudagraph_step_begin(device=device)
            with cast_ctx:
                logits, _cache = model_call(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            torch.autograd.backward(loss / grad_accum)
            tok += int(xb.numel())
    device_synchronize(device)

    dt = time.perf_counter() - t0
    return float(tok / max(dt, 1e-9))


