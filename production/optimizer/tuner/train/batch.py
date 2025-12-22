"""Batch/grad-accum tuning for training."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Protocol

import torch
import torch.nn.functional as F

from production.memory_utils import device_synchronize, empty_device_cache, get_device_mem_stats
from production.selfopt_cache import as_object_pair2, as_str_object_dict, get_cache_entry, set_cache_entry
from production.selfopt_utils import device_sig, hash_cfg, is_oom_error, restore_rng, snapshot_rng

from production.optimizer.tuner.train.types import TrainBatchPlan

BATCH_TUNER_VERSION = 2


class _TrainTuningCfg(Protocol):
    vocab_size: int
    block_size: int


class _TrainTuningModel(Protocol):
    training: bool

    def train(self, mode: bool = True) -> "_TrainTuningModel": ...

    def eval(self) -> "_TrainTuningModel": ...

    def zero_grad(self, *, set_to_none: bool = True) -> None: ...

    def parameters(self) -> Iterable[torch.nn.Parameter]: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]: ...


def _extract_logits(model_out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(model_out, tuple):
        if not model_out:
            raise ValueError("model output tuple is empty")
        return model_out[0]
    return model_out


def _derive_min_stable_tokens(*, model_params: int, dataset_tokens: int) -> int:
    """
    Derive minimum effective batch size for stable gradients.

    Smaller models need larger batches (less internal averaging capacity).
    Larger datasets benefit from larger batches (more diversity).
    """
    # Base on Chinchilla-style scaling: tokens per param
    tokens_per_param = float(dataset_tokens) / max(1.0, float(model_params))

    # Small models (undertrained): need ~100k-200k tokens/batch
    # Large models (well-trained): can go lower, ~50k-100k
    if model_params < 10_000_000:
        base = 150_000
    elif model_params < 100_000_000:
        base = 100_000
    else:
        base = 75_000

    # If dataset is large relative to params (long training), can use smaller batches
    if tokens_per_param > 50:  # well beyond Chinchilla optimal
        base = int(base * 0.7)

    return int(base)


def _find_best_decomposition(
    *,
    target_effective: int,
    seq_len: int,
    device: torch.device,
    cfg: _TrainTuningCfg,
    model: _TrainTuningModel,
    get_batch: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    iters: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[int, int, float] | None:
    """
    Find best (bs, ga) decomposition for a target effective batch size.

    Returns (bs, ga, tok_s) or None if infeasible.
    Strategy: try different bs values, compute required ga, benchmark.
    """
    best_pair: tuple[int, int] | None = None
    best_tok_s = -1.0

    # Candidate microbatch sizes (prefer larger bs for efficiency, but respect memory)
    max_bs = 128 if device.type == "mps" else 256
    bs_candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    bs_candidates = [b for b in bs_candidates if b <= max_bs]

    for bs in bs_candidates:
        # Compute required ga to hit target effective batch
        tokens_per_micro = bs * seq_len
        if tokens_per_micro == 0:
            continue
        ga = max(1, int(math.ceil(float(target_effective) / float(tokens_per_micro))))

        # Skip if effective batch is way off target (>20% deviation)
        actual_effective = bs * ga * seq_len
        if abs(actual_effective - target_effective) / max(target_effective, 1) > 0.2:
            continue

        # Benchmark this (bs, ga) pair
        tok_s = _benchmark_pair(
            bs=bs,
            ga=ga,
            seq_len=seq_len,
            device=device,
            cfg=cfg,
            model=model,
            get_batch=get_batch,
            warmup=warmup,
            iters=iters,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        if tok_s > best_tok_s:
            best_tok_s = tok_s
            best_pair = (bs, ga)

    if best_pair is None:
        return None
    return (best_pair[0], best_pair[1], best_tok_s)


def _benchmark_pair(
    *,
    bs: int,
    ga: int,
    seq_len: int,
    device: torch.device,
    cfg: _TrainTuningCfg,
    model: _TrainTuningModel,
    get_batch: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    iters: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Benchmark a specific (bs, ga) pair, return tok/s or -1 if fails."""
    # MPS guard for large tensors
    if device.type == "mps":
        try:
            v = int(cfg.vocab_size)
            if v > 0 and bs * seq_len * v > 2_147_483_647:
                return -1.0
        except Exception:
            pass

    try:
        empty_device_cache(device)
        device_synchronize(device)

        if amp_enabled:
            cast_ctx = torch.autocast(device.type, dtype=amp_dtype) if device.type != "cpu" else torch.autocast("cpu", dtype=torch.bfloat16)
        else:
            cast_ctx = nullcontext()

        # Warmup
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            xb, yb = get_batch(bs, seq_len)
            with cast_ctx:
                warmup_logits = _extract_logits(model(xb))
                loss = F.cross_entropy(warmup_logits.reshape(-1, warmup_logits.size(-1)), yb.reshape(-1))
            torch.autograd.backward(loss / ga)

        device_synchronize(device)

        # Timed iterations
        t0 = time.perf_counter()
        tok = 0
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            for _ in range(ga):
                xb, yb = get_batch(bs, seq_len)
                with cast_ctx:
                    iter_logits = _extract_logits(model(xb))
                    loss = F.cross_entropy(iter_logits.reshape(-1, iter_logits.size(-1)), yb.reshape(-1))
                torch.autograd.backward(loss / ga)
                tok += xb.numel()

        device_synchronize(device)
        dt = time.perf_counter() - t0

        model.zero_grad(set_to_none=True)
        return float(tok / max(dt, 1e-9))

    except Exception as e:
        if not is_oom_error(e):
            # Silent failure for non-OOM errors during probing
            pass
        return -1.0


def _default_micro_batches(target_gbs: int) -> list[int]:
    # Prefer larger microbatches first (less grad-accum overhead).
    out: list[int] = []
    g = int(max(1, target_gbs))
    b = g
    while b >= 1:
        out.append(int(b))
        if b == 1:
            break
        b = max(1, b // 2)
    # De-dup while preserving order.
    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def tune_batch_by_seq(
    *,
    cache_path: str | None,
    device: torch.device,
    cfg: _TrainTuningCfg,
    model: _TrainTuningModel,
    get_batch: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
    seq_lens: list[int],
    target_gbs: int,
    warmup: int = 1,
    iters: int = 2,
    verbose: bool = False,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> TrainBatchPlan:
    """Tune (batch_size, grad_accum) per seq_len, with caching and RNG preservation."""
    seq_lens = sorted({int(s) for s in seq_lens if int(s) > 0})
    if not seq_lens:
        raise ValueError("seq_lens is empty")

    target_gbs = int(target_gbs)
    auto = bool(target_gbs <= 0)
    warmup = int(max(0, warmup))
    iters = int(max(1, iters))

    key = (
        f"{device_sig(device)}|train|v={BATCH_TUNER_VERSION}|cfg={hash_cfg(cfg)}|"
        f"{'auto' if auto else f'gbs={target_gbs}'}|amp={int(bool(amp_enabled))}|ampdt={str(amp_dtype)}"
    )

    if cache_path:
        cached_obj: object | None = get_cache_entry(cache_path, section="train_plans", key=key)
        cached = as_str_object_dict(cached_obj)
        if cached is not None and "by_seq" in cached:
            try:
                by_seq_from_cache: dict[int, tuple[int, int]] = {}
                raw_by_seq = as_str_object_dict(cached.get("by_seq"))
                if raw_by_seq is not None:
                    for k_str, cache_val in raw_by_seq.items():
                        try:
                            seq = int(k_str)
                        except (TypeError, ValueError):
                            continue

                        pair = as_object_pair2(cache_val)
                        if pair is None:
                            continue
                        v0_item, v1_item = pair

                        if not isinstance(v0_item, (int, float, str)):
                            continue
                        if not isinstance(v1_item, (int, float, str)):
                            continue
                        by_seq_from_cache[seq] = (int(v0_item), int(v1_item))
                if by_seq_from_cache:
                    return TrainBatchPlan(by_seq=by_seq_from_cache, target_gbs=target_gbs, warmup=warmup, iters=iters)
            except Exception:
                pass

    snap = snapshot_rng(device)
    try:
        model_was_training = bool(model.training)
        _ = model.train()

        bs_list = _default_micro_batches(target_gbs) if not auto else []
        by_seq: dict[int, tuple[int, int]] = {}

        # Derive minimum stable effective batch from model scale
        model_params = sum(p.numel() for p in model.parameters())
        dataset_tokens_est = int(cfg.block_size) * 100_000
        min_stable_tokens = _derive_min_stable_tokens(
            model_params=int(model_params),
            dataset_tokens=int(dataset_tokens_est)
        )

        for s in seq_lens:
            best_tok_s = -1.0
            best_pair: tuple[int, int] | None = None
            best_peak = 0.0

            if verbose:
                print(f"[selfopt][train] tuning seq_len={s} (auto gbs)" if auto else f"[selfopt][train] tuning seq_len={s} target_gbs={target_gbs}")

            # Auto mode: jointly optimize (bs, ga) for stability + throughput
            if auto:
                # Derive (bs, ga) pairs that hit min_stable_tokens, benchmark each
                result = _find_best_decomposition(
                    target_effective=min_stable_tokens,
                    seq_len=int(s),
                    device=device,
                    cfg=cfg,
                    model=model,
                    get_batch=get_batch,
                    warmup=warmup,
                    iters=iters,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )

                if result is not None:
                    bs_best, ga_best, tok_s_best = result
                    best_pair = (bs_best, ga_best)
                    best_tok_s = tok_s_best
                else:
                    # Fallback: minimal safe config
                    best_pair = (1, max(1, min_stable_tokens // max(1, int(s))))
                    best_tok_s = 0.0
            else:
                # Non-auto mode: legacy grid search
                for bs_try in bs_list:
                    # MPSGraph logits-size guard: B*T*V must not exceed INT_MAX.
                    if device.type == "mps":
                        try:
                            max_elems = 2_147_483_647
                            v = int(cfg.vocab_size)
                            if v > 0 and int(bs_try) * int(s) * int(v) > max_elems:
                                continue
                        except Exception:
                            pass

                    ok = True
                    tok = 0
                    peak = 0.0
                    try:
                        empty_device_cache(device)
                        device_synchronize(device)
                        if amp_enabled:
                            if device.type == "cpu":
                                cast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
                            else:
                                cast_ctx = torch.autocast(device.type, dtype=amp_dtype)
                        else:
                            cast_ctx = nullcontext()

                        # Warmup
                        for _ in range(warmup):
                            model.zero_grad(set_to_none=True)
                            xb, yb = get_batch(int(bs_try), int(s))
                            with cast_ctx:
                                bm_w_logits = _extract_logits(model(xb))
                                loss = F.cross_entropy(
                                    bm_w_logits.reshape(-1, bm_w_logits.size(-1)), yb.reshape(-1)
                                )
                            torch.autograd.backward(loss)
                        device_synchronize(device)

                        # Timed iters
                        t0 = time.perf_counter()
                        tok = 0
                        for _ in range(iters):
                            model.zero_grad(set_to_none=True)
                            xb, yb = get_batch(int(bs_try), int(s))
                            with cast_ctx:
                                bm_i_logits = _extract_logits(model(xb))
                                loss = F.cross_entropy(
                                    bm_i_logits.reshape(-1, bm_i_logits.size(-1)), yb.reshape(-1)
                                )
                            torch.autograd.backward(loss)
                            tok += int(xb.numel())
                            m = get_device_mem_stats(device)
                            peak = max(
                                peak,
                                float(m.get("mps_mem_driver_bytes", 0.0) or 0.0),
                                float(m.get("cuda_mem_reserved_bytes", 0.0) or 0.0),
                            )
                        device_synchronize(device)
                        dt = time.perf_counter() - t0
                        tok_s = float(tok / max(dt, 1e-9))
                    except Exception as e:
                        ok = False
                        tok_s = 0.0
                        if not is_oom_error(e) and verbose:
                            print(f"[selfopt][train] auto bs={bs_try} failed: {e}")
                    finally:
                        try:
                            model.zero_grad(set_to_none=True)
                        except Exception:
                            pass

                    if not ok:
                        continue
                    if tok_s > best_tok_s:
                        best_tok_s = float(tok_s)
                        best_pair = (int(bs_try), 1)  # ga=1 for non-auto mode
                        best_peak = float(peak)

            # Targeted mode: match requested global batch size with grad-accum.
            for bs_try in bs_list:
                # MPSGraph logits-size guard: B*T*V must not exceed INT_MAX.
                if device.type == "mps":
                    try:
                        max_elems = 2_147_483_647
                        v = int(cfg.vocab_size)
                        if v > 0 and int(bs_try) * int(s) * int(v) > max_elems:
                            continue
                    except Exception:
                        pass

                ga_try = int((max(1, target_gbs) + int(bs_try) - 1) // int(bs_try))
                ga_try = max(1, ga_try)
                gbs_eff = int(bs_try) * int(ga_try)
                if gbs_eff < max(1, target_gbs // 4):
                    continue

                ok = True
                tok = 0
                peak = 0.0
                try:
                    empty_device_cache(device)
                    device_synchronize(device)
                    if amp_enabled:
                        if device.type == "cpu":
                            cast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
                        else:
                            cast_ctx = torch.autocast(device.type, dtype=amp_dtype)
                    else:
                        cast_ctx = nullcontext()

                    # Warmup
                    for _ in range(warmup):
                        model.zero_grad(set_to_none=True)
                        for _m in range(ga_try):
                            xb, yb = get_batch(int(bs_try), int(s))
                            with cast_ctx:
                                ga_w_logits = _extract_logits(model(xb))
                                loss = F.cross_entropy(
                                    ga_w_logits.reshape(-1, ga_w_logits.size(-1)), yb.reshape(-1)
                                )
                            torch.autograd.backward(loss / ga_try)
                    device_synchronize(device)

                    # Timed iters
                    t0 = time.perf_counter()
                    tok = 0
                    for _ in range(iters):
                        model.zero_grad(set_to_none=True)
                        for _m in range(ga_try):
                            xb, yb = get_batch(int(bs_try), int(s))
                            with cast_ctx:
                                ga_i_logits = _extract_logits(model(xb))
                                loss = F.cross_entropy(
                                    ga_i_logits.reshape(-1, ga_i_logits.size(-1)), yb.reshape(-1)
                                )
                            torch.autograd.backward(loss / ga_try)
                            tok += int(xb.numel())
                        m = get_device_mem_stats(device)
                        peak = max(
                            peak,
                            float(m.get("mps_mem_driver_bytes", 0.0) or 0.0),
                            float(m.get("cuda_mem_reserved_bytes", 0.0) or 0.0),
                        )
                    device_synchronize(device)
                    dt = time.perf_counter() - t0
                    tok_s = float(tok / max(dt, 1e-9))
                except Exception as e:
                    ok = False
                    tok_s = 0.0
                    if not is_oom_error(e) and verbose:
                        print(f"[selfopt][train] bs={bs_try} ga={ga_try} failed: {e}")
                finally:
                    try:
                        model.zero_grad(set_to_none=True)
                    except Exception:
                        pass

                if not ok:
                    continue
                if tok_s > best_tok_s:
                    best_tok_s = float(tok_s)
                    best_pair = (int(bs_try), int(ga_try))
                    best_peak = float(peak)

            if best_pair is None:
                best_pair = (1, 1 if auto else max(1, target_gbs))

            # Auto mode: derive grad_accum for stable gradients
            if auto:
                bs_best, _ = best_pair
                tokens_per_micro = int(bs_best) * int(s)
                if tokens_per_micro > 0:
                    # Derive ga to hit min_stable_tokens effective batch
                    target_ga = max(1, int(math.ceil(float(min_stable_tokens) / float(tokens_per_micro))))
                    best_pair = (int(bs_best), int(target_ga))

            by_seq[int(s)] = best_pair
            if verbose:
                bs_b, ga_b = best_pair
                peak_gb = best_peak / (1024.0**3) if best_peak > 0 else 0.0
                print(f"[selfopt][train] best@{s}: bs={bs_b} ga={ga_b} tok/s={best_tok_s:.0f} peak={peak_gb:.2f}GB")

        plan = TrainBatchPlan(by_seq=by_seq, target_gbs=(0 if auto else target_gbs), warmup=warmup, iters=iters)
        if cache_path:
            try:
                payload = {
                    "by_seq": {str(k): [int(v[0]), int(v[1])] for k, v in by_seq.items()},
                    "target_gbs": int(target_gbs),
                    "warmup": int(warmup),
                    "iters": int(iters),
                    "ts": float(time.time()),
                }
                set_cache_entry(str(cache_path), section="train_plans", key=str(key), value=payload)
            except Exception:
                pass

        if not model_was_training:
            _ = model.eval()
        return plan
    finally:
        restore_rng(device, snap)
