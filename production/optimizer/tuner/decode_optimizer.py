"""Decode-plan self-optimizer.

This file stays intentionally small: it orchestrates plan selection and delegates
keying, persistence, search-space construction, and benchmarking to helper
modules in `production.optimizer.tuner.*`.
"""

from __future__ import annotations

import math
import sys
from typing import Callable

import torch

from production.selfopt_utils import device_sig
from production.optimizer.tuner.config import KVSelfOptConfig
from production.optimizer.tuner.decode_bench import bench_plan as _bench_plan
from production.optimizer.tuner.decode_bench import run_plan as _run_plan
from production.optimizer.tuner.decode_keys import decode_plan_key
from production.optimizer.tuner.decode_plan import KVDecodePlan
from production.optimizer.tuner.profiles import get_triton_kernel_profiles
from production.optimizer.tuner.decode_store import DecodePlanStore
from production.optimizer.tuner.triton_availability import (
    triton_decoupled_q4q8q4_available as _triton_decoupled_q4q8q4_available_fallback,
)


def _get_int_attr(o: object, name: str, default: int) -> int:
    v = getattr(o, name, None)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v)
    return int(default)


class KVDecodeSelfOptimizer:
    """Self-optimizes decode performance knobs per prefix-length bucket."""

    def __init__(
        self,
        cfg: KVSelfOptConfig,
        *,
        device: torch.device,
        base_fused: str,
        base_decode_block: int,
        log_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.cfg: KVSelfOptConfig = cfg
        self.device: torch.device = device
        self.base_fused: str = str(base_fused)
        self.base_decode_block: int = int(base_decode_block)
        self.log_callback: Callable[[dict[str, object]], None] | None = log_callback

        self._store: DecodePlanStore = DecodePlanStore(cfg.cache_path, verbose=bool(cfg.verbose))
        self._plans: dict[str, KVDecodePlan] = self._store.load()
        self._last_probe_step: dict[str, int] = {}
        self._step_counter: int = 0

    def _triton_decoupled_q4q8q4_available(self) -> bool:
        """Resolve Triton availability via patchable runtime_tuning alias when present."""
        mod = sys.modules.get("production.runtime_tuning")
        fn = getattr(mod, "_triton_decoupled_q4q8q4_available", None) if mod is not None else None
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return bool(_triton_decoupled_q4q8q4_available_fallback())
        return bool(_triton_decoupled_q4q8q4_available_fallback())

    def _allowed_fused_modes(self, *, cache: object) -> list[str]:
        """Allowed fused-kernel modes given base preference + cache layout."""
        base_fused = str(self.base_fused)
        if base_fused == "none":
            return ["none"]

        if not self._triton_decoupled_q4q8q4_available():
            return ["none"]

        try:
            k_sem = getattr(cache, "k_sem", None)
            k_geo = getattr(cache, "k_geo", None)
            v = getattr(cache, "v", None)
            k_sem_kind = getattr(k_sem, "kind", None)
            k_geo_kind = getattr(k_geo, "kind", None)
            v_kind = getattr(v, "kind", None)
            ok = (
                isinstance(k_sem_kind, str)
                and k_sem_kind == "q4_0"
                and isinstance(k_geo_kind, str)
                and k_geo_kind == "q8_0"
                and isinstance(v_kind, str)
                and v_kind == "q4_0"
            )
        except (AttributeError, TypeError):
            ok = False
        if not ok:
            return ["none"]

        match base_fused:
            case "triton1pass" | "triton2pass":
                return [base_fused]
            case _:
                return ["none", "triton1pass", "triton2pass"]

    def _candidate_plans(self, *, cache: object) -> list[KVDecodePlan]:
        """Generate candidate plans (test-visible helper)."""
        fused_modes = self._allowed_fused_modes(cache=cache)

        decode_blocks = list(dict.fromkeys([int(self.base_decode_block), *self.cfg.decode_blocks]))
        decode_blocks = [int(x) for x in decode_blocks if int(x) > 0]
        decode_blocks.sort()

        use_profiles = (not bool(getattr(self.cfg, "expert_launch_space", False))) and (
            str(getattr(self.cfg, "kernel_profiles", "auto")) != "off"
        )

        block_ns = [int(x) for x in self.cfg.block_ns if int(x) > 0] or [128]
        warps = [int(x) for x in self.cfg.warps if int(x) > 0] or [4]
        stages = [int(x) for x in self.cfg.stages if int(x) > 0] or [2]

        plans: list[KVDecodePlan] = []
        for fused in fused_modes:
            for db in decode_blocks:
                if fused == "none":
                    plans.append(KVDecodePlan(fused="none", decode_block=int(db)))
                    continue

                if use_profiles:
                    profs = get_triton_kernel_profiles(
                        mode=str(getattr(self.cfg, "kernel_profiles", "auto")),
                        device_sig=device_sig(self.device),
                        fused=str(fused),
                        decode_block=int(db),
                    )
                    for pr in profs:
                        bn = int(pr.block_n)
                        if int(db) < bn:
                            continue
                        if fused == "triton1pass":
                            plans.append(
                                KVDecodePlan(
                                    fused=str(fused),
                                    decode_block=int(db),
                                    block_n=bn,
                                    num_warps_1pass=int(pr.num_warps_1pass),
                                    num_stages_1pass=int(pr.num_stages_1pass),
                                )
                            )
                        else:
                            plans.append(
                                KVDecodePlan(
                                    fused=str(fused),
                                    decode_block=int(db),
                                    block_n=bn,
                                    num_warps_part=int(pr.num_warps_part),
                                    num_stages_part=int(pr.num_stages_part),
                                    num_warps_reduce=int(pr.num_warps_reduce),
                                    num_stages_reduce=int(pr.num_stages_reduce),
                                )
                            )
                    continue

                for bn in block_ns:
                    if int(db) < int(bn):
                        continue
                    for w in warps:
                        for st in stages:
                            if fused == "triton1pass":
                                plans.append(
                                    KVDecodePlan(
                                        fused=str(fused),
                                        decode_block=int(db),
                                        block_n=int(bn),
                                        num_warps_1pass=int(w),
                                        num_stages_1pass=int(st),
                                    )
                                )
                            else:
                                plans.append(
                                    KVDecodePlan(
                                        fused=str(fused),
                                        decode_block=int(db),
                                        block_n=int(bn),
                                        num_warps_part=int(w),
                                        num_stages_part=int(st),
                                        num_warps_reduce=1,
                                        num_stages_reduce=1,
                                    )
                                )
        return plans

    def bench_plan(
        self,
        *,
        attn: object,
        cache: object,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        plan: KVDecodePlan,
        sem_scale: float,
        geo_scale: float,
        baseline_out: torch.Tensor | None,
    ) -> float:
        """Benchmark a plan, optionally verifying vs. `baseline_out`."""
        return _bench_plan(
            self.cfg,
            device=self.device,
            attn=attn,
            cache=cache,
            q_sem=q_sem,
            q_geo=q_geo,
            plan=plan,
            sem_scale=sem_scale,
            geo_scale=geo_scale,
            baseline_out=baseline_out,
        )

    def maybe_get_plan(self, *, attn: object, cache: object, prefix_len: int) -> KVDecodePlan | None:
        """Get the best plan for the given attention/cache signature."""
        if self.cfg.mode == "none":
            return None

        self._step_counter += 1
        k = decode_plan_key(device=self.device, attn=attn, cache=cache, prefix_len=int(prefix_len))

        if k in self._plans and self.cfg.mode == "startup":
            return self._plans[k]

        if k in self._plans and self.cfg.mode == "online":
            last = self._last_probe_step.get(k, -10**9)
            if (self._step_counter - last) < int(self.cfg.interval):
                return self._plans[k]

        plans = self._candidate_plans(cache=cache)
        if not plans:
            return None

        batch_size = 1
        try:
            ks = getattr(cache, "k_sem", None)
            buf = getattr(ks, "buf", None)
            if isinstance(buf, torch.Tensor) and buf.ndim >= 1:
                batch_size = int(buf.shape[0])
            else:
                q = getattr(ks, "q", None)
                if isinstance(q, torch.Tensor) and q.ndim >= 1:
                    batch_size = int(q.shape[0])
        except (AttributeError, TypeError, ValueError):
            batch_size = 1

        head_count = _get_int_attr(attn, "H", 1)
        sem_hd = _get_int_attr(attn, "sem_head_dim", 1)
        geo_hd = _get_int_attr(attn, "geo_head_dim", 1)

        q_sem = torch.randn(
            (batch_size, head_count, 1, int(sem_hd)),
            device=self.device,
            dtype=torch.float16,
        )
        q_geo = torch.randn(
            (batch_size, head_count, 1, int(geo_hd)),
            device=self.device,
            dtype=torch.float16,
        )

        sem_scale = 1.0 / math.sqrt(float(max(1, sem_hd)))
        geo_scale = 1.0 / math.sqrt(float(max(1, geo_hd)))

        baseline_plan = self._plans.get(k, KVDecodePlan(fused="none", decode_block=self.base_decode_block))
        baseline_out = None
        if self.cfg.verify:
            try:
                baseline_out = _run_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=baseline_plan,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                ).detach()
            except (RuntimeError, ValueError, TypeError, AttributeError):
                baseline_out = None

        best_plan: KVDecodePlan | None = None
        best_ms: float = float("inf")

        for p in plans:
            try:
                ms = self.bench_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=p,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                    baseline_out=baseline_out,
                )
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                if self.cfg.verbose:
                    print(f"[selfopt] plan failed {p}: {e}")
                ms = float("inf")
            if ms < best_ms:
                best_ms = ms
                best_plan = p

        if best_plan is not None and k in self._plans and self.cfg.mode == "online":
            old = self._plans[k]
            try:
                old_ms = self.bench_plan(
                    attn=attn,
                    cache=cache,
                    q_sem=q_sem,
                    q_geo=q_geo,
                    plan=old,
                    sem_scale=sem_scale,
                    geo_scale=geo_scale,
                    baseline_out=baseline_out,
                )
            except (RuntimeError, ValueError, TypeError, AttributeError):
                old_ms = float("inf")
            if not best_ms < (old_ms * 1.0 - float(self.cfg.hysteresis)):
                best_plan = old
                best_ms = old_ms
            self._last_probe_step[k] = self._step_counter

        if best_plan is not None:
            self._plans[k] = best_plan
            self._store.save(self._plans)
            if self.cfg.verbose:
                print(f"[selfopt] bucket_key={k} -> {best_plan} ({best_ms:.3f} ms)")
            if self.log_callback:
                self.log_callback(
                    {
                        "type": "analysis",
                        "subtype": "selfopt_decode",
                        "bucket_key": k,
                        "decode_block": int(best_plan.decode_block),
                        "fused": str(best_plan.fused),
                        "block_n": int(best_plan.block_n),
                        "best_ms": float(best_ms),
                    }
                )
        return best_plan

    def choose_plan(
        self,
        *,
        attn: object,
        cache: object,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        sem_scale: float,
        geo_scale: float,
    ) -> KVDecodePlan | None:
        """Backward-compatible alias for `maybe_get_plan` (query values do not affect timing)."""
        _ = (q_sem, q_geo, sem_scale, geo_scale)
        prefix_len = int(getattr(cache, "pos", 0))
        return self.maybe_get_plan(attn=attn, cache=cache, prefix_len=prefix_len)

