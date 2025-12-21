#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Allow running as either:
#   - python -m production.bench_end_to_end_memory  (preferred)
#   - python production/bench_end_to_end_memory.py  (convenient)
#
# When executed as a script, Python puts `production/` on sys.path (not the repo root),
# so absolute imports like `import production.config` fail. Add the repo root.
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repo_root))

from production.config import pick_device
from production.memory_utils import (
    device_synchronize,
    diff_mem_stats,
    empty_device_cache,
    get_device_mem_stats,
    pick_primary_mem_key,
    reset_peak_memory_stats,
)
from production.kvcache_backend import KVCacheTensorConfig, LayerKVCache
from production.model import ModelConfig
from production.runtime_tuning import KVCachePolicy, estimate_decoupled_kvcache_bytes, estimate_seq_cache_bytes, load_token_ids_spec


def _now_iso() -> str:
    try:
        import datetime

        return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")
    except Exception:
        return str(time.time())


def _load_config(*, ckpt_path: str, device: torch.device, force_block_size: Optional[int]) -> ModelConfig:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg_dict = ckpt.get("config", None)
    if cfg_dict is None:
        raise ValueError("Checkpoint missing 'config'. Can't reconstruct model safely.")

    cfg = ModelConfig(**cfg_dict)

    # Safety for long-context benchmarking: avoid allocating a gigantic causal mask for null-attn.
    cfg.null_attn = False
    if force_block_size is not None:
        cfg.block_size = int(max(int(cfg.block_size), int(force_block_size)))
    return cfg


def _policy_from_args(args: argparse.Namespace) -> KVCachePolicy:
    if getattr(args, "policy", None):
        return KVCachePolicy.parse(str(args.policy))
    # Default validated heterogeneous policy (q4/q8/q4) + residual tail.
    return KVCachePolicy(
        k_sem_kind="q4_0",
        k_geo_kind="q8_0",
        v_kind="q4_0",
        k_sem_qblock=32,
        k_geo_qblock=32,
        v_qblock=32,
        residual_len=int(getattr(args, "residual_len", 128)),
    )


def _baseline_fp16_policy() -> KVCachePolicy:
    return KVCachePolicy(
        k_sem_kind="fp16",
        k_geo_kind="fp16",
        v_kind="fp16",
        k_sem_qblock=32,
        k_geo_qblock=32,
        v_qblock=32,
        residual_len=0,
    )


def _alloc_decoupled_caches(
    cfg: ModelConfig,
    *,
    batch_size: int,
    max_seq_len: int,
    policy: KVCachePolicy,
    device: torch.device,
    kv_decode_block: int,
    kv_fused: str,
) -> list[Any]:
    k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
    caches: list[Any] = []
    from production.kvcache_backend import DecoupledLayerKVCache

    for _ in range(int(cfg.n_layer)):
        c = DecoupledLayerKVCache(
            batch_size=int(batch_size),
            max_seq_len=int(max_seq_len),
            k_sem_dim=int(cfg.sem_dim),
            k_geo_dim=int(cfg.geo_dim),
            v_dim=int(cfg.attn_dim),
            k_sem_cfg=k_sem_cfg,
            k_geo_cfg=k_geo_cfg,
            v_cfg=v_cfg,
            device=device,
        )
        c.decode_block = int(kv_decode_block)
        c.fused = str(kv_fused)
        caches.append(c)
    return caches


def _kv_dim_for(cfg: ModelConfig) -> int:
    if str(cfg.attn_mode) == "standard":
        return int(cfg.d_model)
    if str(cfg.attn_mode) == "gqa":
        kv_head = int(cfg.kv_head) if cfg.kv_head is not None else int(cfg.n_head)
        head_dim = int(cfg.attn_dim) // int(cfg.n_head)
        return int(kv_head * head_dim)
    return int(cfg.attn_dim)


def _alloc_layer_caches(
    cfg: ModelConfig,
    *,
    batch_size: int,
    max_seq_len: int,
    k_cfg: KVCacheTensorConfig,
    v_cfg: KVCacheTensorConfig,
    device: torch.device,
    kv_decode_block: int,
    kv_fused: str,
) -> list[Any]:
    dim = _kv_dim_for(cfg)
    caches: list[Any] = []
    for _ in range(int(cfg.n_layer)):
        c = LayerKVCache(
            batch_size=int(batch_size),
            max_seq_len=int(max_seq_len),
            k_dim=int(dim),
            v_dim=int(dim),
            k_cfg=k_cfg,
            v_cfg=v_cfg,
            device=device,
        )
        c.decode_block = int(kv_decode_block)
        c.fused = str(kv_fused)
        caches.append(c)
    return caches


def _measure_alloc_delta(
    cfg: ModelConfig,
    *,
    batch_size: int,
    max_seq_len: int,
    attn_mode: str,
    policy: Optional[KVCachePolicy],
    kv_kind: str,
    device: torch.device,
    kv_decode_block: int,
    kv_fused: str,
) -> Dict[str, Any]:
    # Ensure previous allocations are released.
    gc.collect()
    empty_device_cache(device)
    device_synchronize(device)
    reset_peak_memory_stats(device)

    before = get_device_mem_stats(device, include_cuda_peaks=True)

    cfg2 = ModelConfig(**asdict(cfg))
    cfg2.attn_mode = str(attn_mode)  # type: ignore[assignment]

    if str(attn_mode) == "decoupled":
        if policy is None:
            raise ValueError("decoupled measurement requires a KVCachePolicy")
        caches = _alloc_decoupled_caches(
            cfg2,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            policy=policy,
            device=device,
            kv_decode_block=kv_decode_block,
            kv_fused=kv_fused,
        )
    else:
        k_cfg = KVCacheTensorConfig(kind=str(kv_kind), qblock=32, residual_len=0)
        v_cfg = KVCacheTensorConfig(kind=str(kv_kind), qblock=32, residual_len=0)
        caches = _alloc_layer_caches(
            cfg2,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            k_cfg=k_cfg,
            v_cfg=v_cfg,
            device=device,
            kv_decode_block=kv_decode_block,
            kv_fused=kv_fused,
        )
    _ = caches  # keep alive
    device_synchronize(device)

    after = get_device_mem_stats(device, include_cuda_peaks=True)
    delta = diff_mem_stats(after, before)

    # Cleanup.
    del caches
    gc.collect()
    empty_device_cache(device)
    device_synchronize(device)

    return {
        "before": before,
        "after": after,
        "delta": delta,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Measure end-to-end memory footprint at 128k for KV-cache policies (production stack).")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path produced by production runner (must contain 'config' and 'model').")
    ap.add_argument("--context-len", type=int, default=131072, help="Target max_seq_len for cache allocation (e.g. 131072 for 128k).")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", type=str, default=None, help="Override device string (e.g. cuda, mps).")
    ap.add_argument("--out", type=str, default=None, help="Write JSON results to this path (default: alongside ckpt).")
    ap.add_argument("--dry-run", action="store_true", help="Only compute estimated KV-cache bytes; do not allocate caches or measure device memory.")
    ap.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "standard", "bottleneck", "gqa", "decoupled"],
        help="Which attention mode to measure for this run. 'auto' uses the checkpoint config.",
    )
    ap.add_argument(
        "--kv-kind",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
        help="Homogeneous KV kind for non-decoupled modes (and for decoupled if --policy is not provided).",
    )
    ap.add_argument(
        "--decompose",
        action="store_true",
        help="If the checkpoint has decoupled dims, also compute (standard FP16 -> decoupled FP16 -> decoupled policy) factors.",
    )
    ap.add_argument("--baseline-mode", type=str, default="standard", choices=["standard", "bottleneck", "gqa"], help="Baseline mode for decomposition (FP16).")

    # Candidate policy (validated default: q4/q8/q4).
    ap.add_argument("--policy", type=str, default=None, help='Override candidate policy string, e.g. "ksem=q4_0@32,kgeo=q8_0@32,v=q4_0@32,resid=128".')
    ap.add_argument("--residual-len", type=int, default=128, help="Residual fp16 tail length for candidate policy (ignored if --policy is set).")

    # Decode knobs (affects cache metadata and potential fused selection; cache allocation dominates memory).
    ap.add_argument("--kv-decode-block", type=int, default=1024)
    ap.add_argument("--kv-fused", type=str, default="auto", choices=["none", "auto", "triton1pass", "triton2pass"])

    # Optional calibration tokens: stored in the JSON for reproducibility (not required for memory-only mode).
    ap.add_argument("--tokens", type=str, default=None, help="Optional token spec (path to .txt/.npy or whitespace-separated ints). Stored for provenance.")

    args = ap.parse_args(argv)

    dev = pick_device(args.device) if args.device is None else torch.device(str(args.device))

    ctx = int(args.context_len)
    B = int(args.batch_size)
    kv_decode_block = int(args.kv_decode_block)
    kv_fused = str(args.kv_fused)

    ckpt_path = str(args.ckpt)
    out_path = args.out
    if out_path is None:
        out_path = str(Path(ckpt_path).with_suffix(".mem128k.json"))

    # Record token provenance (optional).
    tokens_preview: Optional[list[int]] = None
    if args.tokens:
        try:
            ids = load_token_ids_spec(str(args.tokens))
            tokens_preview = [int(x) for x in ids[:64]]
        except Exception:
            tokens_preview = None

    cfg = _load_config(ckpt_path=ckpt_path, device=dev, force_block_size=ctx)

    mode_to_measure = str(args.mode)
    if mode_to_measure == "auto":
        mode_to_measure = str(getattr(cfg, "attn_mode", "standard"))

    # Per-run estimate for the selected mode.
    est_selected: Dict[str, Any] = {}
    if mode_to_measure == "decoupled":
        pol = _policy_from_args(args) if getattr(args, "policy", None) else KVCachePolicy(
            k_sem_kind=str(args.kv_kind),
            k_geo_kind=str(args.kv_kind),
            v_kind=str(args.kv_kind),
            k_sem_qblock=32,
            k_geo_qblock=32,
            v_qblock=32,
            residual_len=0 if str(args.kv_kind) in ("fp16", "fp32") else int(getattr(args, "residual_len", 128)),
        )
        est_selected = {
            "mode": "decoupled",
            "policy": asdict(pol),
            "estimated_bytes": int(
                estimate_decoupled_kvcache_bytes(
                    n_layer=int(cfg.n_layer),
                    batch_size=B,
                    max_seq_len=ctx,
                    sem_dim=int(cfg.sem_dim),
                    geo_dim=int(cfg.geo_dim),
                    v_dim=int(cfg.attn_dim),
                    policy=pol,
                )
            ),
        }
    else:
        cfg_base = ModelConfig(**asdict(cfg))
        cfg_base.attn_mode = mode_to_measure  # type: ignore[assignment]
        dim = _kv_dim_for(cfg_base)
        kind = str(args.kv_kind)
        tcfg = KVCacheTensorConfig(kind=kind, qblock=32, residual_len=(0 if kind in ("fp16", "fp32") else 0))
        est_bytes = int(cfg.n_layer) * (
            estimate_seq_cache_bytes(batch_size=B, max_seq_len=ctx, dim=dim, cfg=tcfg)
            + estimate_seq_cache_bytes(batch_size=B, max_seq_len=ctx, dim=dim, cfg=tcfg)
        )
        est_selected = {"mode": mode_to_measure, "kv_kind": kind, "estimated_bytes": int(est_bytes)}

    payload: Dict[str, Any] = {
        "ts": _now_iso(),
        "ckpt": str(ckpt_path),
        "device": str(dev),
        "platform": {
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", "unknown"),
            "os": platform.platform(),
        },
        "model_cfg": asdict(cfg),
        "context_len": int(ctx),
        "batch_size": int(B),
        "kv_decode_block": int(kv_decode_block),
        "kv_fused": str(kv_fused),
        "tokens_preview": tokens_preview,
        "estimate_selected": est_selected,
        "decomposition": None,
        "measured": None,
    }

    if args.dry_run:
        Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[bench] dry-run wrote {out_path}")
        print(f"[bench] estimated_bytes[{est_selected.get('mode')}]: {int(est_selected.get('estimated_bytes', 0))}")
        return 0

    primary_key = pick_primary_mem_key(dev)

    # Always measure the selected mode allocation delta.
    pol_sel: Optional[KVCachePolicy] = None
    if mode_to_measure == "decoupled":
        pol_sel = KVCachePolicy(**est_selected["policy"]) if "policy" in est_selected else None  # type: ignore[arg-type]
    meas_selected = _measure_alloc_delta(
        cfg,
        batch_size=B,
        max_seq_len=ctx,
        attn_mode=mode_to_measure,
        policy=pol_sel,
        kv_kind=str(args.kv_kind),
        device=dev,
        kv_decode_block=kv_decode_block,
        kv_fused=kv_fused,
    )

    payload["measured"] = {"primary_key": primary_key, "selected": meas_selected}

    # Optional decomposition (paper mode): standard FP16 -> decoupled FP16 -> decoupled policy.
    if bool(getattr(args, "decompose", False)) and hasattr(cfg, "sem_dim") and hasattr(cfg, "geo_dim"):
        base_mode = str(args.baseline_mode)
        base_policy_dec = _baseline_fp16_policy()
        cand_policy = _policy_from_args(args)

        # Estimate decomposition (bytes) independent of weights.
        dim_base = _kv_dim_for(ModelConfig(**{**asdict(cfg), "attn_mode": base_mode}))
        fp16_cfg = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
        est_base_std = int(cfg.n_layer) * (
            estimate_seq_cache_bytes(batch_size=B, max_seq_len=ctx, dim=dim_base, cfg=fp16_cfg)
            + estimate_seq_cache_bytes(batch_size=B, max_seq_len=ctx, dim=dim_base, cfg=fp16_cfg)
        )
        est_dec_fp16 = int(
            estimate_decoupled_kvcache_bytes(
                n_layer=int(cfg.n_layer),
                batch_size=B,
                max_seq_len=ctx,
                sem_dim=int(cfg.sem_dim),
                geo_dim=int(cfg.geo_dim),
                v_dim=int(cfg.attn_dim),
                policy=base_policy_dec,
            )
        )
        est_cand = int(
            estimate_decoupled_kvcache_bytes(
                n_layer=int(cfg.n_layer),
                batch_size=B,
                max_seq_len=ctx,
                sem_dim=int(cfg.sem_dim),
                geo_dim=int(cfg.geo_dim),
                v_dim=int(cfg.attn_dim),
                policy=cand_policy,
            )
        )
        est_arch = float(est_base_std) / float(max(1, est_dec_fp16))
        est_quant = float(est_dec_fp16) / float(max(1, est_cand))
        est_e2e = float(est_base_std) / float(max(1, est_cand))

        meas_base = _measure_alloc_delta(
            cfg,
            batch_size=B,
            max_seq_len=ctx,
            attn_mode=base_mode,
            policy=None,
            kv_kind="fp16",
            device=dev,
            kv_decode_block=kv_decode_block,
            kv_fused=kv_fused,
        )
        meas_dec_fp16 = _measure_alloc_delta(
            cfg,
            batch_size=B,
            max_seq_len=ctx,
            attn_mode="decoupled",
            policy=base_policy_dec,
            kv_kind="fp16",
            device=dev,
            kv_decode_block=kv_decode_block,
            kv_fused=kv_fused,
        )
        meas_cand = _measure_alloc_delta(
            cfg,
            batch_size=B,
            max_seq_len=ctx,
            attn_mode="decoupled",
            policy=cand_policy,
            kv_kind="fp16",
            device=dev,
            kv_decode_block=kv_decode_block,
            kv_fused=kv_fused,
        )

        ratio_arch_measured = float("nan")
        ratio_quant_measured = float("nan")
        ratio_e2e_measured = float("nan")
        if primary_key is not None:
            a = float(meas_base["delta"].get(primary_key, 0.0))
            b = float(meas_dec_fp16["delta"].get(primary_key, 0.0))
            c = float(meas_cand["delta"].get(primary_key, 0.0))
            if a > 0 and b > 0:
                ratio_arch_measured = float(a / b)
            if b > 0 and c > 0:
                ratio_quant_measured = float(b / c)
            if a > 0 and c > 0:
                ratio_e2e_measured = float(a / c)

        payload["decomposition"] = {
            "baseline_mode": base_mode,
            "policies": {"decoupled_fp16": asdict(base_policy_dec), "candidate_decoupled": asdict(cand_policy)},
            "estimate_bytes": {
                "baseline_fp16_standard_like": int(est_base_std),
                "decoupled_fp16": int(est_dec_fp16),
                "decoupled_candidate": int(est_cand),
                "ratio_arch_standard_over_decoupled_fp16": float(est_arch),
                "ratio_quant_decoupled_fp16_over_candidate": float(est_quant),
                "ratio_e2e_standard_over_candidate": float(est_e2e),
            },
            "measured": {
                "baseline_fp16_standard_like": meas_base,
                "decoupled_fp16": meas_dec_fp16,
                "decoupled_candidate": meas_cand,
                "ratio_arch_standard_over_decoupled_fp16": ratio_arch_measured,
                "ratio_quant_decoupled_fp16_over_candidate": ratio_quant_measured,
                "ratio_e2e_standard_over_candidate": ratio_e2e_measured,
            },
        }

    Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bench] wrote {out_path}")
    print(f"[bench] estimated_bytes[{est_selected.get('mode')}]: {int(est_selected.get('estimated_bytes', 0))}")
    if payload.get("decomposition") is not None and primary_key is not None:
        d = payload["decomposition"]["measured"]
        print(f"[bench] measured ratio arch on {primary_key}: {float(d.get('ratio_arch_standard_over_decoupled_fp16', float('nan'))):.3f}x")
        print(f"[bench] measured ratio quant on {primary_key}: {float(d.get('ratio_quant_decoupled_fp16_over_candidate', float('nan'))):.3f}x")
        print(f"[bench] measured ratio e2e on {primary_key}: {float(d.get('ratio_e2e_standard_over_candidate', float('nan'))):.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


