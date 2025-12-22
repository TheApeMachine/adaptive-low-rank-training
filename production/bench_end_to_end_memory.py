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
from production.kvcache_backend import DecoupledLayerKVCache, KVCacheKind, KVCacheTensorConfig, LayerKVCache
from production.model import ModelConfig
from production.runtime_tuning import KVCachePolicy, estimate_decoupled_kvcache_bytes, estimate_seq_cache_bytes, load_token_ids_spec
from production.selfopt_cache import as_str_object_dict
from production.selfopt_utils import device_sig


def _now_iso() -> str:
    try:
        import datetime

        return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")
    except (ImportError, AttributeError, OSError, ValueError):
        return str(time.time())


def _torch_load_obj(path: str, *, device: torch.device) -> object:
    # `torch.load` is typed as returning `Any` in stubs; isolate it behind an `object` boundary.
    return torch.load(str(path), map_location=device)  # pyright: ignore[reportAny]


def _args_map(args: argparse.Namespace) -> dict[str, object]:
    d = as_str_object_dict(args.__dict__)
    return {} if d is None else d


def _as_int(o: object, default: int) -> int:
    try:
        return int(str(o))
    except (TypeError, ValueError):
        return int(default)


def _as_str(o: object, default: str) -> str:
    try:
        s = str(o)
        return s
    except (TypeError, ValueError):
        return str(default)


def _as_bool(o: object, default: bool) -> bool:
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, int):
        return bool(o != 0)
    if isinstance(o, str):
        s = o.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
    return bool(default)


def _as_kv_kind(o: object) -> KVCacheKind:
    s = str(o or "").strip().lower()
    if s == "fp32":
        return "fp32"
    if s == "q8_0":
        return "q8_0"
    if s == "q4_0":
        return "q4_0"
    if s == "nf4":
        return "nf4"
    return "fp16"


def _load_config(*, ckpt_path: str, device: torch.device, force_block_size: int | None) -> ModelConfig:
    ckpt_obj = _torch_load_obj(str(ckpt_path), device=device)
    ckpt = as_str_object_dict(ckpt_obj)
    if ckpt is None:
        raise ValueError("Checkpoint payload must be a dict-like object")

    cfg_dict_obj = ckpt.get("config", None)
    cfg_dict = as_str_object_dict(cfg_dict_obj)
    if cfg_dict is None:
        raise ValueError("Checkpoint missing 'config'. Can't reconstruct model safely.")

    cfg = ModelConfig.from_dict(cfg_dict, device=device)

    # Safety for long-context benchmarking: avoid allocating a gigantic causal mask for null-attn.
    cfg.null_attn = False
    if force_block_size is not None:
        cfg.block_size = int(max(int(cfg.block_size), int(force_block_size)))
    return cfg


def _policy_from_args(args_map: dict[str, object]) -> KVCachePolicy:
    pol = args_map.get("policy", None)
    if isinstance(pol, str) and pol.strip():
        return KVCachePolicy.parse(str(pol))
    # Default validated heterogeneous policy (q4/q8/q4) + residual tail.
    return KVCachePolicy(
        k_sem_kind="q4_0",
        k_geo_kind="q8_0",
        v_kind="q4_0",
        k_sem_qblock=32,
        k_geo_qblock=32,
        v_qblock=32,
        residual_len=_as_int(args_map.get("residual_len", 128), 128),
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
) -> list[DecoupledLayerKVCache]:
    k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
    caches: list[DecoupledLayerKVCache] = []
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
) -> list[LayerKVCache]:
    dim = _kv_dim_for(cfg)
    caches: list[LayerKVCache] = []
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
        caches.append(c)
    return caches


def _measure_alloc_delta(
    cfg: ModelConfig,
    *,
    batch_size: int,
    max_seq_len: int,
    attn_mode: str,
    policy: KVCachePolicy | None,
    kv_kind: str,
    device: torch.device,
) -> dict[str, object]:
    # Ensure previous allocations are released.
    _ = gc.collect()
    empty_device_cache(device)
    device_synchronize(device)
    reset_peak_memory_stats(device)

    before = get_device_mem_stats(device, include_cuda_peaks=True)

    cfg2_map = as_str_object_dict(asdict(cfg))
    if cfg2_map is None:
        raise ValueError("Could not materialize config mapping for measurement")
    cfg2 = ModelConfig.from_dict(cfg2_map, device=device)
    cfg2.attn_mode = str(attn_mode)

    if str(attn_mode) == "decoupled":
        if policy is None:
            raise ValueError("decoupled measurement requires a KVCachePolicy")
        caches = _alloc_decoupled_caches(
            cfg2,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            policy=policy,
            device=device,
        )
    else:
        kind = _as_kv_kind(kv_kind)
        k_cfg = KVCacheTensorConfig(kind=kind, qblock=32, residual_len=0)
        v_cfg = KVCacheTensorConfig(kind=kind, qblock=32, residual_len=0)
        caches = _alloc_layer_caches(
            cfg2,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            k_cfg=k_cfg,
            v_cfg=v_cfg,
            device=device,
        )
    device_synchronize(device)

    after = get_device_mem_stats(device, include_cuda_peaks=True)
    delta = diff_mem_stats(after, before)

    # Cleanup.
    del caches
    _ = gc.collect()
    empty_device_cache(device)
    device_synchronize(device)

    return {
        "before": before,
        "after": after,
        "delta": delta,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure end-to-end memory footprint at 128k for KV-cache policies (production stack)."
    )
    _ = ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint path produced by production runner (must contain 'config' and 'model').",
    )
    _ = ap.add_argument(
        "--context-len",
        type=int,
        default=131072,
        help="Target max_seq_len for cache allocation (e.g. 131072 for 128k).",
    )
    _ = ap.add_argument("--batch-size", type=int, default=1)
    _ = ap.add_argument("--device", type=str, default=None, help="Override device string (e.g. cuda, mps).")
    _ = ap.add_argument("--out", type=str, default=None, help="Write JSON results to this path (default: alongside ckpt).")
    _ = ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute estimated KV-cache bytes; do not allocate caches or measure device memory.",
    )
    _ = ap.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "standard", "bottleneck", "gqa", "decoupled"],
        help="Which attention mode to measure for this run. 'auto' uses the checkpoint config.",
    )
    _ = ap.add_argument(
        "--kv-kind",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"],
        help="Homogeneous KV kind for non-decoupled modes (and for decoupled if --policy is not provided).",
    )
    _ = ap.add_argument(
        "--decompose",
        action="store_true",
        help="If the checkpoint has decoupled dims, also compute (standard FP16 -> decoupled FP16 -> decoupled policy) factors.",
    )
    _ = ap.add_argument(
        "--baseline-mode",
        type=str,
        default="standard",
        choices=["standard", "bottleneck", "gqa"],
        help="Baseline mode for decomposition (FP16).",
    )

    # Candidate policy (validated default: q4/q8/q4).
    _ = ap.add_argument(
        "--policy",
        type=str,
        default=None,
        help='Override candidate policy string, e.g. "ksem=q4_0@32,kgeo=q8_0@32,v=q4_0@32,resid=128".',
    )
    _ = ap.add_argument(
        "--residual-len",
        type=int,
        default=128,
        help="Residual fp16 tail length for candidate policy (ignored if --policy is set).",
    )

    # Decode knobs (affects cache metadata and potential fused selection; cache allocation dominates memory).
    _ = ap.add_argument("--kv-decode-block", type=int, default=1024)
    _ = ap.add_argument("--kv-fused", type=str, default="auto", choices=["none", "auto", "triton1pass", "triton2pass"])

    # Optional calibration tokens: stored in the JSON for reproducibility (not required for memory-only mode).
    _ = ap.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Optional token spec (path to .txt/.npy or whitespace-separated ints). Stored for provenance.",
    )

    args = ap.parse_args(argv)
    a = _args_map(args)

    dev_override = a.get("device", None)
    dev_s = None if dev_override is None else str(dev_override)
    dev = pick_device(dev_s) if dev_s is None else torch.device(dev_s)

    ctx = _as_int(a.get("context_len", 131072), 131072)
    B = _as_int(a.get("batch_size", 1), 1)
    kv_decode_block = _as_int(a.get("kv_decode_block", 1024), 1024)
    kv_fused = _as_str(a.get("kv_fused", "auto"), "auto")

    ckpt_path = _as_str(a.get("ckpt", ""), "")
    out_path_obj = a.get("out", None)
    out_path = str(out_path_obj) if isinstance(out_path_obj, str) and out_path_obj else str(Path(ckpt_path).with_suffix(".mem128k.json"))

    # Record token provenance (optional).
    tokens_preview: list[int] | None = None
    tok_spec = a.get("tokens", None)
    if isinstance(tok_spec, str) and tok_spec.strip():
        try:
            ids = load_token_ids_spec(str(tok_spec))
            tokens_preview = [int(x) for x in ids[:64]]
        except (OSError, ValueError, TypeError):
            tokens_preview = None

    cfg = _load_config(ckpt_path=ckpt_path, device=dev, force_block_size=ctx)

    mode_to_measure = _as_str(a.get("mode", "auto"), "auto")
    if mode_to_measure == "auto":
        mode_to_measure = str(getattr(cfg, "attn_mode", "standard"))

    # Per-run estimate for the selected mode.
    est_selected: dict[str, object] = {}
    pol_sel: KVCachePolicy | None = None
    if mode_to_measure == "decoupled":
        kv_kind = _as_kv_kind(a.get("kv_kind", "fp16"))
        pol = _policy_from_args(a) if isinstance(a.get("policy", None), str) else KVCachePolicy(
            k_sem_kind=kv_kind,
            k_geo_kind=kv_kind,
            v_kind=kv_kind,
            k_sem_qblock=32,
            k_geo_qblock=32,
            v_qblock=32,
            residual_len=0 if kv_kind in ("fp16", "fp32") else _as_int(a.get("residual_len", 128), 128),
        )
        pol_sel = pol
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
        cfg_base_map = as_str_object_dict(asdict(cfg))
        if cfg_base_map is None:
            raise ValueError("Could not materialize config mapping for estimate")
        cfg_base = ModelConfig.from_dict(cfg_base_map, device=dev)
        cfg_base.attn_mode = mode_to_measure
        dim = _kv_dim_for(cfg_base)
        kind = _as_kv_kind(a.get("kv_kind", "fp16"))
        tcfg = KVCacheTensorConfig(kind=kind, qblock=32, residual_len=0)
        est_bytes = int(cfg.n_layer) * (
            estimate_seq_cache_bytes(batch_size=B, max_seq_len=ctx, dim=dim, cfg=tcfg)
            + estimate_seq_cache_bytes(batch_size=B, max_seq_len=ctx, dim=dim, cfg=tcfg)
        )
        est_selected = {"mode": mode_to_measure, "kv_kind": kind, "estimated_bytes": int(est_bytes)}

    model_cfg_obj: object = asdict(cfg)
    model_cfg = as_str_object_dict(model_cfg_obj) or {}
    # torch.device is not JSON-serializable; store it as a string in artifacts.
    if "device" in model_cfg:
        model_cfg["device"] = str(model_cfg["device"])

    payload: dict[str, object] = {
        "ts": _now_iso(),
        "ckpt": str(ckpt_path),
        "device": str(dev),
        "device_sig": str(device_sig(dev)),
        "platform": {
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", "unknown"),
            "os": platform.platform(),
        },
        "model_cfg": model_cfg,
        "context_len": int(ctx),
        "batch_size": int(B),
        "kv_decode_block": int(kv_decode_block),
        "kv_fused": str(kv_fused),
        "tokens_preview": tokens_preview,
        "estimate_selected": est_selected,
        "decomposition": None,
        "measured": None,
    }

    if _as_bool(a.get("dry_run", False), False):
        _ = Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[bench] dry-run wrote {out_path}")
        est_b = est_selected.get("estimated_bytes", 0)
        print(f"[bench] estimated_bytes[{est_selected.get('mode')}]: {_as_int(est_b, 0)}")
        return 0

    primary_key = pick_primary_mem_key(dev)

    # Always measure the selected mode allocation delta.
    meas_selected = _measure_alloc_delta(
        cfg,
        batch_size=B,
        max_seq_len=ctx,
        attn_mode=mode_to_measure,
        policy=pol_sel,
        kv_kind=str(_as_kv_kind(a.get("kv_kind", "fp16"))),
        device=dev,
    )

    payload["measured"] = {"primary_key": primary_key, "selected": meas_selected}

    # Optional decomposition (paper mode): standard FP16 -> decoupled FP16 -> decoupled policy.
    if _as_bool(a.get("decompose", False), False) and hasattr(cfg, "sem_dim") and hasattr(cfg, "geo_dim"):
        base_mode = _as_str(a.get("baseline_mode", "standard"), "standard")
        base_policy_dec = _baseline_fp16_policy()
        cand_policy = _policy_from_args(a)

        # Estimate decomposition (bytes) independent of weights.
        base_map = as_str_object_dict(asdict(cfg))
        if base_map is None:
            raise ValueError("Could not materialize config mapping for baseline estimate")
        base_map["attn_mode"] = str(base_mode)
        cfg_base = ModelConfig.from_dict(base_map, device=dev)
        dim_base = _kv_dim_for(cfg_base)
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
        )
        meas_dec_fp16 = _measure_alloc_delta(
            cfg,
            batch_size=B,
            max_seq_len=ctx,
            attn_mode="decoupled",
            policy=base_policy_dec,
            kv_kind="fp16",
            device=dev,
        )
        meas_cand = _measure_alloc_delta(
            cfg,
            batch_size=B,
            max_seq_len=ctx,
            attn_mode="decoupled",
            policy=cand_policy,
            kv_kind="fp16",
            device=dev,
        )

        ratio_arch_measured = float("nan")
        ratio_quant_measured = float("nan")
        ratio_e2e_measured = float("nan")
        if primary_key is not None:
            d0 = as_str_object_dict(meas_base.get("delta"))
            d1 = as_str_object_dict(meas_dec_fp16.get("delta"))
            d2 = as_str_object_dict(meas_cand.get("delta"))
            a0 = 0.0 if d0 is None else d0.get(primary_key, 0.0)
            b0 = 0.0 if d1 is None else d1.get(primary_key, 0.0)
            c0 = 0.0 if d2 is None else d2.get(primary_key, 0.0)
            a = float(a0) if isinstance(a0, (int, float)) else 0.0
            b = float(b0) if isinstance(b0, (int, float)) else 0.0
            c = float(c0) if isinstance(c0, (int, float)) else 0.0
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

    _ = Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bench] wrote {out_path}")
    est_b = est_selected.get("estimated_bytes", 0)
    print(f"[bench] estimated_bytes[{est_selected.get('mode')}]: {_as_int(est_b, 0)}")
    if payload.get("decomposition") is not None and primary_key is not None:
        decomp = as_str_object_dict(payload.get("decomposition"))
        measured = None if decomp is None else as_str_object_dict(decomp.get("measured"))
        if measured is not None:
            ra = measured.get("ratio_arch_standard_over_decoupled_fp16", float("nan"))
            rq = measured.get("ratio_quant_decoupled_fp16_over_candidate", float("nan"))
            re = measured.get("ratio_e2e_standard_over_candidate", float("nan"))
            ra_f = float(ra) if isinstance(ra, (int, float)) else float("nan")
            rq_f = float(rq) if isinstance(rq, (int, float)) else float("nan")
            re_f = float(re) if isinstance(re, (int, float)) else float("nan")
            print(f"[bench] measured ratio arch on {primary_key}: {ra_f:.3f}x")
            print(f"[bench] measured ratio quant on {primary_key}: {rq_f:.3f}x")
            print(f"[bench] measured ratio e2e on {primary_key}: {re_f:.3f}x")
            msg = (
                f"[bench] mem128k e2e={re_f:.3f}x "
                f"(device_sig={device_sig(dev)}, torch={getattr(torch, '__version__', 'unknown')})"
            )
            print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

