#!/usr/bin/env python3
"""
compare_kv_cache_policies.py

Small, practical harness to compare KV-cache settings on an existing v29 checkpoint:
  - Runs both FP16 caches and a heterogeneous decoupled policy (e.g. K_sem=q4, K_geo=q8, V=q4)
  - Runs both greedy (temperature=0) and sampling (temperature>0) decodes
  - Prints tok/s, token IDs, and decoded text (tiktoken GPT-2)
  - Reports the first divergence point between two generations

This does NOT modify v29 code; it imports the model and calls GPT.generate().

Example:
  python3.12 compare_kv_cache_policies.py \
    --ckpt runs/m4max_big_decoupled_qsemq4/best.pt \
    --device mps \
    --prompt-ids "0" \
    --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _parse_int_list(s: str) -> List[int]:
    parts: List[int] = []
    for tok in str(s).strip().split():
        if not tok:
            continue
        parts.append(int(tok))
    return parts


def _decode_gpt2(ids: List[int]) -> str:
    try:
        import tiktoken  # type: ignore
    except Exception:
        return "<tiktoken not installed: showing token IDs only>"
    enc = tiktoken.get_encoding("gpt2")
    return enc.decode(ids)


def _first_divergence(a: List[int], b: List[int]) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    if len(a) != len(b):
        return n
    return None


@dataclass
class RunResult:
    name: str
    tok_s: float
    ids: List[int]
    text: str


def _run_generate(
    *,
    name: str,
    model: Any,
    prompt: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    seed: int,
    gen_kwargs: Dict[str, Any],
) -> RunResult:
    torch.manual_seed(int(seed))
    _sync_if_needed(device)
    t0 = time.time()
    out = model.generate(
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=None if top_k is None else int(top_k),
        **gen_kwargs,
    )
    _sync_if_needed(device)
    dt = max(1e-9, time.time() - t0)
    # out includes prompt + generated
    ids = [int(x) for x in out[0].detach().cpu().tolist()]
    txt = _decode_gpt2(ids)
    tok_s = float(max_new_tokens) / float(dt)
    return RunResult(
        name=str(name),
        tok_s=tok_s,
        ids=ids,
        text=txt,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default=None, help="e.g. cpu|mps|cuda. Default auto-picks mps/cuda/cpu.")
    ap.add_argument("--prompt-ids", type=str, default="0", help="Whitespace-separated integer token IDs.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--kv-decode-block", type=int, default=1024)
    ap.add_argument("--kv-fused", type=str, default="none", choices=["none", "auto", "triton1pass", "triton2pass"])

    # Baseline controls
    ap.add_argument("--fp16-residual", type=int, default=0)

    # Heterogeneous decoupled policy controls (the "aggressive semantic compression" experiment)
    ap.add_argument("--qsem", type=str, default="q4_0", choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"])
    ap.add_argument("--qgeo", type=str, default="q8_0", choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"])
    ap.add_argument("--qv", type=str, default="q4_0", choices=["fp16", "fp32", "q8_0", "q4_0", "nf4"])
    ap.add_argument("--qblock", type=int, default=32)
    ap.add_argument("--qresidual", type=int, default=128)

    # Optional: numeric quality guard on a calibration slice (teacher-forced, cache-aware).
    ap.add_argument("--calib-npy", type=str, default=None,
                    help="Optional .npy token file for quality guard (e.g. fineweb_100m.npy). Uses np.load(mmap) and slices a window.")
    ap.add_argument("--calib-offset", type=int, default=0, help="Start offset into --calib-npy (tokens).")
    ap.add_argument("--calib-len", type=int, default=8192, help="Number of tokens to use from --calib-npy.")
    ap.add_argument("--calib-prefill", type=int, default=128, help="Prefill length for teacher-forced cache-aware metrics.")
    ap.add_argument("--calib-decode-steps", type=int, default=32, help="Decode steps for teacher-forced cache-aware metrics.")
    ap.add_argument("--tol-max-abs-logit", type=float, default=1.0,
                    help="Safety fuse: fail if max|Δlogit| exceeds this. Note: this is intentionally a loose canary; ΔNLL/ppl_ratio are the quality gates.")
    ap.add_argument("--disable-max-logit-fuse", action="store_true",
                    help="Disable the max|Δlogit| fuse and gate only on ΔNLL/ppl_ratio (and optional KL).")
    ap.add_argument("--tol-delta-nll", type=float, default=0.02, help="Fail if ΔNLL (nats/token) exceeds this.")
    ap.add_argument("--tol-ppl-ratio", type=float, default=1.02, help="Fail if ppl_ratio exceeds this (exp(ΔNLL)).")
    ap.add_argument("--tol-kl", type=float, default=None, help="Optional fail if KL(p_base||p_cand) exceeds this.")
    ap.add_argument("--compute-kl", action="store_true", help="Compute KL(p_base||p_cand) even if --tol-kl unset (slower).")
    args = ap.parse_args()

    # Import model code (no edits to v29)
    import v29_transformer_decoupled_bottleneck_instrumented as v29  # type: ignore

    if args.device:
        device = torch.device(str(args.device))
    else:
        device = v29.pick_device(None)

    ckpt = torch.load(str(args.ckpt), map_location=device)
    cfg = v29.ModelConfig(**ckpt["config"])
    model = v29.GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt_ids = _parse_int_list(args.prompt_ids)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Two KV-cache configurations
    fp16_name = "fp16"
    fp16_kwargs: Dict[str, Any] = dict(
        kv_cache="fp16",
        kv_residual=int(args.fp16_residual),
        kv_decode_block=int(args.kv_decode_block),
        kv_fused=str(args.kv_fused),
    )

    hetero_name = f"decoupled_qsem={args.qsem}_qgeo={args.qgeo}_qv={args.qv}"
    hetero_kwargs: Dict[str, Any] = dict(
        kv_cache="fp16",  # keep default fp16, override per-tensor
        kv_cache_k_sem=str(args.qsem),
        kv_cache_k_geo=str(args.qgeo),
        kv_cache_v=str(args.qv),
        kv_qblock_k_sem=int(args.qblock),
        kv_qblock_k_geo=int(args.qblock),
        kv_qblock_v=int(args.qblock),
        kv_residual=int(args.qresidual),
        kv_decode_block=int(args.kv_decode_block),
        kv_fused=str(args.kv_fused),
    )

    # Run greedy (deterministic)
    r_fp16_greedy = _run_generate(
        name=fp16_name + "_greedy",
        model=model,
        prompt=prompt,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        temperature=0.0,
        top_k=None,
        seed=int(args.seed),
        gen_kwargs=fp16_kwargs,
    )
    r_het_greedy = _run_generate(
        name=hetero_name + "_greedy",
        model=model,
        prompt=prompt,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        temperature=0.0,
        top_k=None,
        seed=int(args.seed),
        gen_kwargs=hetero_kwargs,
    )

    # Run sampling (reproducible via seed, but still sensitive to logit drift)
    r_fp16_samp = _run_generate(
        name=fp16_name + "_sampling",
        model=model,
        prompt=prompt,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        temperature=1.0,
        top_k=args.top_k,
        seed=int(args.seed),
        gen_kwargs=fp16_kwargs,
    )
    r_het_samp = _run_generate(
        name=hetero_name + "_sampling",
        model=model,
        prompt=prompt,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        temperature=1.0,
        top_k=args.top_k,
        seed=int(args.seed),
        gen_kwargs=hetero_kwargs,
    )

    def _print_run(r: RunResult) -> None:
        print("\n" + "=" * 80)
        print(f"{r.name}")
        print(f"tok/s: {r.tok_s:.2f}")
        print("- token_ids:")
        print(r.ids)
        print("- text:")
        print(r.text)

    print(f"device={device} ckpt={args.ckpt}")
    print(f"prompt_ids={prompt_ids}")
    print(f"max_new_tokens={args.max_new_tokens}")

    print("\n### GREEDY (temperature=0)")
    _print_run(r_fp16_greedy)
    _print_run(r_het_greedy)
    i = _first_divergence(r_fp16_greedy.ids, r_het_greedy.ids)
    print(f"\nfirst_divergence_idx(greedy): {i}")

    print("\n### SAMPLING (temperature=1)")
    _print_run(r_fp16_samp)
    _print_run(r_het_samp)
    j = _first_divergence(r_fp16_samp.ids, r_het_samp.ids)
    print(f"\nfirst_divergence_idx(sampling): {j}")

    # Optional: cache-aware quality guard evaluation (decoupled only).
    if args.calib_npy:
        if cfg.attn_mode != "decoupled":
            print("\n[warn] --calib-npy quality guard is only meaningful for decoupled checkpoints; skipping.")
            return

        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise RuntimeError(f"numpy is required for --calib-npy: {e}")

        arr = np.load(str(args.calib_npy), mmap_mode="r")
        off = int(max(0, args.calib_offset))
        n = int(max(0, args.calib_len))
        sl = np.asarray(arr[off:off + n], dtype=np.int64).reshape(-1)
        calib = torch.from_numpy(sl).to(device=device, dtype=torch.long).unsqueeze(0).contiguous()

        policy = v29.KVCachePolicy(
            k_sem_kind=str(args.qsem),
            k_geo_kind=str(args.qgeo),
            v_kind=str(args.qv),
            k_sem_qblock=int(args.qblock),
            k_geo_qblock=int(args.qblock),
            v_qblock=int(args.qblock),
            residual_len=int(args.qresidual),
        )

        compute_kl = bool(args.compute_kl or (args.tol_kl is not None))
        qm = model._policy_quality_metrics_decoupled(  # type: ignore[attr-defined]
            calib,
            policy=policy,
            prefill=int(args.calib_prefill),
            decode_steps=int(args.calib_decode_steps),
            kv_decode_block=int(args.kv_decode_block),
            compute_kl=compute_kl,
        )

        max_abs = float(qm.get("max_abs_logit", float("inf")))
        dnll = float(qm.get("delta_nll", float("nan")))
        pr = float(qm.get("ppl_ratio", float("nan")))
        klv = float(qm.get("kl_base_cand", float("nan")))

        reasons: List[str] = []
        if (not bool(args.disable_max_logit_fuse)) and max_abs > float(args.tol_max_abs_logit):
            reasons.append(f"max|Δlogit|={max_abs:.4g} > {float(args.tol_max_abs_logit):.4g} (fuse)")
        if dnll == dnll and dnll > float(args.tol_delta_nll):  # dnll==dnll filters NaN
            reasons.append(f"ΔNLL={dnll:.4g} > {float(args.tol_delta_nll):.4g} nats/tok")
        if pr == pr and pr > float(args.tol_ppl_ratio):
            reasons.append(f"ppl_ratio={pr:.4g} > {float(args.tol_ppl_ratio):.4g}")
        if args.tol_kl is not None and klv == klv and klv > float(args.tol_kl):
            reasons.append(f"KL(base||cand)={klv:.4g} > {float(args.tol_kl):.4g} nats/tok")

        print("\n### QUALITY GUARD (teacher-forced, cache-aware)")
        print(f"calib_npy={args.calib_npy} offset={off} len={n} prefill={int(args.calib_prefill)} decode_steps={int(args.calib_decode_steps)}")
        print(f"policy={policy.short()}")
        print(f"max_abs_logit={max_abs:.6g}")
        print(f"delta_nll={dnll:.6g}  (nats/token)")
        print(f"ppl_ratio={pr:.6g}")
        if compute_kl:
            print(f"kl_base_cand={klv:.6g}  (nats/token)")
        if reasons:
            print("QUALITY_GUARD: FAIL")
            for r in reasons:
                print(f"- {r}")
        else:
            print("QUALITY_GUARD: PASS")


if __name__ == "__main__":
    main()


