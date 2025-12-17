#!/usr/bin/env python3
"""
run_v29_suite.py

Experiment runner for the v29 implementation on FineWeb-Edu.

Goals:
  - Make it painless to run a reproducible suite: baseline / GQA / bottleneck / decoupled
  - Support multi-seed runs with consistent out-dir naming
  - Support a small set of ablation toggles that already exist in v29 (no code changes needed)
  - Print commands by default ("dry run"), with an option to execute sequentially

This script does NOT modify v29 model code; it just invokes it.

Example (dry run):
  python3.12 run_v29_suite.py --device mps --data fineweb_100m.npy --seeds 1337,1338

Example (execute):
  python3.12 run_v29_suite.py --device mps --data fineweb_100m.npy --seeds 1337 --run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


V29_SCRIPT = "v29_transformer_decoupled_bottleneck_instrumented.py"


def _csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _q(x: str) -> str:
    return shlex.quote(str(x))


def _fmt_cmd(argv: List[str]) -> str:
    return " ".join(_q(a) for a in argv)


@dataclass(frozen=True)
class Variant:
    name: str
    extra_args: Tuple[str, ...]


def build_variants() -> List[Variant]:
    """
    These variants match the intent of your existing m4max_big runs:
      - baseline: standard attention
      - gqa_kv2: grouped query attention with kv_head=2
      - bottleneck_144: bottleneck attention with d_attn=144
      - decoupled_48_96: decoupled attention with sem/geo = 48/96 (total 144)
    """
    return [
        Variant("baseline", ("--attn-mode", "standard")),
        Variant("gqa_kv2", ("--attn-mode", "gqa", "--kv-head", "2", "--attn-dim", "768")),
        Variant("bottleneck_144", ("--attn-mode", "bottleneck", "--attn-dim", "144", "--sem-dim", "48", "--geo-dim", "96")),
        Variant("decoupled_48_96", ("--attn-mode", "decoupled", "--attn-dim", "144", "--sem-dim", "48", "--geo-dim", "96")),
    ]


def build_ablations() -> Dict[str, Tuple[str, ...]]:
    """
    Ablations that are supported by v29 CLI (no code changes):
      - null token on/off
      - tie_qk on/off
      - rope on/off
      - learned temperature off
    """
    return {
        "null": ("--null-attn",),
        "no_null": ("--no-null-attn",),
        "tie_qk": ("--tie-qk",),
        "no_tie_qk": ("--no-tie-qk",),
        "no_rope": ("--no-rope",),
        "no_learned_temp": ("--no-learned-temp",),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="python3.12", help="Python executable to use.")
    ap.add_argument("--device", type=str, required=True, help="cpu|mps|cuda (or explicit torch device string).")
    ap.add_argument("--data", type=str, required=True, help="Path to FineWeb token dataset (e.g. fineweb_100m.npy).")
    ap.add_argument("--data-format", type=str, default="npy", choices=["auto", "text", "npy", "bin", "pt"])
    ap.add_argument("--vocab-size", type=int, default=50257)

    ap.add_argument("--run-root", type=str, default="runs", help="Root folder for output directories.")
    ap.add_argument("--tag", type=str, default="m4max", help="Prefix for run directory names (default matches runs/m4max_* style).")
    ap.add_argument("--seeds", type=str, default="1337", help="Comma-separated seeds.")

    # Training recipe (defaults match your m4max_big runs)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--d-ff", type=int, default=3072)
    ap.add_argument("--embed-dim", type=int, default=512)

    ap.add_argument("--optimizer", type=str, default="lion", choices=["adamw", "lion"])
    ap.add_argument("--lr", type=str, default="3e-4", help="Learning rate (passed through as string).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--train-seq-len", type=int, default=512)
    ap.add_argument("--seq-schedule", type=str, default="256@0,512@500,1024@2000")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--instrument", type=str, default="full", choices=["off", "basic", "medium", "full"])
    ap.add_argument("--analysis-every", type=int, default=100)
    ap.add_argument("--live", type=str, default="rich", choices=["auto", "off", "basic", "rich"])

    ap.add_argument("--param-dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # Which variants to run
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated variant names to run (baseline,gqa_kv2,bottleneck_144,decoupled_48_96). Default: all.")

    # Ablations to apply to *all* variants (optional)
    ap.add_argument("--ablations", type=str, default=None,
                    help="Comma-separated ablation keys to apply to all variants (e.g. 'no_rope,no_learned_temp').")

    # Execution
    ap.add_argument("--run", action="store_true", help="Execute commands (default prints only).")
    ap.add_argument("--continue-on-fail", action="store_true", help="Keep going if a run fails.")
    ap.add_argument("--if-exists", type=str, default="suffix", choices=["suffix", "skip", "error"],
                    help="What to do if the computed --out-dir already exists. "
                         "'suffix' appends _v2/_v3..., 'skip' skips, 'error' aborts. Default: suffix.")
    ap.add_argument("--cwd", type=str, default=None, help="Working directory (defaults to script directory).")

    args = ap.parse_args()

    seeds = _csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")

    variants = build_variants()
    if args.only:
        wanted = {x.strip() for x in str(args.only).split(",") if x.strip()}
        variants = [v for v in variants if v.name in wanted]
        if not variants:
            raise SystemExit(f"--only selected no variants. Available: {[v.name for v in build_variants()]}")

    ablation_args: Tuple[str, ...] = ()
    if args.ablations:
        ab_map = build_ablations()
        chosen = [x.strip() for x in str(args.ablations).split(",") if x.strip()]
        missing = [x for x in chosen if x not in ab_map]
        if missing:
            raise SystemExit(f"Unknown ablation keys: {missing}. Available: {sorted(ab_map.keys())}")
        ablation_args = tuple(a for k in chosen for a in ab_map[k])

    cwd = args.cwd or os.path.dirname(os.path.abspath(__file__))

    common = [
        "--mode", "train",
        "--device", str(args.device),
        "--data", str(args.data),
        "--data-format", str(args.data_format),
        "--vocab-size", str(int(args.vocab_size)),
        "--steps", str(int(args.steps)),
        "--d-model", str(int(args.d_model)),
        "--layers", str(int(args.layers)),
        "--n-head", str(int(args.n_head)),
        "--d-ff", str(int(args.d_ff)),
        "--embed-dim", str(int(args.embed_dim)),
        "--optimizer", str(args.optimizer),
        "--lr", str(args.lr),
        "--batch-size", str(int(args.batch_size)),
        "--grad-accum", str(int(args.grad_accum)),
        "--train-seq-len", str(int(args.train_seq_len)),
        "--seq-schedule", str(args.seq_schedule),
        "--eval-every", str(int(args.eval_every)),
        "--eval-iters", str(int(args.eval_iters)),
        "--log-every", str(int(args.log_every)),
        "--instrument", str(args.instrument),
        "--analysis-every", str(int(args.analysis_every)),
        "--live", str(args.live),
        "--param-dtype", str(args.param_dtype),
    ]
    if args.amp:
        common += ["--amp", "--amp-dtype", str(args.amp_dtype)]

    jobs: List[List[str]] = []
    for seed in seeds:
        for v in variants:
            base_out = os.path.join(str(args.run_root), f"{args.tag}_{v.name}_seed{seed}")
            out_dir = base_out
            if os.path.exists(out_dir):
                if str(args.if_exists) == "skip":
                    # still print the command later; mark as skipped by storing empty argv
                    out_dir = base_out
                elif str(args.if_exists) == "error":
                    raise SystemExit(f"Refusing to overwrite existing out dir: {out_dir} (use --if-exists suffix/skip)")
                else:
                    # suffix
                    k = 2
                    while os.path.exists(f"{base_out}_v{k}"):
                        k += 1
                    out_dir = f"{base_out}_v{k}"
            argv = [str(args.python), V29_SCRIPT] + common + [
                "--seed", str(int(seed)),
                "--out-dir", out_dir,
            ] + list(v.extra_args) + list(ablation_args)
            jobs.append(argv)

    print(f"cwd={cwd}")
    print(f"n_jobs={len(jobs)} variants={[v.name for v in variants]} seeds={seeds}")
    if ablation_args:
        print(f"ablations={args.ablations} -> {ablation_args}")

    for i, argv in enumerate(jobs, start=1):
        print("\n" + "-" * 80)
        print(f"[{i}/{len(jobs)}] {os.path.join(cwd, V29_SCRIPT)}")
        print(_fmt_cmd(argv))
        if not args.run:
            continue
        # Skip if requested and the base dir existed (we didn't change out_dir, so check again)
        if str(args.if_exists) == "skip" and os.path.exists(argv[argv.index("--out-dir") + 1]):
            print("[skip] out-dir exists")
            continue
        try:
            subprocess.run(argv, cwd=cwd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[error] run failed with exit code {e.returncode}")
            if not args.continue_on_fail:
                raise


if __name__ == "__main__":
    main()


