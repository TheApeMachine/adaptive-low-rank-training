#!/usr/bin/env python3
"""
run_paper_reviewable.py

One command to bring the repo into a *reviewable paper* state.

This is a thin orchestrator that:
  - (Optionally) runs missing v29 suite jobs for requested seeds
  - Regenerates suite summaries + per-seed plots for the paper
  - Regenerates rank-evidence figure
  - Regenerates long-context probes (RoPE extrapolation + needle probe)
  - Checks that the key artifacts referenced by paper.tex exist

By default this is a DRY RUN (prints commands). Add --run to execute.
By default we SKIP existing run directories to avoid re-running work.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def _q(x: object) -> str:
    return shlex.quote(str(x))


def _fmt(argv: List[str]) -> str:
    return " ".join(_q(a) for a in argv)


def _run(argv: List[str], *, cwd: Path, do_run: bool) -> None:
    print("\n" + "-" * 96)
    print(_fmt(argv))
    if do_run:
        subprocess.run(argv, cwd=str(cwd), check=True)


def _pick_latest_run_dir(runs_dir: Path, tag: str, variant: str, seed: int) -> Optional[Path]:
    prefix = f"{tag}_{variant}_seed{seed}"
    cands: List[Path] = []
    if not runs_dir.exists():
        return None
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == prefix or p.name.startswith(prefix + "_v"):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _parse_seeds(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _artifact_check(repo: Path, tag: str) -> Tuple[List[str], List[str]]:
    """
    Returns (present, missing) paths (strings) for key paper artifacts.
    Note: paper.tex uses \\IfFileExists for many of these; this is just a sanity checklist.
    """
    assets = repo / "assets"
    must = [
        assets / f"{tag}_suite_summary.md",
        assets / f"{tag}_suite_rows.tex",
        assets / f"{tag}_rank_evidence.png",
        assets / f"{tag}_baseline_rope_extrapolation.png",
        assets / f"{tag}_decoupled_rope_extrapolation.png",
        assets / f"{tag}_baseline_needle_haystack.png",
        assets / f"{tag}_decoupled_needle_haystack.png",
        assets / f"{tag}_baseline_needle_haystack_delta_nll.png",
        assets / f"{tag}_decoupled_needle_haystack_delta_nll.png",
        assets / f"{tag}_baseline_needle_haystack_log10_p_ratio.png",
        assets / f"{tag}_decoupled_needle_haystack_log10_p_ratio.png",
    ]
    present: List[str] = []
    missing: List[str] = []
    for p in must:
        (present if p.exists() else missing).append(str(p))
    return present, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="python3.12")
    ap.add_argument("--device", type=str, required=True, help="cpu|mps|cuda")
    ap.add_argument("--data-npy", type=str, default="fineweb_100m.npy")

    ap.add_argument("--tag", type=str, default="m4max", help="Main suite tag prefix (runs/<tag>_*).")
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--assets-dir", type=str, default="assets")

    # Main suite
    ap.add_argument("--suite-seeds", type=str, default="1337,1338,1339")
    ap.add_argument(
        "--if-exists",
        type=str,
        default="skip",
        choices=["skip", "suffix", "error"],
        help="What to do when a run dir already exists (default: skip).",
    )
    ap.add_argument("--run-suite", action="store_true", help="Run v29 suite training jobs (baseline/gqa/bottleneck/decoupled).")

    # Summaries + plots
    ap.add_argument("--summarize", action="store_true", default=True, help="Summarize suite + write assets/<tag>_suite_*. (default: on)")
    ap.add_argument("--no-summarize", action="store_false", dest="summarize")
    ap.add_argument("--plot-per-seed", action="store_true", default=True, help="Generate per-seed plots into assets/. (default: on)")
    ap.add_argument("--no-plot-per-seed", action="store_false", dest="plot_per_seed")

    # Rank evidence
    ap.add_argument("--rank-seed", type=int, default=1337)
    ap.add_argument("--rank-seq-len", type=int, default=1024)
    ap.add_argument("--rank-offset", type=int, default=0)
    ap.add_argument("--rank", action="store_true", default=True, help="Regenerate rank evidence figure (default: on)")
    ap.add_argument("--no-rank", action="store_false", dest="rank")

    # Probes
    ap.add_argument("--probe-seed", type=int, default=1337, help="Which seed's checkpoints to probe (baseline/decoupled).")
    ap.add_argument("--probe-offset", type=int, default=0)
    ap.add_argument("--probe-len", type=int, default=2_000_000)
    ap.add_argument("--needle-prompt-style", type=str, default="qa", choices=["qa", "key", "repeat"])
    ap.add_argument("--probes", action="store_true", default=True, help="Run RoPE extrapolation + needle probes (default: on)")
    ap.add_argument("--no-probes", action="store_false", dest="probes")

    ap.add_argument("--run", action="store_true", help="Execute commands (default prints only).")
    args = ap.parse_args()

    repo = Path(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = repo / str(args.runs_dir)

    # Parameter sanity checks (fail early)
    data = Path(str(args.data_npy))
    if not data.is_absolute():
        data = repo / data
    if not data.exists():
        raise SystemExit(f"[error] --data-npy not found: {data}")

    seeds = _parse_seeds(args.suite_seeds)
    if not seeds:
        raise SystemExit("[error] --suite-seeds must contain at least one seed (e.g. 1337,1338,1339)")

    # 1) Optional: run missing suite jobs
    if args.run_suite:
        _run(
            [
                str(args.python),
                "run_v29_suite.py",
                "--device",
                str(args.device),
                "--data",
                str(data),
                "--tag",
                str(args.tag),
                "--seeds",
                ",".join(map(str, seeds)),
                "--if-exists",
                str(args.if_exists),
                "--run",
            ]
            if args.run
            else [
                str(args.python),
                "run_v29_suite.py",
                "--device",
                str(args.device),
                "--data",
                str(data),
                "--tag",
                str(args.tag),
                "--seeds",
                ",".join(map(str, seeds)),
                "--if-exists",
                str(args.if_exists),
            ],
            cwd=repo,
            do_run=args.run,
        )

    # 2) Summarize suite + write assets + plots
    if args.summarize:
        cmd = [
            str(args.python),
            "summarize_v29_suite.py",
            "--tag",
            str(args.tag),
            "--assets",
            str(args.assets_dir),
            "--runs",
            str(args.runs_dir),
            "--write",
        ]
        if args.plot_per_seed:
            cmd.append("--plot")
        _run(cmd, cwd=repo, do_run=args.run)

    # Resolve ckpts for rank/probes
    baseline_dir = _pick_latest_run_dir(runs_dir, str(args.tag), "baseline", int(args.probe_seed))
    dec_dir = _pick_latest_run_dir(runs_dir, str(args.tag), "decoupled_48_96", int(args.probe_seed))
    if baseline_dir is None or dec_dir is None:
        print("\n" + "-" * 96)
        print("[warn] Could not find baseline/decoupled run dirs for the requested probe seed.")
        print("       Expected runs/<tag>_baseline_seed<seed> and runs/<tag>_decoupled_48_96_seed<seed> (or _v2 suffix).")
        baseline_ckpt = None
        dec_ckpt = None
    else:
        baseline_ckpt = baseline_dir / "best.pt"
        dec_ckpt = dec_dir / "best.pt"

    # 3) Rank evidence
    if args.rank:
        # For rank evidence we typically use the same seed as probes unless overridden.
        baseline_rank_dir = _pick_latest_run_dir(runs_dir, str(args.tag), "baseline", int(args.rank_seed))
        dec_rank_dir = _pick_latest_run_dir(runs_dir, str(args.tag), "decoupled_48_96", int(args.rank_seed))
        if baseline_rank_dir is None or dec_rank_dir is None:
            print("\n" + "-" * 96)
            print("[warn] Could not find baseline/decoupled run dirs for rank evidence; skipping.")
        else:
            baseline_rank_ckpt = baseline_rank_dir / "best.pt"
            dec_rank_ckpt = dec_rank_dir / "best.pt"
            _run(
                [
                    str(args.python),
                    "plot_rank_evidence.py",
                    "--ckpt",
                    f"baseline={str(baseline_rank_ckpt)}",
                    "--ckpt",
                    f"decoupled={str(dec_rank_ckpt)}",
                    "--data-npy",
                    str(data),
                    "--offset",
                    str(int(args.rank_offset)),
                    "--seq-len",
                    str(int(args.rank_seq_len)),
                    "--device",
                    str(args.device),
                    "--out",
                    str(Path(args.assets_dir) / f"{args.tag}_rank_evidence.png"),
                ],
                cwd=repo,
                do_run=args.run,
            )

    # 4) Probes (RoPE + needle)
    if args.probes and baseline_ckpt is not None and dec_ckpt is not None:
        probe_cmd = [
            str(args.python),
            "run_long_context_tests_v29.py",
            "--python",
            str(args.python),
            "--device",
            str(args.device),
            "--tag",
            str(args.tag),
            "--ckpt",
            f"baseline={str(baseline_ckpt)}",
            "--ckpt",
            f"decoupled={str(dec_ckpt)}",
            "--data-npy",
            str(data),
            "--offset",
            str(int(args.probe_offset)),
            "--len",
            str(int(args.probe_len)),
            "--needle-prompt-style",
            str(args.needle_prompt_style),
        ]
        if args.run:
            probe_cmd.append("--run")
        _run(probe_cmd, cwd=repo, do_run=args.run)

    # 5) Artifact checklist
    present, missing = _artifact_check(repo, str(args.tag))
    print("\n" + "=" * 96)
    print("Artifact checklist (paper reviewable state)")
    print("- present:", len(present))
    print("- missing:", len(missing))
    if missing:
        print("\nMissing:")
        for p in missing:
            print(" -", p)
        print("\nNotes:")
        print(" - Many appendix figures are wrapped in \\IfFileExists, so missing files won't break compilation,")
        print("   but for a *reviewable* bundle it's better to have all of them.")
        print(" - If the missing items are needle delta plots, rerun run_long_context_tests_v29.py once (it generates them).")
    else:
        print("All key artifacts present.")

    print("\nNext:")
    print(f"- Compile paper.tex (it will pick up assets/{args.tag}_suite_rows.tex, rank evidence, and appendix probes).")


if __name__ == "__main__":
    main()


