#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slug_to_out_dir(run_id: str) -> Path:
    return _repo_root() / "runs" / run_id


def _run_cmd(cmd: List[str], *, cwd: Path, capture: bool) -> Tuple[int, str]:
    if capture:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        return int(p.returncode), out
    p = subprocess.run(cmd, cwd=str(cwd))
    return int(p.returncode), ""


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_print_config(output: str) -> dict:
    # production runner prints raw JSON for --print-config; find the first JSON object in output.
    s = output.strip()
    # Fast path: output is exactly json
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    # Fallback: scan for first '{' and last '}'.
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        return json.loads(s[i : j + 1])
    raise ValueError("Could not parse --print-config JSON from output.")


def _read_resolved_config(out_dir: Path) -> dict:
    p = out_dir / "resolved_config.json"
    if not p.exists():
        raise FileNotFoundError(f"resolved_config.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _validate_expected(cfg: dict, expected: dict) -> List[str]:
    errs: List[str] = []
    for k, v in expected.items():
        if k not in cfg:
            errs.append(f"missing key {k!r} in resolved config")
            continue
        if cfg.get(k) != v:
            errs.append(f"{k}: expected {v!r}, got {cfg.get(k)!r}")
    return errs


def _normalize_instrument_level(level: str) -> str:
    """Normalize instrumentation levels across runner versions.

    Some environments only accept: {auto, off, basic, rich}.
    Our manifest historically used: {auto, off, basic, medium, full, rich}.
    """
    s = str(level).strip().lower()
    if s in ("rich",):
        return "rich"
    if s in ("full",):
        return "rich"
    if s in ("medium",):
        return "basic"
    if s in ("basic", "off", "auto"):
        return s
    # Safe, high-signal default for paper runs.
    return "rich"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run paper experiments from a single manifest with invariant validation.")
    ap.add_argument("--manifest", type=str, default="paper_manifest.json")
    ap.add_argument("--group", type=str, default=None, help="Only run a specific manifest group by name.")
    ap.add_argument("--only", type=str, default=None, help="Only run IDs containing this substring.")
    ap.add_argument("--dry-run", action="store_true", help="Validate and write resolved configs, but do not train.")
    ap.add_argument("--continue-on-error", action="store_true", help="Continue to next run on failure.")
    ap.add_argument("--post-mem128k", action="store_true", help="After training, run 128k KV-cache memory benchmark and write mem128k.json into the run dir.")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable to use (default: current interpreter).")
    args = ap.parse_args(argv)

    root = _repo_root()
    manifest_path = (root / str(args.manifest)).resolve()
    m = _load_manifest(manifest_path)

    defaults: dict = dict(m.get("defaults", {}))

    groups = list(m.get("groups", []))
    if args.group:
        groups = [g for g in groups if str(g.get("name")) == str(args.group)]
        if not groups:
            raise SystemExit(f"No such group in manifest: {args.group!r}")

    for g in groups:
        gname = str(g.get("name"))
        data = str(g.get("data"))
        runs = list(g.get("runs", []))

        print(f"\n== Group: {gname} ({len(runs)} runs) ==")
        print(f"   data={data}")

        for r in runs:
            run_id = str(r["id"])
            if args.only and str(args.only) not in run_id:
                continue

            out_dir = _slug_to_out_dir(run_id)
            _ensure_dir(out_dir)

            # Build argv for `main.py` using the minimal CLI (performance/runtime knobs are auto-tuned).
            cmd: List[str] = [
                str(args.python),
                "main.py",
                "--mode",
                "train",
                "--exp",
                str(r["exp"]),
                "--data",
                data,
                "--out-dir",
                str(out_dir.relative_to(root)),
                "--seed",
                str(r.get("seed", 1337)),
            ]
            # Steps: omit unless explicitly specified in the manifest. Default is AUTO in the runtime.
            if "steps" in r:
                cmd += ["--steps", str(r.get("steps"))]

            # Optional overrides.
            # NOTE: We intentionally do NOT pass training-schedule flags here (eval cadence, save cadence,
            # seq_len curriculum, tokenizer, etc.) because some environments run an older minimal CLI that
            # requires `--expert` for those flags. Paper defaults should be expressed through presets and
            # runtime self-optimization, not CLI surface area.

            # Production policy: self-optimizations are ON by default.

            # Resume semantics (paper harness):
            # - Manifest runs often set `"resume": true` so the harness can be re-run safely.
            # - However the harness also creates the run directory up front, so blindly passing `--resume`
            #   would fail on first launch (no last.pt yet).
            # Therefore, treat `"resume": true` as "resume if checkpoint exists".
            resume_effective = "none"
            if bool(r.get("resume", False)):
                ckpt_last = out_dir / "last.pt"
                ckpt_best = out_dir / "best.pt"
                if ckpt_last.exists():
                    cmd += ["--resume"]
                    resume_effective = "last.pt"
                elif ckpt_best.exists():
                    cmd += ["--resume-path", str(ckpt_best.relative_to(root))]
                    resume_effective = "best.pt"
                else:
                    print(f"[INFO] {run_id}: resume requested but no checkpoint found (last.pt/best.pt). Starting fresh.")

            # W&B paper policy: opt-in via manifest, with connectivity-aware auto mode.
            wandb_project = "production"
            wandb_entity = "p4n0p71c0n"
            cmd += [
                "--wandb",
                "--wandb-project",
                wandb_project,
                "--wandb-entity",
                wandb_entity,
            ]

            # Persist the exact command.
            (out_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

            # Resolve config and validate invariants (always), without requiring a special CLI flag.
            # We do this by running a fast "config-only" launch (steps=0) which writes resolved_config.json.
            cmd_validate = [str(args.python), "main.py", "--mode", "train", "--exp", str(r["exp"]), "--data", data, "--out-dir", str(out_dir.relative_to(root)), "--seed", str(r.get("seed", 1337)), "--steps", "0"]
            rc, out = _run_cmd(cmd_validate, cwd=root, capture=True)
            if rc != 0:
                msg = f"[FAIL] config-validate for {run_id} (exit={rc})\n{out}"
                if not args.continue_on_error:
                    raise SystemExit(msg)
                print(msg)
                continue

            cfg = _read_resolved_config(out_dir)
            _write_json(
                out_dir / "resolved_run.json",
                {"id": run_id, "group": gname, "spec": r, "defaults": defaults, "resume_effective": resume_effective},
            )

            expected = dict(r.get("expected", {}))
            errs = _validate_expected(cfg, expected)
            if errs:
                msg = f"[FAIL] invariant validation for {run_id}:\n  - " + "\n  - ".join(errs)
                if not args.continue_on_error:
                    raise SystemExit(msg)
                print(msg)
                continue

            print(f"[OK] {run_id}: validated config invariants.")

            if args.dry_run:
                continue

            # Train (no capture; stream output).
            print(f"[RUN] {run_id} ...")
            rc2, _ = _run_cmd(cmd, cwd=root, capture=False)
            if rc2 != 0:
                msg = f"[FAIL] training for {run_id} (exit={rc2})"
                if not args.continue_on_error:
                    raise SystemExit(msg)
                print(msg)
                continue

            # Optional post-step: standardized memory benchmark at 128k.
            if args.post_mem128k:
                ckpt_best = out_dir / "best.pt"
                ckpt_last = out_dir / "last.pt"
                ckpt_use = ckpt_best if ckpt_best.exists() else ckpt_last
                if not ckpt_use.exists():
                    msg = f"[FAIL] post-mem128k: no checkpoint found for {run_id} (expected best.pt or last.pt)"
                    if not args.continue_on_error:
                        raise SystemExit(msg)
                    print(msg)
                    continue

                bench_cmd = [
                    str(args.python),
                    "-m",
                    "production.bench_end_to_end_memory",
                    "--ckpt",
                    str(ckpt_use.relative_to(root)),
                    "--context-len",
                    "131072",
                    "--out",
                    str((out_dir / "mem128k.json").relative_to(root)),
                    "--mode",
                    "auto",
                    "--kv-kind",
                    "fp16",
                ]
                # If this is a decoupled run, also compute decomposition + heterogeneous policy.
                exp_attn = str(r.get("expected", {}).get("attn_mode", ""))
                if exp_attn == "decoupled":
                    bench_cmd += [
                        "--decompose",
                        "--baseline-mode",
                        "standard",
                        "--policy",
                        "ksem=q4_0@32,kgeo=q8_0@32,v=q4_0@32,resid=128",
                    ]
                print(f"[POST] {run_id}: mem128k ...")
                rc3, _ = _run_cmd(bench_cmd, cwd=root, capture=False)
                if rc3 != 0:
                    msg = f"[FAIL] post-mem128k for {run_id} (exit={rc3})"
                    if not args.continue_on_error:
                        raise SystemExit(msg)
                    print(msg)
                    continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
