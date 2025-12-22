#!/usr/bin/env python3
"""Run paper experiments from `paper_manifest.json` (train + validate + optional post-bench)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

from production.selfopt_cache import as_object_list, as_str_object_dict

__all__ = [
    "main",
    "_normalize_instrument_level",
    "_parse_print_config",
    "_validate_expected",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _json_loads_obj(text: str) -> object:
    # `json.loads` is typed as returning `Any` in stubs; isolate it behind an `object` boundary.
    return cast(object, json.loads(text))


def _load_manifest(path: Path) -> dict[str, object]:
    raw = as_str_object_dict(_json_loads_obj(path.read_text(encoding="utf-8")))
    if raw is None:
        raise ValueError("manifest root must be an object")
    return raw


def _as_list_of_dicts(x: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    x_list = as_object_list(x)
    if x_list is None:
        return out
    for it_obj in x_list:
        d = as_str_object_dict(it_obj)
        if d is not None:
            out.append(d)
    return out


def _as_str_keyed_dict(x: object) -> dict[str, object]:
    d = as_str_object_dict(x)
    return {} if d is None else d


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slug_to_out_dir(run_id: str) -> Path:
    return _repo_root() / "runs" / run_id


def _run_cmd(cmd: list[str], *, cwd: Path, capture: bool) -> tuple[int, str]:
    if capture:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        out = (p.stdout or "") + (p.stderr or "")
        return int(p.returncode), out
    p = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(p.returncode), ""


def _write_json(path: Path, obj: object) -> None:
    _ = path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_resolved_config(out_dir: Path) -> dict[str, object]:
    p = out_dir / "resolved_config.json"
    if not p.exists():
        raise FileNotFoundError(f"resolved_config.json not found: {p}")
    d = as_str_object_dict(_json_loads_obj(p.read_text(encoding="utf-8")))
    if d is None:
        raise ValueError("resolved_config.json must be an object")
    return d


def _validate_expected(cfg: dict[str, object], expected: dict[str, object]) -> list[str]:
    errs: list[str] = []
    for k, v in expected.items():
        if k not in cfg:
            errs.append(f"missing key {k!r} in resolved config")
            continue
        if cfg.get(k) != v:
            errs.append(f"{k}: expected {v!r}, got {cfg.get(k)!r}")
    return errs


def _parse_print_config(output: str) -> dict[str, object]:
    """
    Extract a JSON object from mixed stdout/stderr.

    The paper harness prints config as a JSON object (potentially surrounded by logs).
    We scan for the first valid JSON object and return it as a dict.
    """
    lines = str(output).splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "{":
            continue
        buf: list[str] = []
        for j in range(i, len(lines)):
            buf.append(lines[j])
            txt = "\n".join(buf)
            try:
                obj = _json_loads_obj(txt)
            except json.JSONDecodeError:
                continue
            d = as_str_object_dict(obj)
            if d is not None:
                return d
    raise ValueError("could not find a JSON object in output")


def _normalize_instrument_level(x: str) -> str:
    """
    Normalize user-facing instrument level.

    Accepted:
    - full/rich -> rich
    - medium/basic -> basic
    - off -> off
    - auto -> auto
    Unknown values default to rich.
    """
    s = str(x or "").strip().lower()
    if s in ("full", "rich"):
        return "rich"
    if s in ("medium", "basic"):
        return "basic"
    if s in ("off",):
        return "off"
    if s in ("auto",):
        return "auto"
    return "rich"



def main(argv: list[str] | None = None) -> int:
    class _Args(argparse.Namespace):
        manifest: str = "paper_manifest.json"
        group: str | None = None
        only: str | None = None
        dry_run: bool = False
        continue_on_error: bool = False
        post_mem128k: bool = False
        python: str = sys.executable

    ap = argparse.ArgumentParser(description="Run paper experiments from a single manifest with invariant validation.")
    _ = ap.add_argument("--manifest", type=str, default="paper_manifest.json")
    _ = ap.add_argument("--group", type=str, default=None, help="Only run a specific manifest group by name.")
    _ = ap.add_argument("--only", type=str, default=None, help="Only run IDs containing this substring.")
    _ = ap.add_argument("--dry-run", action="store_true", help="Validate and write resolved configs, but do not train.")
    _ = ap.add_argument("--continue-on-error", action="store_true", help="Continue to next run on failure.")
    _ = ap.add_argument(
        "--post-mem128k",
        action="store_true",
        help="After training, run 128k KV-cache memory benchmark and write mem128k.json into the run dir.",
    )
    _ = ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    args = ap.parse_args(argv, namespace=_Args())

    root = _repo_root()
    manifest_path = (root / str(args.manifest)).resolve()
    m = _load_manifest(manifest_path)

    defaults = _as_str_keyed_dict(m.get("defaults", {}))

    groups = _as_list_of_dicts(m.get("groups", []))
    if args.group:
        groups = [g for g in groups if str(g.get("name", "")) == str(args.group)]
        if not groups:
            raise SystemExit(f"No such group in manifest: {args.group!r}")

    mem_summaries: list[dict[str, object]] = []

    for g in groups:
        gname = str(g.get("name", ""))
        data = str(g.get("data", ""))
        runs = _as_list_of_dicts(g.get("runs", []))

        print(f"\n== Group: {gname} ({len(runs)} runs) ==")
        print(f"   data={data}")

        for r in runs:
            run_id = str(r["id"])
            if args.only and str(args.only) not in run_id:
                continue

            out_dir = _slug_to_out_dir(run_id)
            _ensure_dir(out_dir)

            # Build argv for `main.py` using the minimal CLI (performance/runtime knobs are auto-tuned).
            cmd: list[str] = [
                str(args.python),
                "main.py",
                "--mode",
                "train",
                "--exp",
                str(r["exp"]),
                # Optional size override (used by some mac_fw100m runs in the manifest).
                *([] if "size" not in r else ["--size", str(r.get("size"))]),
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

            # W&B: enabled by production defaults; no CLI flags needed for paper runs.

            # Persist the exact command.
            _ = (out_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

            # Resolve config and validate invariants (always), without requiring a special CLI flag.
            # We do this by running a fast "config-only" launch (steps=0) which writes resolved_config.json.
            cmd_validate: list[str] = [
                str(args.python),
                "main.py",
                "--mode",
                "train",
                "--exp",
                str(r["exp"]),
                *([] if "size" not in r else ["--size", str(r.get("size"))]),
                "--data",
                data,
                "--out-dir",
                str(out_dir.relative_to(root)),
                "--seed",
                str(r.get("seed", 1337)),
                "--steps",
                "0",
            ]
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

            expected = _as_str_keyed_dict(r.get("expected", {}))
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
                exp_attn = str(_as_str_keyed_dict(r.get("expected", {})).get("attn_mode", ""))
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
                # Best-effort: parse mem128k.json and emit a single harness-level summary row.
                try:
                    mem_path = out_dir / "mem128k.json"
                    mem_obj = as_str_object_dict(_json_loads_obj(mem_path.read_text(encoding="utf-8")))
                    if mem_obj is None:
                        raise ValueError("mem128k.json must be an object")
                    row: dict[str, object] = {
                        "id": run_id,
                        "group": gname,
                        "device_sig": mem_obj.get("device_sig"),
                        "context_len": mem_obj.get("context_len"),
                        "batch_size": mem_obj.get("batch_size"),
                        "attn_mode": _as_str_keyed_dict(mem_obj.get("model_cfg", {})).get("attn_mode"),
                    }
                    decomp = mem_obj.get("decomposition")
                    decomp_d = as_str_object_dict(decomp)
                    if decomp_d is not None:
                        meas_d = as_str_object_dict(decomp_d.get("measured"))
                        if meas_d is not None:
                            row["ratio_e2e_standard_over_candidate"] = meas_d.get("ratio_e2e_standard_over_candidate")
                    mem_summaries.append(row)
                    if "ratio_e2e_standard_over_candidate" in row:
                        ratio_obj = row.get("ratio_e2e_standard_over_candidate", float("nan"))
                        ratio = (
                            float(ratio_obj)
                            if isinstance(ratio_obj, (int, float))
                            else float("nan")
                        )
                        msg = f"[POST] {run_id}: mem128k e2e={ratio:.3f}x (device_sig={row.get('device_sig')})"
                        print(msg)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass

    if mem_summaries:
        try:
            out = root / "runs" / "mem128k_summary.json"
            _ensure_dir(out.parent)
            _write_json(out, mem_summaries)
            print(f"\n[OK] wrote {out.relative_to(root)} ({len(mem_summaries)} rows)")
        except (OSError, ValueError, TypeError):
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
