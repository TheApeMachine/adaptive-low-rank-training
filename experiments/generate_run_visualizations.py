#!/usr/bin/env python3
"""
generate_run_visualizations.py

Generate the "paper-style" visualization pack for arbitrary v29 run directories.

This is meant for new ad-hoc runs like:
  - runs/m4max_big_baseline/
  - runs/m4max_big/
  - runs/m4max_big_gqa/

Unlike older scripts, this one understands v29 JSONL formats:
  - {"type":"eval", ...} for loss curves
  - {"type":"mem", "kv_ctx_bytes":..., "kv_128k_bytes":...} for measured KV cache memory

Outputs (prefixed by --tag) into --out (default: assets/):
  - {tag}_convergence.png
  - {tag}_early_convergence.png
  - {tag}_comparison_bar.png
  - {tag}_pareto.png
  - {tag}_kv_memory_128k.png
  - {tag}_summary.md

Preferred usage (multi-run):
  python3 generate_run_visualizations.py \
      --tag m4max_big \
      --run "baseline=runs/m4max_big_baseline" \
      --run "bottleneck=runs/m4max_big" \
      --run "gqa=runs/m4max_big_gqa" \
      --baseline-label baseline

Back-compat usage (two runs):
  python3 generate_run_visualizations.py \
      --tag m4max_big \
      --baseline runs/m4max_big_baseline \
      --candidate runs/m4max_big
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunSeries:
    name: str
    run_dir: Path

    eval_steps: List[int]
    train_losses: List[float]
    val_losses: List[float]

    attn_mode: str = "unknown"
    attn_dim: int = 0

    kv_ctx_bytes: int = 0
    kv_128k_bytes: int = 0

    @property
    def best_val(self) -> float:
        return min(self.val_losses) if self.val_losses else float("inf")

    @property
    def best_ppl(self) -> float:
        if not math.isfinite(self.best_val):
            return float("inf")
        return float(math.exp(self.best_val))

    @property
    def kv_ctx_mb(self) -> float:
        return float(self.kv_ctx_bytes) / (1024.0 * 1024.0)

    @property
    def kv_128k_gb(self) -> float:
        return float(self.kv_128k_bytes) / (1024.0 * 1024.0 * 1024.0)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def load_run(run_dir: Path, name: str) -> RunSeries:
    log_path = run_dir / "train.jsonl"
    rows = _read_jsonl(log_path)

    eval_steps: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    attn_mode = "unknown"
    attn_dim = 0
    kv_ctx_bytes = 0
    kv_128k_bytes = 0

    for r in rows:
        t = str(r.get("type", ""))
        if t == "meta":
            cfg = r.get("config", {}) if isinstance(r.get("config", {}), dict) else {}
            attn_mode = str(cfg.get("attn_mode", attn_mode))
            # Prefer config.attn_dim when present; decoupled uses sem+geo as "effective" d_attn.
            if attn_mode == "decoupled":
                try:
                    attn_dim = int(cfg.get("sem_dim", 0)) + int(cfg.get("geo_dim", 0))
                except Exception:
                    attn_dim = int(cfg.get("attn_dim", 0) or 0)
            else:
                try:
                    attn_dim = int(cfg.get("attn_dim", 0) or 0)
                except Exception:
                    attn_dim = 0
        elif t == "mem":
            try:
                kv_ctx_bytes = int(r.get("kv_ctx_bytes", kv_ctx_bytes) or kv_ctx_bytes)
            except Exception:
                pass
            try:
                kv_128k_bytes = int(r.get("kv_128k_bytes", kv_128k_bytes) or kv_128k_bytes)
            except Exception:
                pass
        elif t == "eval":
            if "step" in r and "val_loss" in r:
                try:
                    eval_steps.append(int(r["step"]))
                    val_losses.append(float(r["val_loss"]))
                    train_losses.append(float(r.get("train_loss", float("nan"))))
                except Exception:
                    continue

    return RunSeries(
        name=name,
        run_dir=run_dir,
        eval_steps=eval_steps,
        train_losses=train_losses,
        val_losses=val_losses,
        attn_mode=attn_mode,
        attn_dim=attn_dim,
        kv_ctx_bytes=kv_ctx_bytes,
        kv_128k_bytes=kv_128k_bytes,
    )


def _ensure_matplotlib() -> Tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to generate figures: {e}")
    return plt, None


def _parse_run_spec(s: str) -> Tuple[str, str]:
    # Expect "label=path"
    if "=" not in s:
        raise ValueError(f"--run expects 'label=path', got: {s!r}")
    label, path = s.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"--run expects 'label=path', got: {s!r}")
    return label, path


def _style_for(label: str, *, baseline_label: Optional[str]) -> Dict[str, Any]:
    # Consistent styling across plots
    if baseline_label is not None and label == baseline_label:
        return dict(color="#333333", linestyle="--", linewidth=2.75, alpha=0.95, zorder=5)
    # Non-baseline: let color be assigned by caller; keep solid and slightly transparent
    return dict(linestyle="-", linewidth=2.25, alpha=0.9, zorder=4)


def plot_convergence(
    *,
    runs: List[RunSeries],
    baseline_label: Optional[str],
    out_path: Path,
    title: str,
    early_max_step: Optional[int] = None,
) -> None:
    plt, _ = _ensure_matplotlib()
    plt.figure(figsize=(10, 6))

    def _clip(series: RunSeries) -> Tuple[List[int], List[float]]:
        if early_max_step is None:
            return series.eval_steps, series.val_losses
        xs: List[int] = []
        ys: List[float] = []
        for x, y in zip(series.eval_steps, series.val_losses):
            if x <= early_max_step:
                xs.append(x)
                ys.append(y)
        return xs, ys

    # palette for non-baseline curves
    palette = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4", "#F44336", "#795548", "#607D8B"]
    pi = 0
    for r in runs:
        xs, ys = _clip(r)
        if not xs:
            continue
        st = _style_for(r.name, baseline_label=baseline_label)
        if baseline_label is not None and r.name == baseline_label:
            color = st.get("color", "#333333")
        else:
            color = palette[pi % len(palette)]
            pi += 1
        plt.plot(
            xs,
            ys,
            label=f"{r.name} (best {r.best_val:.3f})",
            color=color,
            linestyle=st.get("linestyle", "-"),
            linewidth=float(st.get("linewidth", 2.0)),
            alpha=float(st.get("alpha", 0.9)),
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison_bar(*, runs: List[RunSeries], baseline_label: Optional[str], out_path: Path, title: str) -> None:
    plt, _ = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Sort by best_val (lower is better)
    runs_sorted = sorted([r for r in runs if math.isfinite(r.best_val)], key=lambda r: r.best_val)
    names = [r.name for r in runs_sorted]
    vals = [r.best_val for r in runs_sorted]
    colors: List[str] = []
    for r in runs_sorted:
        if baseline_label is not None and r.name == baseline_label:
            colors.append("#333333")
        else:
            colors.append("#4CAF50")

    bars = ax.barh(names, vals, color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Best Validation Loss", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(*, runs: List[RunSeries], baseline_label: Optional[str], out_path: Path, title: str) -> None:
    plt, _ = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    palette = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4", "#F44336", "#795548", "#607D8B"]
    pi = 0
    base: Optional[RunSeries] = None
    if baseline_label is not None:
        for r in runs:
            if r.name == baseline_label:
                base = r
                break

    for r in runs:
        if not math.isfinite(r.best_val) or int(r.attn_dim) <= 0:
            continue
        if baseline_label is not None and r.name == baseline_label:
            color = "#333333"
        else:
            color = palette[pi % len(palette)]
            pi += 1
        ax.scatter([r.attn_dim], [r.best_val], s=120, color=color, edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(r.name, (r.attn_dim, r.best_val), textcoords="offset points", xytext=(8, 5), fontsize=9, color=color)

        # If we have KV@128k measurements, annotate compression vs baseline for non-baseline points.
        if base is not None and base.kv_128k_bytes > 0 and r.kv_128k_bytes > 0 and r.name != base.name:
            ratio = float(base.kv_128k_bytes) / float(r.kv_128k_bytes)
            ax.annotate(
                f"{ratio:.2f}× KV@128k smaller",
                xy=(r.attn_dim, r.best_val),
                xytext=(20, -25),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75),
            )

    ax.set_xlabel("Attention Dimension (d_attn)", fontsize=12)
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Note: per-point KV@128k ratio callouts are handled above when baseline_label is provided.

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kv_memory_128k(*, runs: List[RunSeries], baseline_label: Optional[str], out_path: Path, title: str) -> None:
    plt, _ = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    runs_with_mem = [r for r in runs if r.kv_128k_bytes > 0]
    # Sort by memory (largest first)
    runs_with_mem.sort(key=lambda r: r.kv_128k_bytes, reverse=True)
    names = [r.name for r in runs_with_mem]
    vals = [r.kv_128k_gb for r in runs_with_mem]
    colors: List[str] = []
    for r in runs_with_mem:
        if baseline_label is not None and r.name == baseline_label:
            colors.append("#333333")
        else:
            colors.append("#4CAF50")

    bars = ax.barh(names, vals, color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("KV Cache @ 128k (GB, measured)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)

    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.02 if max(vals) > 0 else val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f} GB", va="center", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(*, runs: List[RunSeries], baseline_label: Optional[str], out_path: Path, tag: str) -> None:
    lines: List[str] = []
    lines.append(f"# {tag} — Visualization Summary")
    lines.append("")
    lines.append("| Run | attn_mode | d_attn | best_val | best_ppl | KV@ctx (MB) | KV@128k (GB) |")
    lines.append("|-----|----------:|-------:|---------:|---------:|------------:|------------:|")
    for r in runs:
        lines.append(
            f"| {r.name} | {r.attn_mode} | {r.attn_dim} | {r.best_val:.4f} | {r.best_ppl:.1f} | {r.kv_ctx_mb:.2f} | {r.kv_128k_gb:.2f} |"
        )
    lines.append("")
    base: Optional[RunSeries] = None
    if baseline_label is not None:
        for r in runs:
            if r.name == baseline_label:
                base = r
                break
    if base is not None:
        for r in runs:
            if r.name == base.name:
                continue
            if base.kv_128k_bytes > 0 and r.kv_128k_bytes > 0:
                ratio = float(base.kv_128k_bytes) / float(r.kv_128k_bytes)
                lines.append(f"- KV@128k ratio ({base.name}/{r.name}): **{ratio:.2f}×**")
            if math.isfinite(base.best_val) and math.isfinite(r.best_val):
                delta = r.best_val - base.best_val
                lines.append(f"- Best-val delta ({r.name} - {base.name}): **{delta:+.4f}**")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True, help="Filename prefix for outputs (e.g., m4max_big).")
    ap.add_argument("--run", type=str, action="append", default=None,
                    help="Add a run as 'label=path'. Can be passed multiple times.")
    ap.add_argument("--baseline-label", type=str, default=None,
                    help="Optional label (from --run) to render as baseline (dashed).")
    # Back-compat: two-run interface
    ap.add_argument("--baseline", type=str, default=None, help="(Back-compat) Baseline run dir (must contain train.jsonl).")
    ap.add_argument("--candidate", type=str, default=None, help="(Back-compat) Candidate run dir (must contain train.jsonl).")
    ap.add_argument("--out", type=str, default="assets", help="Output directory for generated figures.")
    ap.add_argument("--early-max-step", type=int, default=1000, help="Max step for early convergence plot.")
    ap.add_argument("--baseline-name", type=str, default="baseline", help="(Back-compat) Label for baseline series.")
    ap.add_argument("--candidate-name", type=str, default="candidate", help="(Back-compat) Label for candidate series.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    tag = str(args.tag)

    runs: List[RunSeries] = []
    if args.run:
        for spec in args.run:
            label, path = _parse_run_spec(str(spec))
            runs.append(load_run(Path(path), name=label))
        baseline_label = str(args.baseline_label) if args.baseline_label is not None else None
    else:
        # Back-compat: baseline/candidate
        if args.baseline is None or args.candidate is None:
            raise SystemExit("Provide either --run label=path (repeatable), or --baseline and --candidate.")
        runs = [
            load_run(Path(args.baseline), name=str(args.baseline_name)),
            load_run(Path(args.candidate), name=str(args.candidate_name)),
        ]
        baseline_label = str(args.baseline_name)

    plot_convergence(
        runs=runs,
        baseline_label=baseline_label,
        out_path=out_dir / f"{tag}_convergence.png",
        title=f"{tag}: Validation Loss Convergence",
        early_max_step=None,
    )
    plot_convergence(
        runs=runs,
        baseline_label=baseline_label,
        out_path=out_dir / f"{tag}_early_convergence.png",
        title=f"{tag}: Early Convergence (≤ {int(args.early_max_step)} steps)",
        early_max_step=int(args.early_max_step),
    )
    plot_comparison_bar(
        runs=runs,
        baseline_label=baseline_label,
        out_path=out_dir / f"{tag}_comparison_bar.png",
        title=f"{tag}: Best Validation Loss",
    )
    plot_pareto(
        runs=runs,
        baseline_label=baseline_label,
        out_path=out_dir / f"{tag}_pareto.png",
        title=f"{tag}: d_attn vs Best Val (and KV@128k)",
    )
    plot_kv_memory_128k(
        runs=runs,
        baseline_label=baseline_label,
        out_path=out_dir / f"{tag}_kv_memory_128k.png",
        title=f"{tag}: Measured KV Cache @ 128k",
    )
    write_summary_md(
        runs=runs,
        baseline_label=baseline_label,
        out_path=out_dir / f"{tag}_summary.md",
        tag=tag,
    )

    print(f"✓ Wrote figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


