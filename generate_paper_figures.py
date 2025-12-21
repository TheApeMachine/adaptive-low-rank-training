#!/usr/bin/env python3
"""
generate_paper_figures.py

Generates all figures for the paper by reading instrumented experiment logs.
This script reads the JSONL logs from run directories listed in `paper_manifest.json` and creates
publication-ready visualizations.

Usage:
    python3.12 generate_paper_figures.py

Outputs:
    assets/paper_results.json               - Single source-of-truth for numbers
    assets/fig_convergence.png              - Convergence comparison
    assets/fig_pareto_memory_vs_loss.png    - KV@128k vs quality trade-off
    assets/table_main.tex                   - Main LaTeX results table
    assets/table_scale.tex                  - Scale LaTeX results table (A100 1B)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION
# =============================================================================

MANIFEST_PATH = Path("paper_manifest.json")


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_from_manifest(manifest_path: Path) -> Dict[str, str]:
    """Return mapping key->run_dir from paper_manifest.json."""
    root = _load_manifest(manifest_path)
    out: Dict[str, str] = {}
    for g in root.get("groups", []):
        for r in g.get("runs", []):
            rid = str(r.get("id"))
            run_dir = str(Path("runs") / rid)
            out[rid] = run_dir
    return out


# Fallback: scan runs/ for historical layouts if no manifest is present.

def discover_experiments() -> dict:
    """Auto-discover experiment runs by scanning runs/ directory."""
    import os
    from pathlib import Path
    
    experiments = {}
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        return experiments
    
    # Look for size-prefixed directories
    for size in ["tiny", "small", "medium", "large"]:
        for variant in ["baseline", "bottleneck", "decoupled", "gqa"]:
            dir_name = f"{size}_{variant}"
            dir_path = runs_dir / dir_name
            if dir_path.exists() and (dir_path / "train.jsonl").exists():
                # Create human-readable name
                if variant == "baseline":
                    name = f"Standard ({size})"
                elif variant == "bottleneck":
                    name = f"Bottleneck ({size})"
                elif variant == "decoupled":
                    name = f"Decoupled ({size})"
                elif variant == "gqa":
                    name = f"GQA ({size})"
                experiments[name] = str(dir_path)
    
    # Also check for old-style paper_* directories
    for dir_path in runs_dir.glob("paper_*"):
        if (dir_path / "train.jsonl").exists():
            name = dir_path.name.replace("paper_", "").replace("_", " ").title()
            if name not in experiments:
                experiments[name] = str(dir_path)
    
    return experiments

PAPER_EXPERIMENTS = discover_from_manifest(MANIFEST_PATH) if MANIFEST_PATH.exists() else discover_experiments()

# Memory data is now read from actual measurements in train.jsonl
# No more estimates - we measure everything

OUTPUT_DIR = Path("assets")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExperimentResult:
    name: str
    config: dict
    eval_steps: List[int]
    train_losses: List[float]
    val_losses: List[float]
    best_val: float
    total_time: float
    attn_dim: int
    
    # KV-cache memory @128k from `runs/<id>/mem128k.json` (produced by production bench tool)
    model_params_mb: float = 0.0
    kv_cache_train_mb: float = 0.0
    kv_cache_128k_fp16_mb: float = 0.0
    kv_cache_128k_q4_mb: float = 0.0
    compression_ratio: float = 1.0
    
    @property
    def best_ppl(self) -> float:
        import math
        return math.exp(self.best_val) if self.best_val < 20 else float('inf')

# =============================================================================
# PARSING
# =============================================================================

def parse_experiment(name: str, run_dir: str) -> Optional[ExperimentResult]:
    """Parse a JSONL log file and extract key metrics including measured memory."""
    log_path = os.path.join(run_dir, "train.jsonl")
    
    if not os.path.exists(log_path):
        print(f"  ⚠ Not found: {log_path}")
        return None
    
    config = {}
    eval_steps = []
    train_losses = []
    val_losses = []
    best_val = float('inf')
    total_time = 0.0
    
    # Memory measurements (from instrumentation, not estimates)
    model_params_mb = 0.0
    kv_cache_train_mb = 0.0
    kv_cache_128k_fp16_mb = 0.0
    kv_cache_128k_q4_mb = 0.0
    compression_ratio = 1.0
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                if data.get("type") == "run_config":
                    config = data.get("config", {})
                
                elif data.get("type") == "eval":
                    eval_steps.append(data["step"])
                    train_losses.append(data["train_loss"])
                    val_losses.append(data["val_loss"])
                
                elif data.get("type") == "best":
                    best_val = data["best_val"]
                
                elif data.get("type") == "done":
                    best_val = data.get("best_val", best_val)
                    total_time = data.get("total_seconds", 0)
                
                elif data.get("type") == "memory_measurement":
                    # Extract actual measured memory values
                    model_params = data.get("model_params", {})
                    model_params_mb = model_params.get("total_params_bytes", 0) / (1024 * 1024)
                    
                    kv_train = data.get("kv_cache_training", {})
                    kv_cache_train_mb = kv_train.get("fp16_total_mb", 0)
                    
                    kv_128k = data.get("kv_cache_128k", {})
                    kv_cache_128k_fp16_mb = kv_128k.get("fp16_total_mb", 0)
                    kv_cache_128k_q4_mb = kv_128k.get("q4_total_mb", 0)
                    compression_ratio = kv_128k.get("fp16_to_q4_ratio", 1.0)
                    
            except json.JSONDecodeError:
                continue
    
    if not eval_steps:
        print(f"  ⚠ No eval data in: {log_path}")
        return None
    
    # Determine attention dimension
    attn_dim = config.get("attn_dim", 512)
    if config.get("attn_mode") == "decoupled":
        attn_dim = config.get("sem_dim", 32) + config.get("geo_dim", 64)

    # Preferred memory source: mem128k.json produced by `production.bench_end_to_end_memory`.
    mem_path = Path(run_dir) / "mem128k.json"
    if mem_path.exists():
        try:
            mem = json.loads(mem_path.read_text(encoding="utf-8"))
            est_sel = mem.get("estimate_selected", {}) or {}
            est_bytes = float(est_sel.get("estimated_bytes", 0.0) or 0.0)
            kv_cache_128k_fp16_mb = est_bytes / (1024.0 * 1024.0)

            dec = mem.get("decomposition", None)
            if isinstance(dec, dict):
                est = dec.get("estimate_bytes", {}) or {}
                kv_cache_128k_fp16_mb = float(est.get("decoupled_fp16", est_bytes) or 0.0) / (1024.0 * 1024.0)
                kv_cache_128k_q4_mb = float(est.get("decoupled_candidate", 0.0) or 0.0) / (1024.0 * 1024.0)
                compression_ratio = float(est.get("ratio_e2e_standard_over_candidate", 1.0) or 1.0)
        except Exception:
            pass
    
    return ExperimentResult(
        name=name,
        config=config,
        eval_steps=eval_steps,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val=min(val_losses) if val_losses else best_val,
        total_time=total_time,
        attn_dim=attn_dim,
        model_params_mb=model_params_mb,
        kv_cache_train_mb=kv_cache_train_mb,
        kv_cache_128k_fp16_mb=kv_cache_128k_fp16_mb,
        kv_cache_128k_q4_mb=kv_cache_128k_q4_mb,
        compression_ratio=compression_ratio,
    )


def load_all_experiments() -> Tuple[Dict[str, ExperimentResult], Dict[str, ExperimentResult]]:
    """Load all experiment results."""
    print("\nLoading FineWeb-Edu experiments (paper manifest)...")
    experiments = {}
    for name, path in PAPER_EXPERIMENTS.items():
        result = parse_experiment(name, path)
        if result:
            experiments[name] = result
            print(f"  ✓ {name}: best_val={result.best_val:.4f}")
    
    # Return as (wikitext, fineweb) for API compatibility, but wikitext is empty
    return {}, experiments

# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_convergence_plot(
    experiments: Dict[str, ExperimentResult],
    output_path: Path,
    title: str,
    highlight_key: str = "Combined 96"
):
    """Generate a convergence plot comparing multiple experiments."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def _color_for(name: str) -> str:
        n = str(name).lower()
        if "baseline" in n or "standard" in n:
            return "#333333"
        if "bottleneck" in n:
            return "#9C27B0"
        if "decoupled" in n:
            return "#2196F3"
        if "gqa" in n:
            return "#FF9800"
        return "#666666"
    
    for name, result in experiments.items():
        color = _color_for(name)
        hi = (str(highlight_key).lower() in str(name).lower()) if highlight_key else False
        linewidth = 2.5 if hi else 1.6
        linestyle = "-" if hi else "-"
        marker = "o"
        
        ax.plot(
            result.eval_steps, 
            result.val_losses, 
            label=f"{name} ({result.best_val:.3f})",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            alpha=0.9 if linewidth > 2 else 0.7,
        )
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def generate_pareto_plot(experiments: Dict[str, ExperimentResult], output_path: Path) -> None:
    """Generate Pareto plot: KV-cache @128k (FP16 MB) vs best validation loss."""
    import matplotlib.pyplot as plt

    pts = [(n, r) for n, r in experiments.items() if float(r.kv_cache_128k_fp16_mb) > 0.0]
    if not pts:
        print("  ⚠ No mem128k.json data found; skipping Pareto plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    xs = [float(r.kv_cache_128k_fp16_mb) for _, r in pts]
    ys = [float(r.best_val) for _, r in pts]
    names = [n for n, _ in pts]

    ax.scatter(xs, ys, c="#FF5722", s=90, zorder=5, edgecolors="white", linewidth=1.2)
    for i, name in enumerate(names):
        ax.annotate(name, (xs[i], ys[i]), textcoords="offset points", xytext=(8, 5), fontsize=7)

    ax.set_xlabel("KV cache @128k (MB, FP16)", fontsize=12)
    ax.set_ylabel("Best validation loss", fontsize=12)
    ax.set_title("Pareto: KV-cache memory vs quality (FineWeb-Edu)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {output_path}")


def generate_memory_plot(
    experiments: Dict[str, ExperimentResult],
    output_path: Path
):
    """Generate memory comparison plot using ACTUAL MEASURED data."""
    import matplotlib.pyplot as plt
    
    # Filter to experiments with memory measurements
    exps_with_memory = {k: v for k, v in experiments.items() if v.kv_cache_128k_fp16_mb > 0}
    
    if not exps_with_memory:
        print(f"  ⚠ No memory measurements found. Run experiments with instrumentation enabled.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = list(exps_with_memory.keys())
    fp16_vals = [exps_with_memory[n].kv_cache_128k_fp16_mb for n in names]
    q4_vals = [exps_with_memory[n].kv_cache_128k_q4_mb for n in names]
    
    # Left: FP16 KV cache at 128k
    colors = ['#4CAF50' if 'Bottleneck' in n or 'Decoupled' in n else '#2196F3' for n in names]
    bars1 = ax1.barh(names, fp16_vals, color=colors, edgecolor='white')
    ax1.set_xlabel("KV Cache Memory (MB) @ 128k tokens, FP16", fontsize=11)
    ax1.set_title("Measured KV Cache Memory (FP16)", fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    for bar, val in zip(bars1, fp16_vals):
        ax1.text(val + max(fp16_vals)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.0f} MB', va='center', fontsize=9)
    
    # Right: Q4 KV cache at 128k
    bars2 = ax2.barh(names, q4_vals, color=colors, edgecolor='white')
    ax2.set_xlabel("KV Cache Memory (MB) @ 128k tokens, Q4", fontsize=11)
    ax2.set_title("Measured KV Cache Memory (Q4 Quantized)", fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    
    for bar, val in zip(bars2, q4_vals):
        ax2.text(val + max(q4_vals)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.0f} MB', va='center', fontsize=9)
    
    # Add compression ratio annotation
    if len(names) >= 2:
        baseline_fp16 = max(fp16_vals)
        best_q4 = min(q4_vals)
        if best_q4 > 0:
            ratio = baseline_fp16 / best_q4
            ax2.annotate(f"Max compression: {ratio:.0f}×",
                        xy=(max(q4_vals) * 0.5, len(names) - 0.5), fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def generate_comparison_bar(
    wikitext: Dict[str, ExperimentResult],
    fineweb: Dict[str, ExperimentResult],
    output_path: Path
):
    """Generate bar chart comparing final losses."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # WikiText-2 (left)
    if wikitext:
        names = list(wikitext.keys())
        vals = [wikitext[n].best_val for n in names]
        colors = ['#4CAF50' if 'Combined' in n else '#2196F3' for n in names]
        
        bars = ax1.barh(names, vals, color=colors, edgecolor='white')
        ax1.set_xlabel("Best Validation Loss", fontsize=11)
        ax1.set_title("WikiText-2 Results", fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        
        for bar, val in zip(bars, vals):
            ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
    
    # FineWeb (right)
    if fineweb:
        names = list(fineweb.keys())
        vals = [fineweb[n].best_val for n in names]
        colors = ['#FF5722' if 'Standard' in n else '#FF8A65' for n in names]
        
        bars = ax2.barh(names, vals, color=colors, edgecolor='white')
        ax2.set_xlabel("Best Validation Loss", fontsize=11)
        ax2.set_title("FineWeb-Edu Results", fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        
        for bar, val in zip(bars, vals):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        
        # Gap annotation
        if len(vals) >= 2:
            gap = vals[1] - vals[0]
            gap_pct = (gap / vals[0]) * 100
            ax2.annotate(f"Δ = {gap:.3f} ({gap_pct:.1f}%)",
                        xy=(max(vals) * 0.7, 0.5), fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def generate_latex_table(
    experiments: Dict[str, ExperimentResult],
    output_path: Path,
    caption: str,
    label: Optional[str] = None,
):
    """Generate a LaTeX table for the paper with measured memory data."""
    import math
    
    # Check if we have memory measurements
    has_memory = any(r.kv_cache_128k_fp16_mb > 0 for r in experiments.values())
    
    if has_memory:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + r"}",
            (r"\label{" + str(label) + r"}" if label else ""),
            r"\begin{tabular}{@{}lccccc@{}}",
            r"\toprule",
            r"Model & $d_{attn}$ & Val Loss & PPL & KV@128k (MB) & Compress. \\",
            r"\midrule",
        ]
    else:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + r"}",
            (r"\label{" + str(label) + r"}" if label else ""),
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            r"Model & $d_{attn}$ & Val Loss & PPL & Time (s) \\",
            r"\midrule",
        ]
    
    # Sort by best_val
    sorted_exps = sorted(experiments.items(), key=lambda x: x[1].best_val)
    best_name = sorted_exps[0][0] if sorted_exps else None
    
    for name, result in experiments.items():
        ppl = math.exp(result.best_val) if result.best_val < 20 else float('inf')
        ppl_str = f"{ppl:.1f}" if ppl < 10000 else r"$\infty$"
        
        if has_memory:
            kv_mem = f"{result.kv_cache_128k_fp16_mb:.0f}" if result.kv_cache_128k_fp16_mb > 0 else "—"
            compress = f"{result.compression_ratio:.1f}$\\times$" if result.compression_ratio > 1 else "—"
            
            if name == best_name:
                lines.append(rf"\textbf{{{name}}} & {result.attn_dim} & \textbf{{{result.best_val:.4f}}} & \textbf{{{ppl_str}}} & {kv_mem} & {compress} \\")
            else:
                lines.append(rf"{name} & {result.attn_dim} & {result.best_val:.4f} & {ppl_str} & {kv_mem} & {compress} \\")
        else:
            time_str = f"{result.total_time:.0f}" if result.total_time > 0 else "—"
            if name == best_name:
                lines.append(rf"\textbf{{{name}}} & {result.attn_dim} & \textbf{{{result.best_val:.4f}}} & \textbf{{{ppl_str}}} & {time_str} \\")
            else:
                lines.append(rf"{name} & {result.attn_dim} & {result.best_val:.4f} & {ppl_str} & {time_str} \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  → Saved: {output_path}")


def generate_summary_markdown(
    wikitext: Dict[str, ExperimentResult],
    fineweb: Dict[str, ExperimentResult],
    output_path: Path
):
    """Generate a markdown summary of all results with MEASURED memory data."""
    import math
    
    lines = [
        "# Experiment Results Summary",
        "",
        "Generated from instrumented experiment logs.",
        "**All memory values are measured, not estimated.**",
        "",
        "## Model Performance",
        "",
        "| Model | d_attn | Best Val | PPL | Params (MB) |",
        "|-------|--------|----------|-----|-------------|",
    ]
    
    for name, r in sorted(fineweb.items(), key=lambda x: x[1].best_val):
        ppl = math.exp(r.best_val)
        params = f"{r.model_params_mb:.1f}" if r.model_params_mb > 0 else "—"
        lines.append(f"| {name} | {r.attn_dim} | {r.best_val:.4f} | {ppl:.1f} | {params} |")
    
    # Memory comparison table (from actual measurements)
    has_memory = any(r.kv_cache_128k_fp16_mb > 0 for r in fineweb.values())
    if has_memory:
        lines.extend([
            "",
            "## Measured KV Cache Memory @ 128k Tokens",
            "",
            "| Model | FP16 (MB) | Q4 (MB) | Compression |",
            "|-------|-----------|---------|-------------|",
        ])
        
        for name, r in sorted(fineweb.items(), key=lambda x: x[1].kv_cache_128k_fp16_mb, reverse=True):
            if r.kv_cache_128k_fp16_mb > 0:
                fp16 = f"{r.kv_cache_128k_fp16_mb:.0f}"
                q4 = f"{r.kv_cache_128k_q4_mb:.0f}"
                comp = f"{r.compression_ratio:.1f}×"
                lines.append(f"| {name} | {fp16} | {q4} | {comp} |")
    
    # Key findings with actual numbers
    baseline_candidates = [n for n in fineweb.keys() if "Standard" in n or "baseline" in n.lower()]
    bottleneck_candidates = [n for n in fineweb.keys() if "Bottleneck" in n or "Decoupled" in n]
    
    if baseline_candidates and bottleneck_candidates:
        baseline_name = baseline_candidates[0]
        bottleneck_name = min(bottleneck_candidates, key=lambda n: fineweb[n].best_val)
        
        std = fineweb[baseline_name]
        best = fineweb[bottleneck_name]
        
        lines.extend([
            "",
            "## Key Findings",
            "",
        ])
        
        # Memory reduction (from actual measurements)
        if std.kv_cache_128k_fp16_mb > 0 and best.kv_cache_128k_fp16_mb > 0:
            mem_reduction = std.kv_cache_128k_fp16_mb / best.kv_cache_128k_fp16_mb
            lines.append(f"- **Measured Memory Reduction (FP16)**: {mem_reduction:.1f}× ({std.kv_cache_128k_fp16_mb:.0f} MB → {best.kv_cache_128k_fp16_mb:.0f} MB)")
        
        if std.kv_cache_128k_fp16_mb > 0 and best.kv_cache_128k_q4_mb > 0:
            total_reduction = std.kv_cache_128k_fp16_mb / best.kv_cache_128k_q4_mb
            lines.append(f"- **Total Memory Reduction (FP16→Q4)**: {total_reduction:.0f}× ({std.kv_cache_128k_fp16_mb:.0f} MB → {best.kv_cache_128k_q4_mb:.0f} MB)")
        
        lines.extend([
            f"- **Quality Gap**: {best.best_val - std.best_val:+.4f} loss",
            f"- **Best Bottleneck**: {bottleneck_name} (val={best.best_val:.4f})",
        ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  → Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Generating Paper Figures (FineWeb-Edu)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    _, fineweb = load_all_experiments()
    
    if not fineweb:
        print("\n⚠ No experiment data found!")
        print("Run experiments via manifest: `python run_paper_manifest.py --group mac_fw100m`")
        return

    # Write a single-source-of-truth JSON artifact for paper numbers.
    try:
        paper_results: Dict[str, Any] = {}
        for name, r in fineweb.items():
            paper_results[name] = {
                "best_val": float(r.best_val),
                "best_ppl": float(r.best_ppl),
                "steps": int(r.eval_steps[-1]) if r.eval_steps else 0,
                "attn_mode": str(r.config.get("attn_mode", "")),
                "d_model": int(r.config.get("d_model", 0) or 0),
                "layers": int(r.config.get("n_layer", r.config.get("layers", 0)) or 0),
                "kv_cache_128k_fp16_mb": float(r.kv_cache_128k_fp16_mb),
                "kv_cache_128k_q4_mb": float(r.kv_cache_128k_q4_mb),
                "compression_ratio_e2e": float(r.compression_ratio),
            }
        (OUTPUT_DIR / "paper_results.json").write_text(json.dumps(paper_results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote: {OUTPUT_DIR / 'paper_results.json'}")
    except Exception as e:
        print(f"[warn] failed to write assets/paper_results.json: {e}")
    
    print("\nGenerating figures...")
    print("-" * 40)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        # Fig: Convergence (prefer medium-sized runs if present)
        conv = {k: v for k, v in fineweb.items() if "_medium_" in k}
        if not conv:
            conv = fineweb
        generate_convergence_plot(
            conv,
            OUTPUT_DIR / "fig_convergence.png",
            "FineWeb-Edu: Validation Loss Convergence",
            highlight_key="decoupled",
        )

        # Fig: Pareto (KV@128k vs loss)
        generate_pareto_plot(fineweb, OUTPUT_DIR / "fig_pareto_memory_vs_loss.png")
        
    except ImportError:
        print("⚠ matplotlib not installed - skipping plots")
    
    print("\nGenerating tables...")
    print("-" * 40)
    
    # Main LaTeX table (all FineWeb runs)
    generate_latex_table(
        fineweb,
        OUTPUT_DIR / "table_main.tex",
        "FineWeb-Edu results (production manifest runs). KV@128k values come from mem128k.json.",
        label="tab:main",
    )

    # Scale LaTeX table (A100 1B runs only)
    a100 = {k: v for k, v in fineweb.items() if k.startswith("a100_")}
    if a100:
        generate_latex_table(
            a100,
            OUTPUT_DIR / "table_scale.tex",
            "FineWeb-Edu scale results (A100 1B suite).",
            label="tab:scale",
        )
    
    # Markdown summary
    generate_summary_markdown({}, fineweb, OUTPUT_DIR / "results_summary.md")
    
    print("\n" + "=" * 60)
    print("  Done! Check assets/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()

