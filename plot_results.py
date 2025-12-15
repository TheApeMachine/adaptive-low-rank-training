#!/usr/bin/env python3
"""
plot_results.py
Generate convergence plots from training logs.

Usage:
    python3 plot_results.py

Before running:
    1. Save your terminal output to log files (e.g., runs/v21_baseline/log.txt)
    2. Update the LOGS dictionary below with your actual paths
"""
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

# ============================================================================
# CONFIG: Add your log files here
# Format: "Label": "path/to/logfile.log"
# ============================================================================
LOGS = {
    # WikiText-2 Experiments (Main Results)
    "⭐ Combined 96": "runs/v21_combined_baseline_96/train.log",
    "GQA (kv=2)": "runs/v21_gqa_kv2_parammatch/train.log",
    "Small Model (d=128)": "runs/v21_small_d128_standard/train.log",
    "Decoupled 1024 ctx": "runs/v21_decoupled_sem32_geo64_block1024/train.log",
    "Decoupled 2048 ctx": "runs/v21_decoupled_sem32_geo64_block2048/train.log",
    # FineWeb Experiments (add when ready)
    # "Baseline (512)": "runs/v21_fineweb_baseline/train.log",
    # "Decoupled (32/64)": "runs/v21_fineweb_decoupled/train.log",
}

# Output paths
OUTPUT_CONVERGENCE = "assets/convergence_plot.png"
OUTPUT_PARETO = "assets/pareto_curve.png"


def parse_log(filepath: str) -> tuple[list[int], list[float]]:
    """
    Parse training log to extract eval steps and validation losses.
    
    Expected log format:
        == eval step 200 | train 5.3370 | val 5.8770 | val_ppl 356.74 | 203.3s
    """
    steps = []
    val_losses = []
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return [], []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Regex to find "== eval step X | ... val Y"
            match = re.search(r"== eval step (\d+) .* val (\d+\.\d+)", line)
            if match:
                steps.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
    
    return steps, val_losses


def plot_convergence():
    """Generate the convergence plot (Step vs Val Loss)."""
    plt.figure(figsize=(10, 6))
    
    found_any = False
    for label, path in LOGS.items():
        steps, losses = parse_log(path)
        if steps:
            plt.plot(steps, losses, label=label, marker='o', markersize=3, linewidth=2)
            found_any = True
            print(f"  ✓ {label}: {len(steps)} eval points, final loss = {losses[-1]:.4f}")
    
    if not found_any:
        print("No logs found! Update the LOGS dictionary with your actual paths.")
        return
    
    plt.title("Validation Loss: Baseline vs. Bottleneck Architectures", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss (Lower is Better)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Ensure output directory exists
    Path(OUTPUT_CONVERGENCE).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_CONVERGENCE, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved {OUTPUT_CONVERGENCE}")


def plot_pareto():
    """
    Generate the Pareto curve (Attention Dim vs Final Val Loss).
    
    This requires running experiments with different attn_dim values
    and recording the final validation loss for each.
    """
    # Manual data entry from your experiments
    # Format: (attn_dim, final_val_loss, label)
    PARETO_DATA = [
        (512, 5.37, "Standard 512"),
        (96, 5.33, "⭐ Combined 96"),   # THE WINNER! Beats full baseline
        (128, 5.48, "Bottleneck 128"),
        (96, 5.59, "Decoupled 32/64"),  # sem=32 + geo=64 = 96 effective dims
        (128, 5.63, "GQA kv=2"),        # GQA with attn_dim=128
        (128, 5.74, "Small d=128"),     # Full small model (proves wide residual matters)
    ]
    
    if not PARETO_DATA:
        print("No Pareto data configured. Update PARETO_DATA with your results.")
        return
    
    dims = [d[0] for d in PARETO_DATA]
    losses = [d[1] for d in PARETO_DATA]
    labels = [d[2] for d in PARETO_DATA]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, losses, 'bo-', linewidth=2, markersize=10)
    
    # Annotate points
    for i, label in enumerate(labels):
        plt.annotate(label, (dims[i], losses[i]), 
                     textcoords="offset points", xytext=(0, 10), 
                     ha='center', fontsize=9)
    
    # Highlight the "flat" region
    plt.axhspan(5.35, 5.65, alpha=0.2, color='green', label='Acceptable Range (<5% gap)')
    
    plt.title("Pareto Curve: Attention Dimension vs. Perplexity", fontsize=14)
    plt.xlabel("Attention Dimension (d_attn)", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks([32, 64, 128, 256, 512], ['32', '64', '128', '256', '512'])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    Path(OUTPUT_PARETO).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PARETO, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {OUTPUT_PARETO}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Research Plots")
    print("=" * 60)
    
    print("\n1. Convergence Plot (Step vs Val Loss)")
    print("-" * 40)
    plot_convergence()
    
    print("\n2. Pareto Curve (Attention Dim vs Loss)")
    print("-" * 40)
    plot_pareto()
    
    print("\n" + "=" * 60)
    print("Done! Check the assets/ folder for outputs.")

