#!/usr/bin/env python3
"""
plot_memory.py
Generate the "Money Shot" memory footprint comparison chart.

This is the viral image for your paper/blog.
Shows KV cache memory required for 128k context across different architectures.

Usage:
    python3 plot_memory.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIG: Model configuration (Llama-2-7B scale for dramatic effect)
# ============================================================================
CONTEXT_LEN = 128 * 1024  # 128k context
LAYERS = 32
HEADS = 32
D_MODEL = 4096  # Standard Llama-2-7B dimension
BATCH_SIZE = 1

# Output path
OUTPUT_PATH = "assets/memory_footprint.png"


def calc_mem_gb(d_kv: int, quantization_bits: int = 16, n_layers: int = LAYERS) -> float:
    """
    Calculate KV cache memory in GB.
    
    Formula: 2 * layers * d_kv * context_len * (bits/8) / 1024^3
             ^-- K and V
    """
    total_elements = 2 * n_layers * d_kv * CONTEXT_LEN * BATCH_SIZE
    total_bytes = total_elements * (quantization_bits / 8)
    return total_bytes / (1024 ** 3)


def main():
    print("=" * 60)
    print("Generating Memory Footprint Chart")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Context Length: {CONTEXT_LEN:,} tokens (128k)")
    print(f"  Layers: {LAYERS}")
    print(f"  d_model: {D_MODEL}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    # Calculate memory for each architecture
    # =========================================================================
    
    # 1. Standard Transformer (d_kv = d_model = 4096, FP16)
    mem_std = calc_mem_gb(D_MODEL, 16)
    
    # 2. GQA with 8 groups (4x reduction in KV heads)
    mem_gqa = calc_mem_gb(D_MODEL // 4, 16)
    
    # 3. MLA (DeepSeek-V2 style, 93% reduction)
    mem_mla = calc_mem_gb(D_MODEL, 16) * 0.067  # 93.3% reduction
    
    # 4. Bottleneck (d_attn = 128, FP16)
    mem_bottleneck_fp16 = calc_mem_gb(128, 16)
    
    # 5. Bottleneck (d_attn = 128, Q4_0)
    mem_bottleneck_q4 = calc_mem_gb(128, 4)
    
    # 6. Decoupled Bottleneck (sem=32 + geo=64 = 96 dims, Q4_0)
    # The "Ultimate" configuration
    mem_decoupled_q4 = calc_mem_gb(96, 4)
    
    print(f"\nMemory Requirements:")
    print(f"  Standard (FP16):     {mem_std:.2f} GB")
    print(f"  GQA 8x (FP16):       {mem_gqa:.2f} GB")
    print(f"  MLA (FP16):          {mem_mla:.2f} GB")
    print(f"  Bottleneck (FP16):   {mem_bottleneck_fp16:.2f} GB")
    print(f"  Bottleneck (Q4):     {mem_bottleneck_q4:.2f} GB")
    print(f"  Decoupled (Q4):      {mem_decoupled_q4:.2f} GB")
    
    # Calculate compression ratios
    print(f"\nCompression vs Standard:")
    print(f"  GQA:         {mem_std/mem_gqa:.1f}x")
    print(f"  MLA:         {mem_std/mem_mla:.1f}x")
    print(f"  Bottleneck:  {mem_std/mem_bottleneck_q4:.1f}x")
    print(f"  Decoupled:   {mem_std/mem_decoupled_q4:.1f}x  ← The Winner")
    
    # Plotting
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    labels = [
        'Standard\n(FP16)', 
        'GQA 8×\n(FP16)', 
        'MLA\n(FP16)',
        'Bottleneck\n(FP16)',
        'Bottleneck\n(Q4)',
        'Decoupled\n(Q4)'
    ]
    values = [
        mem_std, 
        mem_gqa, 
        mem_mla,
        mem_bottleneck_fp16,
        mem_bottleneck_q4, 
        mem_decoupled_q4
    ]
    
    # Color scheme: red (bad) -> yellow -> green (good)
    colors = ['#ff6b6b', '#ffa94d', '#69db7c', '#38d9a9', '#20c997', '#12b886']
    
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add text labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        # Position text inside bar for tall bars, above for short bars
        if height > 5:
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{val:.1f} GB',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.2f} GB',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add compression ratio annotations
    for i, (bar, val) in enumerate(zip(bars, values)):
        if i > 0:  # Skip standard
            ratio = mem_std / val
            ax.text(bar.get_x() + bar.get_width()/2., -2,
                    f'{ratio:.0f}×',
                    ha='center', va='top', fontsize=10, 
                    color='#2d6a4f', fontweight='bold')
    
    ax.axhline(y=24, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(5.6, 25, 'RTX 4090 (24 GB)', fontsize=9, color='red', ha='right')
    
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(5.6, 81, 'A100 (80 GB)', fontsize=9, color='orange', ha='right')
    
    plt.title('KV Cache Memory for 128k Context (Llama-7B Scale)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('VRAM Required (GB)', fontsize=12)
    plt.xlabel('Architecture', fontsize=12)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=-4, top=max(values) * 1.1)
    
    # Add subtitle with key insight
    fig.text(0.5, 0.02, 
             'Decoupled Bottleneck + Q4 achieves 64× memory reduction vs. Standard Transformer',
             ha='center', fontsize=11, style='italic', color='#2d6a4f')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved {OUTPUT_PATH}")
    
    # Also save a version with dark background for presentations
    plt.style.use('dark_background')
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bars2 = ax2.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars2, values):
        height = bar.get_height()
        if height > 5:
            ax2.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{val:.1f} GB', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='black')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.2f} GB', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='white')
    
    ax2.set_title('KV Cache Memory for 128k Context', fontsize=16, fontweight='bold')
    ax2.set_ylabel('VRAM Required (GB)', fontsize=12)
    ax2.set_ylim(bottom=0)
    
    dark_path = OUTPUT_PATH.replace('.png', '_dark.png')
    plt.savefig(dark_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {dark_path}")


if __name__ == "__main__":
    main()

