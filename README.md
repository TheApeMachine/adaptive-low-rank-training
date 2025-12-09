# Adaptive Low-Rank Training for Transformers

This repository contains research and experimental code for **Adaptive Low-Rank Training**, a method to dynamically optimize the rank of linear layers in Transformer models during training.

Unlike static compression techniques (like LoRA) or fixed-rank approximations, this approach allows the model to "learn" its own optimal topology. Layers with high information density grow in rank, while redundant layers shrink, automatically allocating parameters where they are most needed.

## 🚀 Latest State: v13 Lazy SVD Adaptive

The current state-of-the-art implementation in this project is **`v13_transformer_lowrank_lazy_svd_adaptive.py`**.

### Key Innovations in v13
*   **Fixed-Shape, Dynamic Rank**: Unlike previous versions that resized tensors (destroying optimizer state), v13 allocates "maximal" buffers (`U_full`, `V_full`) and slices them during the forward pass (`U_full[:, :rank]`). This **preserves AdamW momentum** throughout training, solving the "re-initialization shock" problem.
*   **Lazy SVD Scheduling**: SVD is computationally expensive, so it is not run every step. The model waits for a warmup period and then runs SVD checks at intervals.
*   **Adaptive Intervals**: The time between SVD checks is dynamic. If the layer is "stable" (rank isn't changing), the interval grows (checking less often). If the layer is "unstable" (high tail energy), the interval shrinks to adapt quickly.
*   **Spectral Energy Targeting**: Ranks are chosen to preserve a specific percentage of spectral energy (default 98%), ensuring that only noise is pruned.

## 📂 Project Structure & Evolution

The codebase documents the evolution of the research idea:

| Version | Description |
|---------|-------------|
| **v13** | **Current Best.** Lazy SVD with adaptive intervals. Uses fixed-shape buffers to preserve optimizer momentum. |
| **v11** | Momentum-based rank adaptation experiments. |
| **v10** | Scaled Spectral Adaptive. Introduced Stable Rank estimation and SVD resizing (but reset optimizer). |
| **v8 - v9** | Introduction of Spectral methods and Bidirectional rank adjustment. |
| **v7** | **Dense Baseline** (`v7_transformer_dense_baseline.py`) and various experiments with EMA and Autograd-based rank adaptation. |
| **v3 - v6** | Early Adaptive Low-Rank implementations. |
| **v1 - v2** | Initial experiments with **Gradient Grouping** and clustering neurons based on similarity. |

## 🛠️ Installation & Requirements

The project relies on standard PyTorch.

```bash
pip install torch torchvision tqdm
```

*Note: The code is designed to be hardware-agnostic, automatically selecting CUDA (NVIDIA), MPS (Apple Silicon), or CPU.*

## 🏃 Usage

### 1. Prepare Data
The scripts expect a text file. The v13 script includes a character-level loader, but you can point it to any text file.

### 2. Run the Training
You can run the latest implementation directly:

```bash
python3 v13_transformer_lowrank_lazy_svd_adaptive.py \
    --data-file wiki.train.tokens \
    --log-file v13_log.jsonl \
    --epochs 30 \
    --init-rank 64 \
    --d-model 256 \
    --n-layers 6 \
    --n-heads 4
```

## 📊 Methodology (v13)

### The Algorithm
1.  **Allocation**: Allocate `U_full` and `V_full` with `max_rank` (e.g., 512). Initialize an integer `rank` pointer (e.g., 64).
2.  **Forward Pass**: Use slicing: $ W_{active} = U_{full}[:, :rank] \times V_{full}[:rank, :] $. This is efficient and allows gradients to flow only to active components.
3.  **Lazy Check**: Every $N$ steps (where $N$ is dynamic), compute the SVD of the active weight matrix $W_{active}$ on the CPU.
4.  **Rank Decision**:
    *   Calculate the cumulative spectral energy.
    *   Find the smallest rank $k$ that retains 98% of the energy.
    *   Smoothly adjust the current `rank` pointer towards $k$ (clamped step size).
5.  **Interval Adjustment**:
    *   If "tail energy" (discarded energy) is low, the approximation is good $\to$ increase check interval (save compute).
    *   If "tail energy" is high, we are losing info $\to$ decrease check interval (adapt faster).

### Findings
*   **Stability**: Preserving optimizer state (AdamW buffers) is critical for convergence. Virtual slicing is superior to physical resizing for this reason.
*   **Efficiency**: SVD on CPU is fast enough if done "lazily" (every 100-1000 steps). It adds negligible overhead to the training loop.

## 📄 Research Notes
*   `Adaptive_LowRank_Training_Research.docx.pdf`: Detailed theoretical background.
*   `v10_review_gemini_3_pro_temp_0.md`: AI-assisted review of the spectral approach.

## 📜 License
[MIT](LICENSE)
