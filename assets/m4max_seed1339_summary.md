# m4max_seed1339 — Visualization Summary

| Run | attn_mode | d_attn | best_val | best_ppl | KV@ctx (MB) | KV@128k (GB) |
|-----|----------:|-------:|---------:|---------:|------------:|------------:|
| baseline | standard | 512 | 6.2202 | 502.8 | 9.00 | 4.39 |
| bottleneck | bottleneck | 144 | 6.3538 | 574.7 | 1.69 | 0.82 |
| gqa | gqa | 768 | 6.5273 | 683.6 | 1.50 | 0.73 |
| decoupled | decoupled | 144 | 5.7078 | 301.2 | 1.69 | 0.82 |

- KV@128k ratio (baseline/bottleneck): **5.33×**
- Best-val delta (bottleneck - baseline): **+0.1336**
- KV@128k ratio (baseline/gqa): **6.00×**
- Best-val delta (gqa - baseline): **+0.3072**
- KV@128k ratio (baseline/decoupled): **5.33×**
- Best-val delta (decoupled - baseline): **-0.5124**
