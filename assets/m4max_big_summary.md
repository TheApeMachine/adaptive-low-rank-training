# m4max_big — Visualization Summary

| Run | attn_mode | d_attn | best_val | best_ppl | KV@ctx (MB) | KV@128k (GB) |
|-----|----------:|-------:|---------:|---------:|------------:|------------:|
| baseline | standard | 512 | 6.3262 | 559.0 | 9.00 | 4.39 |
| gqa | gqa | 768 | 6.3024 | 545.9 | 1.50 | 0.73 |
| bottleneck | bottleneck | 144 | 6.1271 | 458.1 | 1.69 | 0.82 |
| decoupled | decoupled | 144 | 5.8622 | 351.5 | 1.69 | 0.82 |

- KV@128k ratio (baseline/gqa): **6.00×**
- Best-val delta (gqa - baseline): **-0.0238**
- KV@128k ratio (baseline/bottleneck): **5.33×**
- Best-val delta (bottleneck - baseline): **-0.1991**
- KV@128k ratio (baseline/decoupled): **5.33×**
- Best-val delta (decoupled - baseline): **-0.4640**
