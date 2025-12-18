# v29 Suite Summary

- tag: `m4max`
- seeds: `1337,1338,1339`

## Best val loss / ppl (per seed)

| variant | seed | best_val_loss | best_ppl | kv@128k | run_dir |
|---|---:|---:|---:|---:|---|
| baseline | 1337 | 6.296133 | 542.47 | 4.39GB | `runs/m4max_baseline_seed1337` |
| baseline | 1338 | 6.637763 | 763.39 | 4.39GB | `runs/m4max_baseline_seed1338` |
| baseline | 1339 | 6.220188 | 502.80 | 4.39GB | `runs/m4max_baseline_seed1339` |
| gqa_kv2 | 1337 | 6.214722 | 500.06 | 750.0MB | `runs/m4max_gqa_kv2_seed1337` |
| gqa_kv2 | 1338 | 6.302364 | 545.86 | 750.0MB | `runs/m4max_gqa_kv2_seed1338` |
| gqa_kv2 | 1339 | 6.527348 | 683.58 | 750.0MB | `runs/m4max_gqa_kv2_seed1339` |
| bottleneck_144 | 1337 | 6.525506 | 682.32 | 843.8MB | `runs/m4max_bottleneck_144_seed1337` |
| bottleneck_144 | 1338 | 6.391262 | 596.61 | 843.8MB | `runs/m4max_bottleneck_144_seed1338` |
| bottleneck_144 | 1339 | 6.353791 | 574.67 | 843.8MB | `runs/m4max_bottleneck_144_seed1339` |
| decoupled_48_96 | 1337 | 5.878566 | 357.30 | 843.8MB | `runs/m4max_decoupled_48_96_seed1337` |
| decoupled_48_96 | 1338 | 5.824183 | 338.38 | 843.8MB | `runs/m4max_decoupled_48_96_seed1338` |
| decoupled_48_96 | 1339 | 5.707824 | 301.21 | 843.8MB | `runs/m4max_decoupled_48_96_seed1339` |

## Aggregate across seeds (mean ± std)

| variant | best_val_loss | best_ppl |
|---|---:|---:|
| baseline | 6.385 ± 0.222 | 602.9 ± 140.4 |
| gqa_kv2 | 6.348 ± 0.161 | 576.5 ± 95.5 |
| bottleneck_144 | 6.424 ± 0.090 | 617.9 ± 56.9 |
| decoupled_48_96 | 5.804 ± 0.087 | 332.3 ± 28.5 |

