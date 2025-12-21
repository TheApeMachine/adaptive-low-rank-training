# Run Summary

- Created: `2025-12-20T21:50:15+01:00`
- Out dir: `runs/mac_fw100m_small_baseline_s1337`
- Device: `mps`
- Command: `main.py --mode train --size small --exp paper_baseline --data fineweb_100m.npy --out-dir runs/mac_fw100m_small_baseline_s1337 --seed 1337 --steps 50000 --eval-every 1000 --eval-iters 20 --save-every 2000 --instrument full --tokenizer tiktoken --block 1024 --wandb --wandb-name mac_fw100m_small_baseline_s1337 --wandb-project production --wandb-entity p4n0p71c0n --wandb-group mac_fw100m`

## Model Config

```json
{
  "attn_dim": 512,
  "attn_mode": "standard",
  "block_size": 1024,
  "d_ff": 3072,
  "d_model": 768,
  "decoupled_gate": true,
  "dropout": 0.0,
  "embed_dim": 768,
  "geo_dim": 64,
  "kv_head": null,
  "learned_temp": true,
  "mlp": "swiglu",
  "n_head": 12,
  "n_layer": 12,
  "null_attn": false,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 32,
  "tie_qk": false,
  "vocab_size": 50257
}
```

