# Run Summary

- Created: `2025-12-20T13:50:20+01:00`
- Out dir: `runs/1b_train_decoupled_fast`
- Device: `mps`
- Command: `main.py --mode train --size 1b --exp train_decoupled_fast --data fineweb_1b.npy --wandb`

## Model Config

```json
{
  "attn_dim": 768,
  "attn_mode": "decoupled",
  "block_size": 4096,
  "d_ff": 8192,
  "d_model": 2048,
  "decoupled_gate": true,
  "dropout": 0.0,
  "embed_dim": 2048,
  "geo_dim": 512,
  "kv_head": null,
  "learned_temp": true,
  "mlp": "swiglu",
  "n_head": 16,
  "n_layer": 18,
  "null_attn": false,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 256,
  "tie_qk": true,
  "vocab_size": 50256
}
```

