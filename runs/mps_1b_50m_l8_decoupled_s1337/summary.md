# Run Summary

- Created: `2025-12-21T00:40:32+01:00`
- Out dir: `runs/mps_1b_50m_l8_decoupled_s1337`
- Device: `mps`
- Command: `main.py --mode train --data fineweb_1b.npy --result decoupled`

## Model Config

```json
{
  "attn_dim": 192,
  "attn_mode": "decoupled",
  "block_size": 1024,
  "d_ff": 3072,
  "d_model": 768,
  "decoupled_gate": true,
  "dropout": 0.0,
  "embed_dim": 768,
  "geo_dim": 132,
  "kv_head": null,
  "learned_temp": true,
  "mlp": "swiglu",
  "n_head": 6,
  "n_layer": 8,
  "null_attn": false,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 60,
  "tie_qk": true,
  "vocab_size": 50257
}
```

