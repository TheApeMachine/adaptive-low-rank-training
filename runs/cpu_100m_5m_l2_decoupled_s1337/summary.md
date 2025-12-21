# Run Summary

- Created: `2025-12-21T00:08:29+01:00`
- Out dir: `runs/cpu_100m_5m_l2_decoupled_s1337`
- Device: `cpu`
- Command: `main.py --mode train --data fineweb_100m.npy --result decoupled`

## Model Config

```json
{
  "attn_dim": 128,
  "attn_mode": "decoupled",
  "block_size": 32,
  "d_ff": 2048,
  "d_model": 512,
  "decoupled_gate": true,
  "dropout": 0.0,
  "embed_dim": 512,
  "geo_dim": 64,
  "kv_head": null,
  "learned_temp": true,
  "mlp": "swiglu",
  "n_head": 4,
  "n_layer": 2,
  "null_attn": false,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 64,
  "tie_qk": true,
  "vocab_size": 50257
}
```

