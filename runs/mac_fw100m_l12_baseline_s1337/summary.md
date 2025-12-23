# Run Summary

- Created: `2025-12-23T06:05:24+01:00`
- Out dir: `runs/mac_fw100m_l12_baseline_s1337`
- Device: `mps`
- Command: `main.py --mode train --exp paper_baseline --data fineweb_100m.npy --out-dir runs/mac_fw100m_l12_baseline_s1337 --seed 1337 --steps 6000`

## Model Config

```json
{
  "attn_dim": 104,
  "attn_mode": "standard",
  "block_size": 512,
  "d_ff": 600,
  "d_model": 128,
  "decoupled_gate": true,
  "decoupled_gate_dynamic": true,
  "device": "mps",
  "diffusion_head": false,
  "diffusion_head_cfg_dropout_p": 0.1,
  "diffusion_head_cfg_guidance_scale": 1.5,
  "diffusion_head_loss_weight": 0.1,
  "diffusion_head_mlp_mult": 4,
  "diffusion_head_num_infer_steps": 12,
  "diffusion_head_num_train_timesteps": 1000,
  "diffusion_head_scheduler": "ddim",
  "diffusion_head_time_embed_dim": 128,
  "dim_multiplier": 0,
  "dropout": 0.0,
  "embed_dim": 32,
  "geo_dim": 0,
  "head_dim": 32,
  "head_policy": "standard",
  "kv_head": null,
  "learned_temp": true,
  "mlp": "swiglu",
  "n_head": 4,
  "n_layer": 12,
  "null_attn": false,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 104,
  "tie_qk": false,
  "train_long_seq_enabled": true,
  "train_long_seq_local_window": null,
  "train_long_seq_mem_block": null,
  "train_long_seq_mem_summarizer": "conv",
  "train_long_seq_q_chunk": null,
  "train_long_seq_threshold": null,
  "vocab_size": 50257
}
```

