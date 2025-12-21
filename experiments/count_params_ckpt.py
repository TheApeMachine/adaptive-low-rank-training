#!/usr/bin/env python3
"""
count_params_ckpt.py

Compute parameter counts for a saved checkpoint (.pt) produced by the v29 training script.
Useful for substantiating "parameter-matched" / "same budget" claims in the paper.

Usage:
  python3.12 count_params_ckpt.py runs/m4max_big_baseline/best.pt
  python3.12 count_params_ckpt.py runs/m4max_big_gqa/best.pt runs/m4max_big_decoupled/best.pt
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import torch


def count_state_dict_params(sd: Dict[str, Any]) -> Tuple[int, int]:
    """
    Returns (n_params, n_bytes_assuming_sd_dtype).
    """
    n_params = 0
    n_bytes = 0
    for _, v in sd.items():
        if not torch.is_tensor(v):
            continue
        n = int(v.numel())
        n_params += n
        n_bytes += n * int(v.element_size())
    return n_params, n_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="Paths to .pt checkpoints (best.pt / last.pt)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu recommended for counting")
    args = ap.parse_args()

    device = torch.device(str(args.device))

    for p in args.ckpts:
        ckpt = torch.load(p, map_location=device)
        cfg = ckpt.get("config", {})
        sd = ckpt.get("model", {})
        if not isinstance(sd, dict):
            raise RuntimeError(f"{p}: checkpoint['model'] is not a state_dict dict")

        n_params, n_bytes = count_state_dict_params(sd)
        mb = float(n_bytes) / (1024.0 * 1024.0)
        print("=" * 80)
        print(p)
        print(f"n_params: {n_params:,}")
        print(f"state_dict_bytes: {n_bytes:,} ({mb:.1f} MB)")
        if isinstance(cfg, dict):
            # Print a small signature to sanity check comparability
            keys = ["attn_mode", "d_model", "n_layer", "n_head", "attn_dim", "sem_dim", "geo_dim", "kv_head", "vocab_size"]
            sig = {k: cfg.get(k, None) for k in keys}
            print(f"config_signature: {sig}")


if __name__ == "__main__":
    main()


