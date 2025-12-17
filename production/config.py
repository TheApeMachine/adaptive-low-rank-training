from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import Any, Dict, Optional


def pick_device(explicit: Optional[str] = None) -> torch.device:
    import torch
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import torch
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Experiment presets (v27)
# -----------------------------

SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    # params, d_model, layers, heads, d_ff, context, batch, steps
    "tiny": dict(d_model=512, layers=6, n_head=8, d_ff=2048, block=1024, batch_size=16, steps=6000),
    "small": dict(d_model=768, layers=12, n_head=12, d_ff=3072, block=1024, batch_size=8, steps=10000),
    "medium": dict(d_model=1024, layers=24, n_head=16, d_ff=4096, block=2048, batch_size=4, steps=15000),
    "large": dict(d_model=1536, layers=24, n_head=16, d_ff=6144, block=2048, batch_size=2, steps=20000),
}

# Attention dims for the "paper_*" experiments.
BOTTLENECK_ATTN_DIM: Dict[str, int] = {"tiny": 96, "small": 144, "medium": 192, "large": 288}
DECOUPLED_SEM_DIM: Dict[str, int] = {"tiny": 32, "small": 48, "medium": 64, "large": 96}
DECOUPLED_GEO_DIM: Dict[str, int] = {"tiny": 64, "small": 96, "medium": 128, "large": 192}
DECOUPLED_ATTN_DIM: Dict[str, int] = {"tiny": 96, "small": 144, "medium": 192, "large": 288}
GQA_KV_HEAD: Dict[str, int] = {"tiny": 2, "small": 3, "medium": 4, "large": 4}

EXP_PRESETS: Dict[str, Dict[str, Any]] = {
    "paper_baseline": dict(attn_mode="standard"),
    "paper_bottleneck": dict(attn_mode="bottleneck", null_attn=True),
    "paper_decoupled": dict(attn_mode="decoupled", tie_qk=True, null_attn=True, rope=True),
    "paper_gqa": dict(attn_mode="gqa"),
}


def _argv_has_flag(flag: str) -> bool:
    # Detect explicit user overrides (argparse defaults are otherwise indistinguishable).
    return flag in sys.argv


def apply_size_preset(args: argparse.Namespace) -> None:
    if not getattr(args, "size", None):
        return
    size = str(args.size)
    if size not in SIZE_PRESETS:
        raise ValueError(f"Unknown size preset: {size}")
    p = SIZE_PRESETS[size]
    # Only override if the user did not specify the corresponding CLI flag.
    if not _argv_has_flag("--d-model"):
        args.d_model = p["d_model"]
    if not _argv_has_flag("--layers"):
        args.layers = p["layers"]
    if not _argv_has_flag("--n-head"):
        args.n_head = p["n_head"]
    if not _argv_has_flag("--d-ff"):
        args.d_ff = p["d_ff"]
    if not _argv_has_flag("--block"):
        args.block = p["block"]
    if not _argv_has_flag("--batch-size"):
        args.batch_size = p["batch_size"]
    if not _argv_has_flag("--steps"):
        args.steps = p["steps"]
    # default embed_dim tracks d_model unless explicitly set
    if not _argv_has_flag("--embed-dim"):
        args.embed_dim = p["d_model"]


def apply_exp_preset(args: argparse.Namespace) -> None:
    if not getattr(args, "exp", None):
        return
    exp = str(args.exp)
    if exp not in EXP_PRESETS and exp != "paper_all":
        raise ValueError(f"Unknown experiment preset: {exp}")

    # For paper_all, we don't set mode here; the runner loops over EXP_PRESETS.
    if exp == "paper_all":
        return

    preset = EXP_PRESETS[exp]
    size = str(getattr(args, "size", "")) if getattr(args, "size", None) else None

    # attn_mode
    if not _argv_has_flag("--attn-mode") and "attn_mode" in preset:
        args.attn_mode = preset["attn_mode"]

    # Experiment-specific dims (size-dependent)
    if size is not None:
        if exp == "paper_bottleneck":
            if not _argv_has_flag("--attn-dim"):
                args.attn_dim = BOTTLENECK_ATTN_DIM[size]
        if exp == "paper_decoupled":
            if not _argv_has_flag("--sem-dim"):
                args.sem_dim = DECOUPLED_SEM_DIM[size]
            if not _argv_has_flag("--geo-dim"):
                args.geo_dim = DECOUPLED_GEO_DIM[size]
            if not _argv_has_flag("--attn-dim"):
                args.attn_dim = DECOUPLED_ATTN_DIM[size]
        if exp == "paper_gqa":
            if not _argv_has_flag("--kv-head"):
                args.kv_head = GQA_KV_HEAD[size]
            if not _argv_has_flag("--attn-dim"):
                # Keep head_dim identical to baseline by default.
                args.attn_dim = int(getattr(args, "d_model", SIZE_PRESETS[size]["d_model"]))

    # Bool toggles (only set if user didn't explicitly toggle)
    if "null_attn" in preset:
        if (not _argv_has_flag("--null-attn")) and (not _argv_has_flag("--no-null-attn")):
            args.null_attn = bool(preset["null_attn"])
    if "tie_qk" in preset:
        if (not _argv_has_flag("--tie-qk")) and (not _argv_has_flag("--no-tie-qk")):
            args.tie_qk = bool(preset["tie_qk"])
    if "rope" in preset:
        if (not _argv_has_flag("--no-rope")) and (not _argv_has_flag("--rope")):
            # v26 default is rope=True; keep explicit override logic anyway.
            if preset["rope"]:
                args.no_rope = False
            else:
                args.no_rope = True


def default_out_dir(args: argparse.Namespace) -> Optional[str]:
    """
    If the user didn't set --out-dir, build it as runs/{size}_{expSuffix}.
    Returns None if we cannot infer.
    """
    if getattr(args, "out_dir", None):
        return str(args.out_dir)
    size = getattr(args, "size", None)
    exp = getattr(args, "exp", None)
    run_root = getattr(args, "run_root", "runs")
    tag = getattr(args, "run_tag", None)
    if not size or not exp or exp == "paper_all":
        return None
    suffix = str(exp).replace("paper_", "")
    name = f"{size}_{suffix}"
    if tag:
        name = f"{name}_{tag}"
    return os.path.join(run_root, name)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


