from __future__ import annotations

import argparse
import datetime
import math
import os
import re
import sys
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def pick_device(explicit: Optional[str] = None) -> torch.device:
    import torch
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_layers_from_out_dir(out_dir: str) -> Optional[int]:
    """Infer model depth from the run directory name.

    Required for the intent-first workflow: the public CLI does not expose architecture flags.
    Expected patterns include: `..._l22_...`, `..._layers22_...`, `...-l22-...`.
    """
    try:
        base = os.path.basename(str(out_dir).rstrip("/")).lower()
    except Exception:
        return None
    m = re.search(r"(?:^|[_\\-])(?:l|layers)(\d+)(?:$|[_\\-])", base)
    if not m:
        return None
    try:
        n = int(m.group(1))
        return n if n > 0 else None
    except Exception:
        return None


def infer_dataset_tokens_from_path(path: str) -> Optional[int]:
    """Infer dataset token count from filename conventions (no scanning).

    Supported examples:
      - `fineweb_20b.npy`   -> 20_000_000_000
      - `fineweb_100m.npy`  -> 100_000_000
      - `fineweb_1b.npy`    -> 1_000_000_000
    """
    try:
        base = os.path.basename(str(path)).lower()
    except Exception:
        return None
    m = re.search(r"(\d+)([bm])", base)
    if not m:
        # Optional: read sibling `.meta` file (no scanning; supports arbitrary filenames).
        try:
            meta_path = str(path) + ".meta"
            if os.path.exists(meta_path):
                raw = open(meta_path, "r", encoding="utf-8").read()
                for line in raw.splitlines():
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    if k.strip().lower() == "tokens":
                        try:
                            n = int(str(v).strip().replace("_", ""))
                            return n if n > 0 else None
                        except Exception:
                            return None
        except Exception:
            return None
        return None
    try:
        k = int(m.group(1))
        suf = str(m.group(2))
        if suf == "m":
            return int(k * 1_000_000)
        if suf == "b":
            return int(k * 1_000_000_000)
    except Exception:
        return None
    return None


def set_seed(seed: int) -> None:
    import torch
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Intent → derived model shape
# -----------------------------

# Chinchilla-style default: optimal tokens ≈ 20 × params.
_TOKENS_PER_PARAM: float = 20.0
# GPT-style: d_ff = 4 × d_model.
_MLP_RATIO: int = 4
# Rough param model for GPT blocks (QKV+O + MLP) with MLP ratio 4: ~ 12 * d_model^2 per layer.
_PARAMS_PER_LAYER_COEFF: float = 12.0
# Bottleneck/decoupled KV reduction target (d_model / attn_dim).
_BOTTLENECK_RATIO: float = 5.3333333333


def _round_up(x: float, multiple: int) -> int:
    multiple = int(max(1, multiple))
    return int(math.ceil(float(x) / float(multiple)) * multiple)


def _lcm(a: int, b: int) -> int:
    a = int(abs(a))
    b = int(abs(b))
    if a == 0 or b == 0:
        return int(max(a, b))
    return int(a // math.gcd(a, b) * b)


def _round_nearest(x: float, multiple: int) -> int:
    multiple = int(max(1, multiple))
    return int(round(float(x) / float(multiple)) * multiple)


def derive_target_params(dataset_tokens: int) -> int:
    return int(max(1.0, float(dataset_tokens) / float(_TOKENS_PER_PARAM)))


def derive_d_model(*, layers: int, target_params: int, multiple: int = 128, min_d_model: int = 256) -> int:
    layers = int(max(1, layers))
    target_params = int(max(1, target_params))
    raw = math.sqrt(float(target_params) / (float(_PARAMS_PER_LAYER_COEFF) * float(layers)))
    return int(max(int(min_d_model), _round_up(raw, multiple)))


def derive_n_head(d_model: int) -> int:
    d_model = int(max(1, d_model))
    # Prefer 128-dim heads when possible; fall back to 64.
    want = max(1, int(round(d_model / 128.0)))
    # Find nearest divisor of d_model (so head_dim is integral).
    for delta in range(0, 64):
        for cand in (want - delta, want + delta):
            if cand >= 1 and d_model % cand == 0:
                return int(cand)
    # Fallback: smallest divisor >= 1
    for cand in range(1, d_model + 1):
        if d_model % cand == 0:
            return int(cand)
    return 1


def apply_intent(args: argparse.Namespace) -> None:
    """Fill architecture fields from *intent* (exp + layers + dataset scale).

    Required inputs (must be inferable):
    - layers: inferred from `--out-dir` tag like `_l22_` if not already set
    - dataset token scale: inferred from `--data` filename like `fineweb_20b.npy`

    Everything else is derived.
    """
    # 1) Resolve dataset token scale (or explicit target params).
    # - If `args.target_params` exists, treat it as the parameter budget directly.
    # - Else infer dataset_tokens from `args.dataset_tokens` (if provided) or filename conventions.
    target_params: Optional[int] = None
    try:
        tp = getattr(args, "target_params", None)
        if tp is not None and int(tp) > 0:
            target_params = int(tp)
    except Exception:
        target_params = None

    data = getattr(args, "data", None)
    dataset_tokens: Optional[int] = None
    try:
        dtok = getattr(args, "dataset_tokens", None)
        if dtok is not None and int(dtok) > 0:
            dataset_tokens = int(dtok)
    except Exception:
        dataset_tokens = None

    if target_params is None and data:
        dataset_tokens = dataset_tokens if dataset_tokens is not None else infer_dataset_tokens_from_path(str(data))
        if dataset_tokens is None:
            raise ValueError(f"Could not infer dataset token scale from data path {data!r}. Use a name like `fineweb_20b.npy` or provide a sibling `.meta` with `tokens: ...`.")
        target_params = derive_target_params(int(dataset_tokens))

    # 2) Resolve layers.
    if getattr(args, "layers", None) in (None, 0, "0"):
        out_dir = getattr(args, "out_dir", None)
        L = infer_layers_from_out_dir(str(out_dir)) if out_dir else None
        if L is None:
            if target_params is None:
                raise ValueError("Need either --layers, --size/target_params, or a dataset token scale to infer layers.")

            # Prefer ~128-dim heads by choosing a depth that yields a clean divisor layout.
            # (This remains a cheap heuristic; hardware-aware layer selection can live in optimizer.py.)
            if int(target_params) < 50_000_000:
                candidates = (2, 4, 6, 8, 12)
                min_d_model = 64
                multiple = 64
            elif int(target_params) < 500_000_000:
                candidates = (8, 12, 16, 20, 22, 24, 32)
                min_d_model = 256
                multiple = 128
            else:
                candidates = (12, 16, 20, 22, 24, 32, 48, 64, 96)
                min_d_model = 256
                multiple = 128

            best_L = int(candidates[0])
            best_score = float("inf")
            for cand in candidates:
                d_model = derive_d_model(layers=int(cand), target_params=int(target_params), multiple=multiple, min_d_model=min_d_model)
                n_head = derive_n_head(int(d_model))
                head_dim = float(d_model) / max(1.0, float(n_head))
                score = abs(head_dim - 128.0) / 128.0
                if score < best_score:
                    best_score = score
                    best_L = int(cand)
            L = int(best_L)
        args.layers = int(L)

    # If we're missing data (e.g. sample mode), there's nothing else to infer here.
    if not data:
        return

    if target_params is None:
        # Should be impossible given the checks above, but keep it safe.
        dtok = infer_dataset_tokens_from_path(str(data))
        if dtok is None:
            return
        target_params = derive_target_params(int(dtok))

    if int(getattr(args, "d_model", 0) or 0) <= 0:
        args.d_model = derive_d_model(layers=int(args.layers), target_params=int(target_params))
    if int(getattr(args, "n_head", 0) or 0) <= 0:
        args.n_head = derive_n_head(int(args.d_model))
    if int(getattr(args, "d_ff", 0) or 0) <= 0:
        args.d_ff = int(_MLP_RATIO) * int(args.d_model)
    if int(getattr(args, "embed_dim", 0) or 0) <= 0:
        args.embed_dim = int(args.d_model)

EXP_PRESETS: Dict[str, Dict[str, Any]] = {
    "paper_baseline": dict(attn_mode="standard"),
    "paper_bottleneck": dict(attn_mode="bottleneck", null_attn=True),
    # Decoupled flagship path: keep null_attn off by default to avoid extra branching and to preserve
    # fused/streaming decode fast paths. See `production/ablate_null_attn.py` for an explicit ablation.
    "paper_decoupled": dict(attn_mode="decoupled", tie_qk=True, null_attn=False, rope=True),
    "paper_gqa": dict(attn_mode="gqa"),
    # Training-oriented preset: expresses intent only; runtime performance is auto-tuned.
    "train_decoupled_fast": dict(attn_mode="decoupled", tie_qk=True, rope=True, null_attn=False),
}


def _argv_has_flag(flag: str) -> bool:
    # Detect explicit user overrides (argparse defaults are otherwise indistinguishable).
    #
    # Support both:
    #   --d-model 123
    #   --d-model=123
    #
    # NOTE: argparse normalizes both forms, but we inspect sys.argv directly here to decide whether
    # presets should override a value.
    if flag in sys.argv:
        return True
    prefix = str(flag) + "="
    return any(str(a).startswith(prefix) for a in sys.argv)


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

    # attn_mode
    if not _argv_has_flag("--attn-mode") and "attn_mode" in preset:
        args.attn_mode = preset["attn_mode"]

    # Experiment-specific dims (derived from d_model; no size tables).
    d_model = int(getattr(args, "d_model", 0) or 0)
    if d_model <= 0:
        return
    mode = str(getattr(args, "attn_mode", "") or "")
    if mode in ("standard", "gqa"):
        if int(getattr(args, "attn_dim", 0) or 0) <= 0:
            args.attn_dim = int(d_model)
    if mode == "bottleneck":
        if int(getattr(args, "attn_dim", 0) or 0) <= 0:
            args.attn_dim = int(max(32, _round_up(float(d_model) / float(_BOTTLENECK_RATIO), 32)))
    if mode == "decoupled":
        nh = int(getattr(args, "n_head", 0) or 0)
        rope_enabled = not bool(getattr(args, "no_rope", False))
        # Decoupled requires sem_dim, geo_dim, and attn_dim each divisible by n_head.
        # Additionally, if RoPE is enabled, geo_head_dim must be even ⇒ geo_dim divisible by 2*n_head.
        #
        # We keep `attn_dim` aligned to 32 for matmul efficiency, but make the multiple compatible with heads.
        sem_geo_multiple = int(2 * nh) if (rope_enabled and nh > 0) else int(max(1, nh))
        attn_multiple = _lcm(32, int(max(1, sem_geo_multiple))) if nh > 0 else 32

        if int(getattr(args, "attn_dim", 0) or 0) <= 0:
            args.attn_dim = int(max(attn_multiple, _round_up(float(d_model) / float(_BOTTLENECK_RATIO), attn_multiple)))

        attn_dim = int(args.attn_dim)

        # Only auto-derive sem/geo when not provided (minimal CLI doesn't expose these knobs).
        if int(getattr(args, "sem_dim", 0) or 0) <= 0 or int(getattr(args, "geo_dim", 0) or 0) <= 0:
            if nh <= 0:
                # Fallback: preserve previous behavior (should be rare; apply_intent normally sets n_head).
                sem = int(max(32, _round_up(float(attn_dim) / 3.0, 32)))
                sem = min(sem, attn_dim - 32)
                geo = int(attn_dim - sem)
            else:
                # Choose a semantic/geometric split near 1/3 vs 2/3, rounded to a head-compatible multiple.
                sem = int(_round_nearest(float(attn_dim) / 3.0, sem_geo_multiple))
                sem = int(max(sem_geo_multiple, min(sem, attn_dim - sem_geo_multiple)))
                geo = int(attn_dim - sem)
            args.sem_dim = int(sem)
            args.geo_dim = int(geo)
    if mode == "gqa":
        if getattr(args, "kv_head", None) is None:
            nh = int(getattr(args, "n_head", 0) or 0)
            if nh > 0:
                # prefer 1/4 query heads as KV heads (and ensure it divides).
                cand = max(1, nh // 4)
                while cand > 1 and (nh % cand) != 0:
                    cand -= 1
                args.kv_head = int(max(1, cand))

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
    """If the user didn't set --out-dir, build it as runs/{exp}_{tag}.

    NOTE: paper harnesses should always pass --out-dir so intent tags (like `_l22_`) are preserved.
    """
    if getattr(args, "out_dir", None):
        return str(args.out_dir)
    exp = getattr(args, "exp", None)
    run_root = getattr(args, "run_root", "runs")
    tag = getattr(args, "run_tag", None)
    if not exp or exp == "paper_all":
        return None
    name = str(exp).replace("paper_", "")
    if tag:
        name = f"{name}_{tag}"
    return os.path.join(run_root, name)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")
