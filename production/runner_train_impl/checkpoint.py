"""Checkpoint save/load helpers.

Why this exists:
- Training state needs to survive process restarts (preemption, iteration, debugging).
- Keeping checkpoint I/O separate makes the training loop easier to read and safer to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def _to_str_object_dict(d: object) -> dict[str, object]:
    if not isinstance(d, dict):
        return {}
    return {str(k): v for k, v in d.items()}  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]


@dataclass(frozen=True)
class TrainCheckpoint:
    """Why: strongly-typed view of the checkpoint payload we care about."""

    opt_step: int
    model_state: dict[str, object]
    optim_state: dict[str, object]


def save_checkpoint(
    *,
    out_dir: str,
    opt_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: object,
    extra: dict[str, object] | None = None,
) -> str | None:
    """Why: persist enough state to resume training deterministically."""
    try:
        os.makedirs(str(out_dir), exist_ok=True)
        last_path = os.path.join(str(out_dir), "last.pt")
        step_path = os.path.join(str(out_dir), f"ckpt_step{int(opt_step)}.pt")

        cfg_dict: dict[str, object] = {}
        raw = getattr(cfg, "__dict__", None)
        if isinstance(raw, dict):
            raw_obj: object = raw  # pyright: ignore[reportUnknownVariableType]
            cfg_dict = _to_str_object_dict(raw_obj)  # pyright: ignore[reportUnknownArgumentType]
        if "device" in cfg_dict:
            cfg_dict["device"] = str(cfg_dict["device"])
        if "attn_mode" in cfg_dict and hasattr(cfg_dict["attn_mode"], "value"):
            cfg_dict["attn_mode"] = getattr(cfg_dict["attn_mode"], "value")

        payload: dict[str, object] = {
            "opt_step": int(opt_step),
            "model": model.state_dict(),
            # Key name expected by resume harness/tests.
            "opt": optimizer.state_dict(),
            # Back-compat alias.
            "optimizer": optimizer.state_dict(),
            # Config payload used by sampling/bench code.
            "config": cfg_dict,
            # Back-compat alias.
            "cfg": cfg_dict,
        }
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, last_path)
        # Also keep a step-specific snapshot for debugging (best-effort).
        try:
            torch.save(payload, step_path)
        except Exception:
            pass
        return str(last_path)
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _torch_load_obj(path: str) -> object:
    # `torch.load` is typed as returning `Any` in stubs; isolate it behind an `object` boundary.
    return torch.load(str(path), map_location="cpu")  # pyright: ignore[reportAny]


def load_checkpoint(path: str) -> TrainCheckpoint | None:
    """Why: isolate deserialization hazards and keep training code clean."""
    try:
        raw_obj = _torch_load_obj(path)
        if not isinstance(raw_obj, dict):
            return None
        raw: dict[str, object] = _to_str_object_dict(raw_obj)  # pyright: ignore[reportUnknownArgumentType]

        opt_step_obj = raw.get("opt_step", 0)
        opt_step = int(opt_step_obj) if isinstance(opt_step_obj, (int, bool, float, str)) else 0

        model_state_obj = raw.get("model", {})
        optim_state_obj = raw.get("opt", raw.get("optimizer", {}))

        if not isinstance(model_state_obj, dict) or not isinstance(optim_state_obj, dict):
            return None
        model_state_obj2: object = model_state_obj  # pyright: ignore[reportUnknownVariableType]
        optim_state_obj2: object = optim_state_obj  # pyright: ignore[reportUnknownVariableType]
        model_state: dict[str, object] = _to_str_object_dict(model_state_obj2)  # pyright: ignore[reportUnknownArgumentType]
        optim_state: dict[str, object] = _to_str_object_dict(optim_state_obj2)  # pyright: ignore[reportUnknownArgumentType]
        return TrainCheckpoint(
            opt_step=int(opt_step),
            model_state=model_state,
            optim_state=optim_state,
        )
    except (OSError, RuntimeError, ValueError, TypeError):
        return None

