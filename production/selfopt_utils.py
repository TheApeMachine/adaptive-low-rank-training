from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict

import torch


def device_sig(device: torch.device) -> str:
    """Stable device signature for caching/plans."""
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "cuda"
        try:
            props = torch.cuda.get_device_properties(idx)
            cc = f"cc{props.major}{props.minor}"
        except Exception:
            cc = "cc"
        return f"cuda:{idx}:{name}:{cc}"
    return str(device.type)


def hash_cfg(cfg: Any) -> str:
    """Short hash of a (dataclass-like) config for caching keys."""
    try:
        payload = json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")
    except Exception:
        payload = json.dumps(getattr(cfg, "__dict__", {}), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def snapshot_rng(device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import random

        out["py_random"] = random.getstate()
    except Exception:
        pass
    try:
        out["torch_cpu"] = torch.get_rng_state()
    except Exception:
        pass
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            out["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return out


def restore_rng(device: torch.device, snap: Dict[str, Any]) -> None:
    try:
        import random

        if "py_random" in snap:
            random.setstate(snap["py_random"])
    except Exception:
        pass
    try:
        if "torch_cpu" in snap:
            torch.set_rng_state(snap["torch_cpu"])
    except Exception:
        pass
    try:
        if device.type == "cuda" and torch.cuda.is_available() and "torch_cuda" in snap:
            torch.cuda.set_rng_state_all(snap["torch_cuda"])
    except Exception:
        pass


def is_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return (
        ("out of memory" in msg)
        or ("cuda error: out of memory" in msg)
        or ("cudnn error: out of memory" in msg)
        or ("mps backend out of memory" in msg)
        or ("resource exhausted" in msg)
    )


