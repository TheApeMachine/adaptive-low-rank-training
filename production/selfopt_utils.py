from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass

import torch
from production.selfopt_cache import as_object_list


def device_sig(device: torch.device) -> str:
    """Stable device signature for caching/plans."""
    if device.type == "cuda" and torch.cuda.is_available():
        idx_obj = getattr(device, "index", None)
        idx = int(idx_obj) if isinstance(idx_obj, int) else int(torch.cuda.current_device())
        try:
            name = torch.cuda.get_device_name(idx)
        except (RuntimeError, ValueError, TypeError):
            name = "cuda"
        def _cuda_props_obj(device_idx: int) -> object:
            # Torch stubs type this as Unknown; isolate it behind an `object` boundary.
            return torch.cuda.get_device_properties(device_idx)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        try:
            props = _cuda_props_obj(idx)
            maj = getattr(props, "major", None)
            minr = getattr(props, "minor", None)
            if isinstance(maj, int) and isinstance(minr, int):
                cc = f"cc{maj}{minr}"
            else:
                cc = "cc"
        except (RuntimeError, ValueError, TypeError):
            cc = "cc"
        return f"cuda:{idx}:{name}:{cc}"
    return str(device.type)


def hash_cfg(cfg: object) -> str:
    """Short hash of a (dataclass-like) config for caching keys."""
    try:
        if is_dataclass(cfg) and not isinstance(cfg, type):
            payload = json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")
        else:
            raise TypeError("not a dataclass")
    except (TypeError, ValueError):
        payload = json.dumps(getattr(cfg, "__dict__", {}), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def snapshot_rng(device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {}
    try:
        import random

        out["py_random"] = random.getstate()
    except Exception:  # pragma: no cover (best-effort)
        pass
    try:
        out["torch_cpu"] = torch.get_rng_state()
    except Exception:  # pragma: no cover (best-effort)
        pass
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            out["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:  # pragma: no cover (best-effort)
        pass
    return out


def restore_rng(device: torch.device, snap: dict[str, object]) -> None:
    try:
        import random

        if "py_random" in snap:
            st0 = snap["py_random"]
            if isinstance(st0, tuple):
                # random.getstate() is untyped in stubs; treat it as opaque.
                random.setstate(st0)  # pyright: ignore[reportUnknownArgumentType]
    except Exception:  # pragma: no cover (best-effort)
        pass
    try:
        if "torch_cpu" in snap:
            st = snap["torch_cpu"]
            if isinstance(st, torch.Tensor):
                torch.set_rng_state(st)
    except Exception:  # pragma: no cover (best-effort)
        pass
    try:
        if device.type == "cuda" and torch.cuda.is_available() and "torch_cuda" in snap:
            st2 = snap["torch_cuda"]
            st2_list = as_object_list(st2)
            if st2_list is not None:
                tensors: list[torch.Tensor] = []
                ok = True
                for t in st2_list:
                    if isinstance(t, torch.Tensor):
                        tensors.append(t)
                    else:
                        ok = False
                        break
                if ok and tensors:
                    torch.cuda.set_rng_state_all(tensors)
    except Exception:  # pragma: no cover (best-effort)
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


