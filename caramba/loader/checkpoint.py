"""
checkpoint provides weight-loading utilities for PyTorch checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file
from torch import Tensor, nn


def _get_torch_version() -> tuple[int, int]:
    """Parse PyTorch version into (major, minor) tuple."""
    version_str = torch.__version__.split("+")[0]  # Remove build suffix like +cu118
    parts = version_str.split(".")
    return int(parts[0]), int(parts[1])


def _safe_torch_load(path: Path) -> dict[str, Tensor]:
    """Load a torch checkpoint with weights_only=True when supported.

    weights_only=True was introduced in PyTorch 2.4 and became the default in 2.6.
    This function provides backwards compatibility for older versions.
    """
    major, minor = _get_torch_version()

    # PyTorch >= 2.4 supports weights_only=True
    if (major, minor) >= (2, 4):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Fallback if weights_only causes issues
            return torch.load(path, map_location="cpu")
    else:
        # Older PyTorch versions don't support weights_only
        return torch.load(path, map_location="cpu")


class CheckpointLoader:
    """Loads state dicts from various checkpoint formats."""

    def load(self, path: Path) -> dict[str, Tensor]:
        """Load a state_dict from torch, safetensors, or sharded index files."""
        path = Path(path)
        if path.name.endswith(".index.json"):
            return self.load_sharded(path)
        if path.suffix == ".safetensors":
            return self.load_safetensors(path)
        # Use weights_only=True for safety (requires PyTorch >= 2.4)
        return _safe_torch_load(path)

    def load_mapped(
        self,
        model: nn.Module,
        state_dict: dict[str, Tensor],
        mapping: dict[str, str],
        *,
        strict: bool = True,
    ) -> None:
        """Load a state_dict into a model using key remapping."""
        if not mapping:
            raise ValueError("mapping must be non-empty")

        mapped = {dst: state_dict[src] for src, dst in mapping.items()}
        try:
            result = model.load_state_dict(mapped, strict=strict)
        except RuntimeError as e:
            # strict=True raises RuntimeError with missing/unexpected info
            raise ValueError(f"load failed: {e}") from e
        # strict=False returns (missing_keys, unexpected_keys)
        # Only raise an error when strict=True was requested
        if strict and result is not None:
            missing, unexpected = result
            if missing or unexpected:
                raise ValueError(f"load failed: missing={missing}, unexpected={unexpected}")

    def load_sharded(self, index_path: Path) -> dict[str, Tensor]:
        """Load from a sharded checkpoint index."""
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Invalid index file: missing weight_map")

        out: dict[str, Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            shard_path = index_path.parent / shard
            # Prevent infinite recursion: shards should not be index files
            if shard_path.name.endswith(".index.json"):
                raise ValueError(f"Shard {shard} is an index file, expected tensor file")
            for key, value in self.load(shard_path).items():
                if key in out:
                    raise ValueError(f"Duplicate key in shards: {key}")
                out[key] = value
        return out

    def load_safetensors(self, path: Path) -> dict[str, Tensor]:
        """Load from safetensors format."""
        return load_file(str(path), device="cpu")


class AttentionLoader:
    """Loads attention weights from teacher Q/K/V/O tensors."""

    def load(
        self,
        student: nn.Module,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        o: Tensor,
    ) -> None:
        """Load Q/K/V/O weights into student attention module.

        Supports both old attribute names (query/key/value/out) and
        new attribute names (q_proj/k_proj/v_proj/out_proj).
        """
        # Try new attribute names first
        q_proj = getattr(student, "q_proj", None)
        k_proj = getattr(student, "k_proj", None)
        v_proj = getattr(student, "v_proj", None)
        out_proj = getattr(student, "out_proj", None)

        student_class_name = type(student).__name__

        # Check all four projections for new-style names
        if q_proj is not None and k_proj is not None and v_proj is not None and out_proj is not None:
            self.copy(cast(nn.Linear, q_proj), q)
            self.copy(cast(nn.Linear, k_proj), k)
            self.copy(cast(nn.Linear, v_proj), v)
            self.copy(cast(nn.Linear, out_proj), o)
        else:
            # Fallback to old attribute names per-projection
            q_found = False
            if q_proj is not None:
                self.copy(cast(nn.Linear, q_proj), q)
                q_found = True
            elif hasattr(student, "query"):
                self.copy(cast(nn.Linear, student.query), q)  # type: ignore[attr-defined]
                q_found = True
            if not q_found:
                raise ValueError(
                    f"Could not find Q projection in {student_class_name}. "
                    "Expected 'q_proj' or 'query' attribute."
                )

            k_found = False
            if k_proj is not None:
                self.copy(cast(nn.Linear, k_proj), k)
                k_found = True
            elif hasattr(student, "key"):
                self.copy(cast(nn.Linear, student.key), k)  # type: ignore[attr-defined]
                k_found = True
            if not k_found:
                raise ValueError(
                    f"Could not find K projection in {student_class_name}. "
                    "Expected 'k_proj' or 'key' attribute."
                )

            v_found = False
            if v_proj is not None:
                self.copy(cast(nn.Linear, v_proj), v)
                v_found = True
            elif hasattr(student, "value"):
                self.copy(cast(nn.Linear, student.value), v)  # type: ignore[attr-defined]
                v_found = True
            if not v_found:
                raise ValueError(
                    f"Could not find V projection in {student_class_name}. "
                    "Expected 'v_proj' or 'value' attribute."
                )

            out_found = False
            if out_proj is not None:
                self.copy(cast(nn.Linear, out_proj), o)
                out_found = True
            elif hasattr(student, "out"):
                self.copy(cast(nn.Linear, student.out), o)  # type: ignore[attr-defined]
                out_found = True
            if not out_found:
                raise ValueError(
                    f"Could not find Out projection in {student_class_name}. "
                    "Expected 'out_proj' or 'out' attribute."
                )

    def copy(self, dst: nn.Linear, src: Tensor) -> None:
        """Copy weight tensor to destination module."""
        dst.weight.data.copy_(src)
