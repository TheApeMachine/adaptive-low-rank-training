"""Base defaults for the intent-first workflow.

Why this exists:
- The public CLI intentionally stays small, but downstream runtime expects a
  larger set of fields to exist.
- These defaults keep legacy runners stable without expanding user-facing flags.
"""

from __future__ import annotations

import argparse
import os

from production.optimizer.watch import OptimizerLike

from production.optimizer.apply_impl.args import arg


class BaseDefaults:
    """Derive stable baseline defaults that runners rely on."""

    @staticmethod
    def apply(o: OptimizerLike, *, args: argparse.Namespace) -> None:
        """Why: ensure required fields exist even when the CLI doesn't expose them."""
        if not hasattr(args, "instrument"):
            o.set("instrument", "rich" if str(o.get("mode")) == "train" else "off")
        if not hasattr(args, "live_plot"):
            o.set("live_plot", False)
        if not hasattr(args, "tb"):
            o.set("tb", False)

        # W&B defaults (paper-friendly, still overrideable via argv if present).
        # Keep CLI minimal; these are internal defaults consumed by RunLogger/WandBWriter.
        if not hasattr(args, "wandb_project"):
            o.set("wandb_project", "production")
        if not hasattr(args, "wandb_entity"):
            o.set("wandb_entity", "p4n0p71c0n")
        if not hasattr(args, "wandb_mode"):
            o.set("wandb_mode", "auto")
        if not hasattr(args, "wandb_name"):
            o.set("wandb_name", None)
        if not hasattr(args, "wandb_group"):
            o.set("wandb_group", None)
        if not hasattr(args, "wandb_tags"):
            o.set("wandb_tags", None)

        # Runtime KV cache defaults (keeps runners stable).
        for name, val in [
            ("kv_cache", "fp16"),
            ("kv_qblock", 32),
            ("kv_residual", 128),
            ("kv_decode_block", 1024),
            ("kv_fused", "auto"),
        ]:
            if not hasattr(args, name):
                o.set(name, val)

        # Sampling mode often needs an output directory for artifact/log writing.
        if (not o.get("out_dir")) and str(o.get("mode")) == "sample":
            ckpt = arg(args, "ckpt", None)
            if ckpt:
                try:
                    o.set("out_dir", os.path.dirname(str(ckpt)) or ".")
                except (OSError, ValueError, TypeError):
                    pass


