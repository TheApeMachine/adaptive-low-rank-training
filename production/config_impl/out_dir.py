"""Output directory naming helpers.

Why this exists:
- Some harnesses rely on a predictable `runs/{exp}_{tag}` layout.
- Keeping this logic centralized avoids ad-hoc path building in runners.
"""

from __future__ import annotations

import argparse
import os

from production.selfopt_cache import as_str_object_dict


def _args_map(args: argparse.Namespace) -> dict[str, object]:
    d = as_str_object_dict(getattr(args, "__dict__", {}))
    return {} if d is None else d


def default_out_dir(args: argparse.Namespace) -> str | None:
    """Why: provide a stable default when the user doesn't pass --out-dir."""
    a = _args_map(args)
    out_dir = a.get("out_dir", None)
    if isinstance(out_dir, str) and out_dir:
        return str(out_dir)
    exp = a.get("exp", None)
    run_root = a.get("run_root", "runs")
    tag = a.get("run_tag", None)
    if not isinstance(exp, str) or not exp or exp == "paper_all":
        return None
    name = str(exp).replace("paper_", "")
    if isinstance(tag, str) and tag:
        name = f"{name}_{tag}"
    root = str(run_root) if isinstance(run_root, str) else "runs"
    return os.path.join(root, name)


