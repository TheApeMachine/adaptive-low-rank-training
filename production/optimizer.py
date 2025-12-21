from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple


_COUNT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)([kmbt])?\s*$", flags=re.IGNORECASE)
_DATASET_COUNT_RE = re.compile(r"(\d+)([bm])", flags=re.IGNORECASE)


def _parse_count(spec: Any) -> Optional[int]:
    """Parse a human count like `100m`, `1b`, `1.5b`, `2e9` → int."""
    if spec is None:
        return None
    if isinstance(spec, (int,)):
        return int(spec) if int(spec) > 0 else None
    s = str(spec).strip().lower().replace("_", "")
    if not s:
        return None
    try:
        # Accept scientific notation (e.g. 2e9).
        v = float(s)
        if math.isfinite(v) and v > 0:
            return int(v)
    except Exception:
        pass
    m = _COUNT_RE.match(s)
    if not m:
        return None
    num = float(m.group(1))
    suf = (m.group(2) or "").lower()
    mult = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}.get(suf, 1)
    v = int(num * mult)
    return v if v > 0 else None


def _format_count(n: int) -> str:
    n = int(n)
    if n % 1_000_000_000_000 == 0:
        return f"{n // 1_000_000_000_000}t"
    if n % 1_000_000_000 == 0:
        return f"{n // 1_000_000_000}b"
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def _read_tokens_from_meta(data_path: str) -> Optional[int]:
    """Read `tokens: ...` from sibling `.meta` file if present."""
    try:
        p = str(data_path)
        meta_path = p + ".meta"
        if not os.path.exists(meta_path):
            return None
        raw = open(meta_path, "r", encoding="utf-8").read()
        for line in raw.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            if k.strip().lower() == "tokens":
                return _parse_count(v.strip())
    except Exception:
        return None
    return None


def _infer_dataset_tokens(data_path: Optional[str]) -> Optional[int]:
    if not data_path:
        return None
    try:
        from production.config import infer_dataset_tokens_from_path

        inferred = infer_dataset_tokens_from_path(str(data_path))
    except Exception:
        inferred = None
    if inferred is not None:
        return int(inferred)
    return _read_tokens_from_meta(str(data_path))


def _infer_dataset_tokens_with_source(data_path: Optional[str]) -> Tuple[Optional[int], str]:
    if not data_path:
        return None, ""
    try:
        base = os.path.basename(str(data_path)).lower()
    except Exception:
        base = ""
    m = _DATASET_COUNT_RE.search(base)
    if m:
        try:
            k = int(m.group(1))
            suf = str(m.group(2)).lower()
            if suf == "m":
                return int(k * 1_000_000), "filename"
            if suf == "b":
                return int(k * 1_000_000_000), "filename"
        except Exception:
            pass
    meta = _read_tokens_from_meta(str(data_path))
    if meta is not None:
        return int(meta), "meta"
    # Fallback to production.config inference (may scan alternative conventions).
    inferred = _infer_dataset_tokens(str(data_path))
    return (int(inferred) if inferred is not None else None), ("config" if inferred is not None else "")


def _normalize_result_type(exp_or_result: str) -> str:
    s = str(exp_or_result).strip().lower()
    if not s:
        return ""
    if s.startswith("paper_"):
        s = s[len("paper_") :]
    return s


def _result_type_to_exp(result_type: str) -> str:
    rt = _normalize_result_type(result_type)
    if rt in ("baseline", "standard"):
        return "paper_baseline"
    if rt in ("bottleneck",):
        return "paper_bottleneck"
    if rt in ("decoupled",):
        return "paper_decoupled"
    if rt in ("gqa",):
        return "paper_gqa"
    # Allow passing full preset ids (including non-paper training presets).
    if rt.startswith("train_"):
        return rt
    return f"paper_{rt}" if rt else ""


def _derive_lr_from_params(target_params: int) -> float:
    # Simple, stable heuristic (can be replaced by online tuning later).
    # ~3e-4 at 1B; slightly higher for smaller models.
    ref = 1_000_000_000.0
    base = 3e-4
    exp = 0.10
    scale = (float(target_params) / ref) ** (-exp)
    return float(base * scale)


def _choose_layers(*, target_params: int, device_type: str) -> int:
    """Pick a depth that yields sane head_dim and fits common hardware profiles."""
    from production.config import derive_d_model, derive_n_head

    target_params = int(max(1, target_params))
    device_type = str(device_type or "cpu")

    if target_params < 50_000_000:
        candidates = (2, 4, 6, 8, 12)
        min_d_model = 64
        multiple = 64
    elif target_params < 500_000_000:
        candidates = (8, 12, 16, 20, 22, 24, 32)
        min_d_model = 256
        multiple = 128
    else:
        candidates = (12, 16, 20, 22, 24, 32, 48, 64, 96)
        min_d_model = 256
        multiple = 128

    # Slightly prefer shallower models on CPU/MPS to reduce overhead.
    depth_bias = 0.15 if device_type in ("cpu", "mps") else 0.0

    best_L = int(candidates[0])
    best_score = float("inf")

    for L in candidates:
        d_model = derive_d_model(layers=int(L), target_params=int(target_params), multiple=multiple, min_d_model=min_d_model)
        n_head = derive_n_head(int(d_model))
        head_dim = int(d_model) / max(1, int(n_head))

        # Prefer ~128-dim heads when possible; soft bias towards ~24 layers at scale.
        score = abs(float(head_dim) - 128.0) / 128.0
        score += depth_bias * (float(L) / 24.0)

        if score < best_score:
            best_score = float(score)
            best_L = int(L)

    return int(best_L)


@dataclass
class _Watch:
    deps: Tuple[str, ...]
    fn: Callable[["Optimizer"], None]
    name: str
    fired: bool = False


class Optimizer:
    """Dependency-driven resolver for intent → concrete runtime config.

    This is intentionally *not* a CLI surface. It takes high-level intent and derives all other
    knobs from (a) that intent and (b) the local runtime environment (hardware).

    The core mechanism is a small watcher graph: derivations can "wait" until their dependencies
    exist, then materialize new values. This keeps config derivation decoupled and order-independent.
    """

    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._watches: list[_Watch] = []
        self._pumping = False

    def has(self, key: str) -> bool:
        return str(key) in self._values

    def get(self, key: str) -> Any:
        return self._values[str(key)]

    def maybe(self, key: str, default: Any = None) -> Any:
        return self._values.get(str(key), default)

    def set(self, key: str, value: Any) -> None:
        self._values[str(key)] = value
        self._pump()

    def when_ready(self, deps: Sequence[str], fn: Callable[["Optimizer"], None], *, name: Optional[str] = None) -> None:
        self._watches.append(_Watch(tuple(str(d) for d in deps), fn, name or getattr(fn, "__name__", "watch")))
        self._pump()

    def _pump(self, *, max_iters: int = 512) -> None:
        if self._pumping:
            return
        self._pumping = True
        try:
            for _ in range(int(max_iters)):
                progressed = False
                for w in self._watches:
                    if w.fired:
                        continue
                    if all(d in self._values for d in w.deps):
                        w.fired = True
                        w.fn(self)
                        progressed = True
                if not progressed:
                    return
            raise RuntimeError("Optimizer graph did not converge (possible dependency cycle).")
        finally:
            self._pumping = False

    def apply_to_args(self, args: argparse.Namespace) -> None:
        for k, v in self._values.items():
            try:
                setattr(args, k, v)
            except Exception:
                pass


def apply_dynamic_config(args: argparse.Namespace, *, device: Any) -> None:
    """Populate the runner-required fields from intent, without expanding CLI surface area."""
    import torch

    device_type = str(getattr(device, "type", "") or "cpu")
    opt = Optimizer()

    # ---- Seed known intent ----
    mode0 = str(getattr(args, "mode", "train"))
    opt.set("mode", mode0)
    opt.set("data", getattr(args, "data", None))
    opt.set("out_dir", getattr(args, "out_dir", None))
    opt.set("seed", int(getattr(args, "seed", 1337)))
    opt.set("device_type", device_type)

    exp = getattr(args, "exp", None)
    result = getattr(args, "result", None)
    exp_source = "exp" if exp is not None else ("result" if result is not None else "")
    if exp is None and result is not None:
        exp = _result_type_to_exp(str(result))
    elif exp is not None:
        exp = _result_type_to_exp(str(exp))
    opt.set("exp", exp)
    opt.set("exp_source", exp_source)

    # Optional high-level intent.
    opt.set("size", getattr(args, "size", None))
    opt.set("layers", getattr(args, "layers", None))

    # ---- Base defaults (always) ----
    def _derive_base_defaults(o: Optimizer) -> None:
        # Always-present fields used in both train/sample paths.
        if not hasattr(args, "instrument"):
            o.set("instrument", "rich" if str(o.get("mode")) == "train" else "off")
        if not hasattr(args, "live_plot"):
            o.set("live_plot", False)
        if not hasattr(args, "tb"):
            o.set("tb", False)
        for name, val in [
            ("kv_cache", "fp16"),
            ("kv_qblock", 32),
            ("kv_residual", 128),
            ("kv_decode_block", 1024),
            ("kv_fused", "auto"),
        ]:
            if not hasattr(args, name):
                o.set(name, val)

        # Sample-mode convenience: if out_dir omitted, use ckpt dir (keeps logs/co-located).
        if (not o.get("out_dir")) and str(o.get("mode")) == "sample":
            ckpt = getattr(args, "ckpt", None)
            if ckpt:
                try:
                    o.set("out_dir", os.path.dirname(str(ckpt)) or ".")
                except Exception:
                    pass

    opt.when_ready(["mode", "out_dir"], _derive_base_defaults, name="base_defaults")

    # ---- Training derivations (watchers) ----
    if mode0 == "train":
        def _derive_dataset_tokens(o: Optimizer) -> None:
            tokens, src = _infer_dataset_tokens_with_source(o.get("data"))
            o.set("dataset_tokens", (int(tokens) if tokens is not None else None))
            o.set("dataset_tokens_source", str(src))

        opt.when_ready(["data"], _derive_dataset_tokens, name="dataset_tokens_from_data")

        def _derive_target_params(o: Optimizer) -> None:
            # Explicit model size wins. Otherwise, use a Chinchilla-style token budget.
            size_raw = o.get("size")
            size = _parse_count(size_raw)
            if size_raw is not None and size is None:
                raise ValueError(f"Unparseable --size {size_raw!r}. Use e.g. 100m, 1b, 1.5b, 2e9.")
            if size is not None:
                o.set("target_params", int(size))
                o.set("target_params_source", "size")
                return
            dtok = o.get("dataset_tokens")
            if dtok is None:
                raise ValueError(
                    "Could not infer dataset token count from --data path. "
                    "Name it like `fineweb_20b.npy`, provide a sibling `.meta` with `tokens: ...`, "
                    "or pass `--size` explicitly."
                )
            # Default compute-optimal token ratio (can be tuned later).
            tokens_per_param = 20.0
            o.set("tokens_per_param", float(tokens_per_param))
            o.set("target_params", int(max(1.0, float(dtok) / tokens_per_param)))
            o.set("target_params_source", f"dataset_tokens/{tokens_per_param:g}")

        opt.when_ready(["size", "dataset_tokens"], _derive_target_params, name="target_params")

        def _derive_layers(o: Optimizer) -> None:
            raw_layers = o.get("layers")
            try:
                if raw_layers is not None and int(raw_layers) > 0:
                    o.set("layers", int(raw_layers))
                    o.set("layers_source", "override")
                    return
            except Exception:
                pass

            # Back-compat: allow `_l{N}_` tags in out_dir.
            out_dir = o.get("out_dir")
            if out_dir:
                try:
                    from production.config import infer_layers_from_out_dir

                    inferred = infer_layers_from_out_dir(str(out_dir))
                    if inferred is not None and int(inferred) > 0:
                        o.set("layers", int(inferred))
                        o.set("layers_source", "out_dir")
                        return
                except Exception:
                    pass

            target_params = int(o.get("target_params"))
            L = _choose_layers(target_params=target_params, device_type=str(o.get("device_type")))
            o.set("layers", int(L))
            o.set("layers_source", "auto")

        opt.when_ready(["layers", "out_dir", "target_params", "device_type"], _derive_layers, name="layers")

        def _derive_training_defaults(o: Optimizer) -> None:
            # Minimal required fields for the runner. These should be treated as derived policy, not CLI.
            exp0 = o.get("exp")
            dtok = o.get("dataset_tokens")
            target_params = int(o.get("target_params"))
            L = int(o.get("layers"))

            # Data / split
            if not hasattr(args, "tokenizer"):
                data_path = str(o.get("data") or "")
                base = os.path.basename(data_path).lower()
                # FineWeb-Edu is GPT-2 BPE by contract; everything else defaults to raw integer IDs.
                tok = "tiktoken" if ("fineweb" in base or base.endswith(".tokens")) else "raw"
                o.set("tokenizer", tok)
            if not hasattr(args, "val_frac"):
                o.set("val_frac", 0.1)
            if not hasattr(args, "data_format"):
                o.set("data_format", "auto")
            if not hasattr(args, "data_dtype"):
                o.set("data_dtype", "int64")

            # Model-core knobs (let apply_intent/apply_exp_preset fill dims)
            for name, val in [
                ("layers", int(L)),
                ("d_model", 0),
                ("n_head", 0),
                ("d_ff", 0),
                ("embed_dim", 0),
                ("attn_mode", ""),
                ("attn_dim", 0),
                ("sem_dim", 0),
                ("geo_dim", 0),
                ("kv_head", None),
                ("rope_base", 10000.0),
                ("tie_qk", False),
                ("null_attn", False),
                ("no_rope", False),
                ("no_decoupled_gate", False),
                ("no_learned_temp", False),
                ("mlp", "swiglu"),
                ("dropout", 0.0),
            ]:
                if not hasattr(args, name):
                    o.set(name, val)

            # Context length heuristic: bigger datasets typically benefit from longer context, but keep
            # conservative defaults for non-CUDA devices.
            if not hasattr(args, "block"):
                if str(o.get("device_type")) == "cuda":
                    block = 2048
                elif str(o.get("device_type")) == "mps":
                    block = 512 if target_params < 50_000_000 else 1024
                else:
                    if target_params < 50_000_000:
                        block = 32
                    else:
                        block = 256 if (dtok is None or int(dtok) <= 200_000_000) else 512
                o.set("block", int(block))

            # Training knobs
            if not hasattr(args, "optimizer"):
                o.set("optimizer", "adamw")
            if not hasattr(args, "weight_decay"):
                o.set("weight_decay", 0.1)
            if not hasattr(args, "lr"):
                o.set("lr", float(_derive_lr_from_params(target_params)))
            if not hasattr(args, "min_lr"):
                o.set("min_lr", float(o.get("lr")) * 0.1)
            if not hasattr(args, "lr_schedule"):
                o.set("lr_schedule", "cosine")
            if not hasattr(args, "warmup_steps"):
                o.set("warmup_steps", 0)
            if not hasattr(args, "adam_eps"):
                o.set("adam_eps", 1e-8)
            if not hasattr(args, "adam_betas"):
                o.set("adam_betas", "0.9,0.95")
            if not hasattr(args, "lion_betas"):
                o.set("lion_betas", "0.9,0.99")
            if not hasattr(args, "opt_foreach"):
                o.set("opt_foreach", False)
            if not hasattr(args, "opt_fused"):
                o.set("opt_fused", False)
            if not hasattr(args, "eval_iters"):
                o.set("eval_iters", 20)
            if not hasattr(args, "eval_every"):
                o.set("eval_every", 0)
            if not hasattr(args, "save_every"):
                o.set("save_every", 0)
            if not hasattr(args, "log_every"):
                o.set("log_every", 0)

            # Derive out_dir if omitted.
            if not o.get("out_dir") and exp0:
                rt = _normalize_result_type(str(exp0))
                dtok_s = _format_count(int(dtok)) if dtok is not None else "data"
                p_s = _format_count(int(target_params))
                dev_s = str(o.get("device_type"))
                run_id = f"{dev_s}_{dtok_s}_{p_s}_l{int(L)}_{rt}_s{int(o.get('seed'))}"
                o.set("out_dir", os.path.join("runs", run_id))

        opt.when_ready(
            ["exp", "dataset_tokens", "target_params", "layers", "device_type", "seed", "out_dir"],
            _derive_training_defaults,
            name="train_defaults",
        )

    # Ensure all derived values are applied back onto args (and re-run intent/preset shaping).
    opt.apply_to_args(args)

    if mode0 == "train":
        # Fill architecture dims and mode-specific derived dims.
        from production.config import apply_exp_preset, apply_intent

        apply_intent(args)
        apply_exp_preset(args)

        # Post-condition: enforce that core numeric fields exist and are positive where required.
        # (We keep this intentionally lightweight; deeper validation belongs to the model code.)
        for name in ("block", "layers", "d_model", "n_head", "d_ff", "embed_dim", "attn_dim"):
            if not hasattr(args, name):
                raise RuntimeError(f"dynamic config did not set required field: {name}")
            if int(getattr(args, name)) <= 0:
                raise RuntimeError(f"dynamic config produced non-positive {name}={getattr(args, name)!r}")

    # Ensure a few always-present fields used directly in sample mode.
    if not hasattr(args, "tokenizer"):
        args.tokenizer = "tiktoken"
    if not hasattr(args, "instrument"):
        args.instrument = "off"
    if not hasattr(args, "live_plot"):
        args.live_plot = False
    if not hasattr(args, "tb"):
        args.tb = False

    # Keep exp normalized (apply_exp_preset expects canonical keys).
    if getattr(args, "exp", None):
        args.exp = _result_type_to_exp(str(getattr(args, "exp")))

    # Attach a compact summary for observability (printed by the runner).
    try:
        head_dim = None
        try:
            dm = int(getattr(args, "d_model", 0) or 0)
            nh = int(getattr(args, "n_head", 0) or 0)
            if dm > 0 and nh > 0:
                head_dim = int(dm // nh)
        except Exception:
            head_dim = None

        args._selfopt_summary = {
            "mode": str(getattr(args, "mode", "")),
            "device_type": str(device_type),
            "data": getattr(args, "data", None),
            "exp": getattr(args, "exp", None),
            "exp_source": getattr(args, "exp_source", None),
            "dataset_tokens": getattr(args, "dataset_tokens", None),
            "dataset_tokens_source": getattr(args, "dataset_tokens_source", None),
            "target_params": getattr(args, "target_params", None),
            "target_params_source": getattr(args, "target_params_source", None),
            "tokens_per_param": getattr(args, "tokens_per_param", None),
            "layers": getattr(args, "layers", None),
            "layers_source": getattr(args, "layers_source", None),
            "block": getattr(args, "block", None),
            "d_model": getattr(args, "d_model", None),
            "n_head": getattr(args, "n_head", None),
            "head_dim": head_dim,
            "d_ff": getattr(args, "d_ff", None),
            "embed_dim": getattr(args, "embed_dim", None),
            "attn_mode": getattr(args, "attn_mode", None),
            "attn_dim": getattr(args, "attn_dim", None),
            "sem_dim": getattr(args, "sem_dim", None),
            "geo_dim": getattr(args, "geo_dim", None),
            "rope": (not bool(getattr(args, "no_rope", False))),
            "tie_qk": bool(getattr(args, "tie_qk", False)),
            "null_attn": bool(getattr(args, "null_attn", False)),
            "optimizer": getattr(args, "optimizer", None),
            "lr": getattr(args, "lr", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "lr_schedule": getattr(args, "lr_schedule", None),
            "warmup_steps": getattr(args, "warmup_steps", None),
            "min_lr": getattr(args, "min_lr", None),
            "steps": getattr(args, "steps", None),
            "out_dir": getattr(args, "out_dir", None),
        }
    except Exception:
        pass
