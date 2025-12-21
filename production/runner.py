from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from production.memory_utils import device_synchronize, empty_device_cache, get_device_mem_stats


def run_single(args: argparse.Namespace, device: torch.device) -> None:
    # Make stdout/stderr line-buffered even when piped (common in IDE consoles).
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Local import so CLI --help doesn't require torch/tiktoken.
    from production.data import (
        determine_vocab_size,
        get_batch_any,
        infer_data_format,
        load_tokens_any,
        split_train_val,
    )
    from production.instrumentation import RunLogger
    from production.model import GPT, ModelConfig
    from production.runtime_tuning import KVCachePolicy, KVSelfOptConfig

    try:
        import tiktoken  # type: ignore
    except Exception:
        tiktoken = None  # type: ignore

    # Self-optimization is always enabled and non-configurable: the system chooses the policy.
    #
    # NOTE: We still persist/cache tuned plans for reuse, but callers cannot override tuning knobs.
    cache_path = None
    try:
        out_dir = str(getattr(args, "out_dir", "") or "")
        if out_dir:
            parent = os.path.dirname(out_dir.rstrip(os.sep)) or "."
            cache_path = os.path.join(parent, "selfopt_cache.json")
    except Exception:
        cache_path = None

    self_opt_cfg: Optional[KVSelfOptConfig] = KVSelfOptConfig(mode="online", scope="all", cache_path=cache_path)

    # -------------------------
    # Sample mode
    # -------------------------
    if args.mode == "sample":
        if not args.ckpt:
            raise ValueError("--ckpt is required for --mode sample")

        ckpt = torch.load(args.ckpt, map_location=device)
        cfg_dict = ckpt.get("config", None)
        if cfg_dict is None:
            raise ValueError("Checkpoint missing 'config'. Can't reconstruct model safely.")
        cfg = ModelConfig(**cfg_dict)
        model = GPT(cfg).to(device)

        incompatible = model.load_state_dict(ckpt["model"], strict=False)
        bad_missing = [k for k in incompatible.missing_keys if "decoupled_gate_logit" not in k]
        bad_unexpected = [k for k in incompatible.unexpected_keys if "decoupled_gate_logit" not in k]
        if bad_missing or bad_unexpected:
            model.load_state_dict(ckpt["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"[warn] Non-strict checkpoint load. Missing={incompatible.missing_keys} Unexpected={incompatible.unexpected_keys}")
        model.eval()

        # Prompt: either raw token IDs or text (tiktoken only)
        try:
            prompt_ids = [int(t) for t in args.prompt_tokens.strip().split()]
        except ValueError:
            if args.tokenizer != "tiktoken":
                raise ValueError("Text prompts require --tokenizer tiktoken")
            if tiktoken is None:
                raise ImportError("tiktoken needed for text prompts")
            enc = tiktoken.get_encoding("gpt2")
            prompt_ids = enc.encode_ordinary(args.prompt_tokens)

        prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)

        # Expert override: force an atomic decoupled KV cache policy from a single string.
        kv_policy_s = getattr(args, "kv_policy", None)
        if kv_policy_s:
            if str(getattr(cfg, "attn_mode", "")) != "decoupled":
                raise ValueError("--kv-policy is only supported for decoupled attention checkpoints")
            pol = KVCachePolicy.parse(str(kv_policy_s))
            # Apply as per-tensor overrides (so model.generate() stays unchanged).
            args.kv_cache_k_sem = pol.k_sem_kind
            args.kv_cache_k_geo = pol.k_geo_kind
            args.kv_cache_v = pol.v_kind
            args.kv_qblock_k_sem = int(pol.k_sem_qblock)
            args.kv_qblock_k_geo = int(pol.k_geo_qblock)
            args.kv_qblock_v = int(pol.v_qblock)
            args.kv_residual = int(pol.residual_len)

            # If selfopt is enabled, keep decode-plan tuning but disable cache-policy tuning (policy is forced).
            if self_opt_cfg is not None:
                try:
                    self_opt_cfg.scope = "decode"
                except Exception:
                    pass

        logger = None
        if args.instrument != "off" or args.live_plot or args.tb or bool(getattr(args, "wandb", False)):
            logger = RunLogger(
                args.out_dir,
                instrument=args.instrument,
                cfg=cfg,
                args=args,
                device=device,
                live_plot=bool(args.live_plot),
                tb=bool(args.tb),
                wandb=bool(getattr(args, "wandb", False)),
            )

        print(f"Generating {args.max_new_tokens} tokens...")
        try:
            if getattr(args, "draft_ckpt", None):
                dckpt = torch.load(str(args.draft_ckpt), map_location=device)
                dcfg_dict = dckpt.get("config", None)
                if dcfg_dict is None:
                    raise ValueError("Draft checkpoint missing 'config'. Can't reconstruct draft model safely.")
                dcfg = ModelConfig(**dcfg_dict)
                draft = GPT(dcfg).to(device)
                incompatible_d = draft.load_state_dict(dckpt["model"], strict=False)
                bad_missing_d = [k for k in incompatible_d.missing_keys if "decoupled_gate_logit" not in k]
                bad_unexpected_d = [k for k in incompatible_d.unexpected_keys if "decoupled_gate_logit" not in k]
                if bad_missing_d or bad_unexpected_d:
                    draft.load_state_dict(dckpt["model"], strict=True)
                if incompatible_d.missing_keys or incompatible_d.unexpected_keys:
                    print(f"[warn] Non-strict draft checkpoint load. Missing={incompatible_d.missing_keys} Unexpected={incompatible_d.unexpected_keys}")

                # Basic safety: vocab size must match for token IDs to be meaningful.
                if int(dcfg.vocab_size) != int(cfg.vocab_size):
                    raise ValueError(f"Draft vocab_size {dcfg.vocab_size} != main vocab_size {cfg.vocab_size}")

                # Match main model's inference behavior (disable dropout, etc.)
                draft.eval()

                out = model.generate_speculative(
                    prompt,
                    draft_model=draft,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=(None if args.top_k is None else int(args.top_k)),
                    kv_cache=str(args.kv_cache),
                    kv_qblock=int(args.kv_qblock),
                    kv_residual=int(args.kv_residual),
                    kv_decode_block=int(args.kv_decode_block),
                    kv_fused=str(args.kv_fused),
                    self_opt=self_opt_cfg,
                    kv_cache_k=getattr(args, "kv_cache_k", None),
                    kv_cache_v=getattr(args, "kv_cache_v", None),
                    kv_cache_k_sem=getattr(args, "kv_cache_k_sem", None),
                    kv_cache_k_geo=getattr(args, "kv_cache_k_geo", None),
                    kv_qblock_k=getattr(args, "kv_qblock_k", None),
                    kv_qblock_v=getattr(args, "kv_qblock_v", None),
                    kv_qblock_k_sem=getattr(args, "kv_qblock_k_sem", None),
                    kv_qblock_k_geo=getattr(args, "kv_qblock_k_geo", None),
                    spec_k=int(getattr(args, "spec_k", 4)),
                    spec_method=str(getattr(args, "spec_method", "reject_sampling")),
                    spec_extra_token=bool(getattr(args, "spec_extra_token", False)),
                    spec_disable_below_accept=float(getattr(args, "spec_disable_below_accept", 0.0)),
                    log_callback=(logger.log if logger is not None else None),
                )
            else:
                out = model.generate(
                    prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_k=(None if args.top_k is None else int(args.top_k)),
                    kv_cache=str(args.kv_cache),
                    kv_qblock=int(args.kv_qblock),
                    kv_residual=int(args.kv_residual),
                    kv_decode_block=int(args.kv_decode_block),
                    kv_fused=str(args.kv_fused),
                    self_opt=self_opt_cfg,
                    kv_cache_k=getattr(args, "kv_cache_k", None),
                    kv_cache_v=getattr(args, "kv_cache_v", None),
                    kv_cache_k_sem=getattr(args, "kv_cache_k_sem", None),
                    kv_cache_k_geo=getattr(args, "kv_cache_k_geo", None),
                    kv_qblock_k=getattr(args, "kv_qblock_k", None),
                    kv_qblock_v=getattr(args, "kv_qblock_v", None),
                    kv_qblock_k_sem=getattr(args, "kv_qblock_k_sem", None),
                    kv_qblock_k_geo=getattr(args, "kv_qblock_k_geo", None),
                    log_callback=(logger.log if logger is not None else None),
                )
        finally:
            if logger is not None:
                logger.close()

        out_ids = out[0].detach().to("cpu").tolist()
        if args.tokenizer == "tiktoken":
            if tiktoken is None:
                raise ImportError("tiktoken not installed")
            enc = tiktoken.get_encoding("gpt2")
            print(enc.decode(out_ids))
        else:
            print(out_ids)
        return

    # -------------------------
    # Train mode
    # -------------------------
    if args.data is None:
        raise ValueError("--data is required for --mode train")
    if args.out_dir is None:
        raise ValueError("--out-dir is required for --mode train (or provide --size + --exp for auto dirs).")

    # For tokenized FineWeb-Edu / GPT-2 BPE streams, vocab_size is known and should not require scanning data.
    if getattr(args, "vocab_size", None) is None and str(getattr(args, "tokenizer", "tiktoken")) == "tiktoken":
        try:
            args.vocab_size = 50257
        except Exception:
            pass

    vocab = getattr(args, "vocab_size", None)
    if vocab is None:
        # Fallback: determine from data by scanning.
        data_path = Path(args.data)
        fmt = infer_data_format(data_path, str(args.data_format))
        tokens_any = load_tokens_any(path=data_path, fmt=fmt, data_dtype=str(args.data_dtype))
        vocab = determine_vocab_size(tokens_any=tokens_any, vocab_size=None, tokenizer=str(args.tokenizer))
        n_total = int(tokens_any.numel()) if isinstance(tokens_any, torch.Tensor) else int(len(tokens_any))
        train_view, val_view = split_train_val(tokens_any, val_frac=float(args.val_frac))
    else:
        data_path = Path(args.data)
        fmt = infer_data_format(data_path, str(args.data_format))
        tokens_any = None
        train_view = None
        val_view = None

    cfg = ModelConfig(
        vocab_size=int(vocab),
        block_size=int(args.block),
        n_layer=int(args.layers),
        n_head=int(args.n_head),
        kv_head=args.kv_head,
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        embed_dim=int(args.embed_dim),
        attn_mode=str(args.attn_mode),
        attn_dim=int(args.attn_dim),
        sem_dim=int(args.sem_dim),
        geo_dim=int(args.geo_dim),
        decoupled_gate=(not args.no_decoupled_gate),
        rope=(not args.no_rope),
        rope_base=float(args.rope_base),
        tie_qk=bool(args.tie_qk),
        null_attn=bool(args.null_attn),
        learned_temp=(not args.no_learned_temp),
        mlp=str(args.mlp),
        dropout=float(args.dropout),
    )

    # Always write resolved config for reproducibility and harness validation.
    try:
        os.makedirs(str(args.out_dir), exist_ok=True)
        Path(os.path.join(str(args.out_dir), "resolved_config.json")).write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass

    # Compact visibility into "intent → derived config" (minimal CLI by design, so we print this).
    try:
        def _fmt_int(x: Any) -> str:
            try:
                return f"{int(x):_d}"
            except Exception:
                return "?"

        def _fmt_float(x: Any) -> str:
            try:
                return f"{float(x):.3g}"
            except Exception:
                return "?"

        summ = getattr(args, "_selfopt_summary", None)
        exp_s = str(getattr(args, "exp", None) or "")
        data_s = str(getattr(args, "data", None) or "")
        out_s = str(getattr(args, "out_dir", None) or "")

        ds_tok = getattr(args, "dataset_tokens", None)
        tp = getattr(args, "target_params", None)
        ds_src = getattr(args, "dataset_tokens_source", None)
        tp_src = getattr(args, "target_params_source", None)
        layers_src = getattr(args, "layers_source", None)
        exp_src = getattr(args, "exp_source", None)

        head_dim = None
        try:
            head_dim = int(cfg.d_model // max(1, int(cfg.n_head)))
        except Exception:
            head_dim = None

        print(
            f"[intent] device={str(device)} exp={exp_s or '?'} ({exp_src or '?'}) data={os.path.basename(data_s) or '?'} "
            f"dataset_tokens={_fmt_int(ds_tok)} ({ds_src or '?'}) target_params={_fmt_int(tp)} ({tp_src or '?'}) "
            f"layers={int(cfg.n_layer)} ({layers_src or '?'}) out_dir={out_s or '?'}",
            flush=True,
        )
        print(
            f"[model] block={int(cfg.block_size)} d_model={int(cfg.d_model)} n_head={int(cfg.n_head)} head_dim={_fmt_int(head_dim)} "
            f"d_ff={int(cfg.d_ff)} embed_dim={int(cfg.embed_dim)}",
            flush=True,
        )
        print(
            f"[attn] mode={str(cfg.attn_mode)} attn_dim={int(cfg.attn_dim)} sem_dim={int(cfg.sem_dim)} geo_dim={int(cfg.geo_dim)} "
            f"rope={int(bool(cfg.rope))} tie_qk={int(bool(cfg.tie_qk))} null_attn={int(bool(cfg.null_attn))}",
            flush=True,
        )
        print(
            f"[traincfg] steps={getattr(args, 'steps', None)} lr={_fmt_float(getattr(args, 'lr', None))} "
            f"wd={_fmt_float(getattr(args, 'weight_decay', None))} opt={str(getattr(args, 'optimizer', ''))} "
            f"sched={str(getattr(args, 'lr_schedule', ''))} warmup={_fmt_int(getattr(args, 'warmup_steps', None))} "
            f"min_lr={_fmt_float(getattr(args, 'min_lr', None))}",
            flush=True,
        )
        try:
            if self_opt_cfg is None:
                # Shouldn't happen (selfopt is always enabled), but keep logs robust.
                print("[selfopt] mode=startup", flush=True)
            else:
                print(
                    f"[selfopt] mode={str(getattr(self_opt_cfg, 'mode', ''))} scope={str(getattr(self_opt_cfg, 'scope', ''))} "
                    f"cache={str(getattr(self_opt_cfg, 'cache_path', '') or '')}",
                    flush=True,
                )
        except Exception:
            pass
    except Exception:
        pass

    # Config-only fast path (used by the paper harness): avoid loading/scanning the dataset.
    # NOTE: `--steps <0` means AUTO and should *train*; only `--steps 0` is validate-only.
    if int(getattr(args, "steps", 0)) == 0:
        return

    # If we didn't load tokens earlier, load now for training.
    if tokens_any is None:
        try:
            print(f"[data] loading tokens: {str(data_path)} (format={fmt})", flush=True)
        except Exception:
            pass
        tokens_any = load_tokens_any(path=data_path, fmt=fmt, data_dtype=str(args.data_dtype))
        n_total = int(tokens_any.numel()) if isinstance(tokens_any, torch.Tensor) else int(len(tokens_any))
        train_view, val_view = split_train_val(tokens_any, val_frac=float(args.val_frac))
        try:
            print(f"[data] ready: n_tokens={n_total} train={len(train_view)} val={len(val_view)}", flush=True)
        except Exception:
            pass

    model = GPT(cfg).to(device)

    # Parameter dtype.
    def _supports_dtype(dev: torch.device, dt: torch.dtype) -> bool:
        try:
            x = torch.ones(8, device=dev, dtype=dt)
            y = (x * 1.0001).sum()
            _ = float(y.detach().to("cpu").item())
            return True
        except Exception:
            return False

    # Always-on self-optimizer: derive dtype/AMP/batch/compile policy (no user overrides).
    from production.selfopt_controller import SelfOptController

    # Dataset feasibility caps (split-aware): runtime seq lengths must fit both splits.
    train_seq_cap = int(max(2, int(len(train_view)) - 2))
    eval_seq_cap = int(max(2, int(len(val_view)) - 2))

    # Decision log is per-run (reproducible record of what the optimizer chose).
    selfopt_log_path = None
    try:
        selfopt_log_path = os.path.join(str(args.out_dir), "selfopt_decisions.jsonl")
    except Exception:
        selfopt_log_path = None

    controller = SelfOptController(
        cache_path=getattr(self_opt_cfg, "cache_path", None),
        log_path=selfopt_log_path,
        device=device,
        cfg=cfg,
    )
    model, runtime_plan = controller.plan_runtime(
        model=model,
        train_view=train_view,
        val_view=val_view,
        get_batch=lambda bs, sl: get_batch_any(train_view, batch_size=int(bs), block_size=int(sl), device=device),
        train_seq_len_cap=train_seq_cap,
        eval_seq_len_cap=eval_seq_cap,
    )

    # Apply plan to args for downstream logging/scheduling.
    try:
        args.train_seq_len = int(runtime_plan.train_seq_len)
        args.eval_seq_len = int(runtime_plan.eval_seq_len)
    except Exception:
        pass
    try:
        by_seq = runtime_plan.batch_plan.by_seq
        args.batch_by_seq = ",".join([f"{int(k)}:{int(v[0])}x{int(v[1])}" for k, v in sorted(by_seq.items())])
        bs0, ga0 = by_seq.get(int(runtime_plan.train_seq_len), next(iter(by_seq.values())))
        args.batch_size = int(bs0)
        args.grad_accum = int(ga0)
    except Exception:
        pass
    try:
        args.compile = bool(getattr(runtime_plan.compile_plan, "enabled", False))
        args.compile_mode = str(getattr(runtime_plan.compile_plan, "mode", "reduce-overhead"))
    except Exception:
        pass
    try:
        pd = runtime_plan.param_dtype
        args.param_dtype = ("bf16" if pd == torch.bfloat16 else ("fp16" if pd == torch.float16 else "fp32"))
    except Exception:
        pass

    amp_state = {"enabled": bool(runtime_plan.amp_enabled), "dtype": runtime_plan.amp_dtype}
    try:
        args.amp = bool(amp_state["enabled"])
        args.amp_dtype = ("bf16" if amp_state["dtype"] == torch.bfloat16 else "fp16")
    except Exception:
        pass
    amp_enabled = bool(amp_state["enabled"])
    amp_dtype = amp_state["dtype"]
    amp_disabled_due_to_nonfinite = False

    try:
        # Start with checkpointing disabled; enable only if we hit OOM at small batches.
        args.grad_checkpoint = False
        model.grad_checkpointing = False
    except Exception:
        pass

    @contextlib.contextmanager
    def autocast_ctx():
        if not bool(amp_state["enabled"]):
            yield
            return
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=amp_state["dtype"]):
                yield
            return
        if device.type == "mps":
            with torch.autocast("mps", dtype=amp_state["dtype"]):
                yield
            return
        with torch.autocast("cpu", dtype=torch.bfloat16):
            yield

    # Optional GradScaler (CUDA fp16 only). BF16 does not require scaling.
    scaler = None
    try:
        if bool(amp_state["enabled"]) and device.type == "cuda" and amp_state["dtype"] == torch.float16:
            from torch.cuda.amp import GradScaler  # type: ignore

            scaler = GradScaler()
    except Exception:
        scaler = None

    # Optimizer
    def _parse_two_floats(s: str, default: Tuple[float, float]) -> Tuple[float, float]:
        try:
            a, b = str(s).split(",")
            return float(a), float(b)
        except Exception:
            return default

    class Lion(torch.optim.Optimizer):
        def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
            if lr <= 0.0:
                raise ValueError(f"Invalid lr: {lr}")
            b1, b2 = betas
            if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
                raise ValueError(f"Invalid betas: {betas}")
            defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):  # type: ignore
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = float(group["lr"])
                wd = float(group.get("weight_decay", 0.0))
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("Lion does not support sparse gradients.")
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(g, alpha=(1.0 - beta1))
                    p.add_(exp_avg.sign(), alpha=-lr)
                    exp_avg.mul_(beta2).add_(g, alpha=(1.0 - beta2))
            return loss

    opt_name = str(args.optimizer)
    if opt_name == "lion":
        lion_betas = _parse_two_floats(str(args.lion_betas), (0.9, 0.99))
        opt = Lion(model.parameters(), lr=float(args.lr), betas=lion_betas, weight_decay=float(args.weight_decay))
    else:
        adam_betas = _parse_two_floats(str(args.adam_betas), (0.9, 0.95))
        base_opt_kwargs: Dict[str, Any] = dict(
            lr=float(args.lr),
            betas=adam_betas,
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )

        # Optimizer backend is a pure performance knob; choose automatically unless explicitly requested.
        want_foreach = bool(getattr(args, "opt_foreach", False))
        want_fused = bool(getattr(args, "opt_fused", False))
        if (not want_foreach) and (not want_fused):
            if device.type == "cuda":
                want_fused = True
            else:
                want_foreach = True

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        if want_fused:
            candidates.append(("fused", {**base_opt_kwargs, "fused": True}))
        if want_foreach:
            candidates.append(("foreach", {**base_opt_kwargs, "foreach": True}))
        candidates.append(("default", dict(base_opt_kwargs)))

        opt = None
        last_err: Optional[BaseException] = None
        for name, kw in candidates:
            try:
                opt = torch.optim.AdamW(model.parameters(), **kw)
                try:
                    args.opt_fused = bool(name == "fused")
                    args.opt_foreach = bool(name == "foreach")
                except Exception:
                    pass
                break
            except Exception as e:
                last_err = e
                continue
        if opt is None:
            raise RuntimeError(f"Could not construct AdamW optimizer (last error: {last_err})") from last_err

    def _first_nonfinite_grad(model: torch.nn.Module) -> Optional[str]:
        """Return a short description of the first parameter with a non-finite grad, if any."""
        try:
            for name, p in model.named_parameters():
                g = p.grad
                if g is None:
                    continue
                finite = torch.isfinite(g.detach())
                if bool(finite.all()):
                    continue
                # keep it cheap: just a few summary stats
                g_det = g.detach()
                num = int(g_det.numel())
                n_finite = int(finite.sum().to("cpu").item()) if num > 0 else 0
                # avoid nan in max by filtering if possible
                try:
                    max_abs = float(torch.nan_to_num(g_det.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().to("cpu").item())
                except Exception:
                    max_abs = float("nan")
                return f"{name} grad dtype={g_det.dtype} finite={n_finite}/{num} max|g|~{max_abs:.3g}"
        except Exception:
            return None
        return None

    def _clip_grad_norm_fp32(params, max_norm: float) -> torch.Tensor:
        """MPS-friendly grad clipping: accumulate norm in fp32 to avoid bf16 overflow in norm."""
        grads: List[torch.Tensor] = []
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            grads.append(g)
        if not grads:
            return torch.zeros([], device=device, dtype=torch.float32)
        total_sq = torch.zeros([], device=grads[0].device, dtype=torch.float32)
        for g in grads:
            gd = g.detach()
            if not torch.isfinite(gd).all():
                return torch.tensor(float("nan"), device=gd.device, dtype=torch.float32)
            total_sq = total_sq + (gd.float() * gd.float()).sum()
        total_norm = torch.sqrt(total_sq)
        # scale in-place
        denom = total_norm + 1e-6
        clip_coef = float(max_norm) / float(denom.to("cpu").item())
        if clip_coef < 1.0:
            for g in grads:
                g.mul_(clip_coef)
        return total_norm

    def lr_for_step(step: int, *, base_lr: float, total_steps: int, schedule: str, warmup_steps: int = 0, min_lr: float = 0.0) -> float:
        schedule = str(schedule).lower()
        total_steps = max(int(total_steps), 1)
        warmup_steps = max(int(warmup_steps), 0)
        if schedule == "constant":
            if warmup_steps > 0 and step < warmup_steps:
                return base_lr * (step + 1) / warmup_steps
            return base_lr
        if schedule == "cosine":
            if warmup_steps > 0 and step < warmup_steps:
                return base_lr * (step + 1) / warmup_steps
            denom = max(total_steps - warmup_steps, 1)
            t = (step - warmup_steps) / denom
            t = min(max(t, 0.0), 1.0)
            return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
        return base_lr

    # Logger
    want_logger = (
        str(args.instrument) != "off"
        or bool(getattr(args, "live_plot", False))
        or bool(getattr(args, "tb", False))
        or bool(getattr(args, "wandb", False))
    )
    logger = (
        RunLogger(
            str(args.out_dir),
            instrument=str(args.instrument),
            cfg=cfg,
            args=args,
            device=device,
            live_plot=bool(args.live_plot),
            tb=bool(args.tb),
            wandb=bool(getattr(args, "wandb", False)),
        )
        if want_logger
        else None
    )

    best_val = float("inf")
    last_step = 0
    t_start = time.time()
    tok_count = 0

    def _is_oom_error(e: BaseException) -> bool:
        msg = str(e).lower()
        return (
            ("out of memory" in msg)
            or ("cuda error: out of memory" in msg)
            or ("cudnn error: out of memory" in msg)
            or ("mps backend out of memory" in msg)
            or ("resource exhausted" in msg)
        )

    def _parse_seq_schedule(spec: Optional[str]) -> Optional[List[Tuple[int, int]]]:
        if spec is None:
            return None
        spec = str(spec).strip()
        if not spec:
            return None
        pairs: List[Tuple[int, int]] = []
        for part in spec.split(","):
            part = part.strip()
            if not part or "@" not in part:
                continue
            a, b = part.split("@", 1)
            try:
                seq = int(a)
                st = int(b)
                pairs.append((st, seq))
            except Exception:
                continue
        pairs.sort(key=lambda x: x[0])
        return pairs if pairs else None

    def _seq_len_for_step(step_idx: int, *, default_seq_len: int, schedule: Optional[List[Tuple[int, int]]]) -> int:
        if not schedule:
            return int(default_seq_len)
        s = int(default_seq_len)
        for st, ln in schedule:
            if int(step_idx) >= int(st):
                s = int(ln)
            else:
                break
        return int(s)

    def _parse_bs_ga_pair(s: str) -> Optional[Tuple[int, int]]:
        s = str(s).strip().lower()
        if not s:
            return None
        # Accept separators: "64x1", "64*1"
        for sep in ("x", "*"):
            if sep in s:
                a, b = s.split(sep, 1)
                try:
                    bs = int(a.strip())
                    ga = int(b.strip())
                    if bs > 0 and ga > 0:
                        return bs, ga
                except Exception:
                    return None
        return None

    def _parse_batch_schedule(spec: Optional[str]) -> Optional[List[Tuple[int, int, int]]]:
        """'64x1@0,32x2@200' -> [(0,64,1),(200,32,2)] sorted by step."""
        if spec is None:
            return None
        spec = str(spec).strip()
        if not spec:
            return None
        out: List[Tuple[int, int, int]] = []
        for part in spec.split(","):
            part = part.strip()
            if not part or "@" not in part:
                continue
            lhs, rhs = part.split("@", 1)
            pair = _parse_bs_ga_pair(lhs)
            if pair is None:
                continue
            try:
                st = int(rhs.strip())
            except Exception:
                continue
            bs, ga = pair
            out.append((st, bs, ga))
        out.sort(key=lambda t: t[0])
        return out if out else None

    def _batch_for_step(step_idx: int, schedule: Optional[List[Tuple[int, int, int]]], *, default_bs: int, default_ga: int) -> Tuple[int, int]:
        if not schedule:
            return int(default_bs), int(default_ga)
        bs = int(default_bs)
        ga = int(default_ga)
        for st, b, g in schedule:
            if int(step_idx) >= int(st):
                bs = int(b)
                ga = int(g)
            else:
                break
        return int(max(1, bs)), int(max(1, ga))

    def _parse_batch_by_seq(spec: Optional[str]) -> Optional[Dict[int, Tuple[int, int]]]:
        """'512:64x1,1024:32x2' -> {512:(64,1), 1024:(32,2)}"""
        if spec is None:
            return None
        spec = str(spec).strip()
        if not spec:
            return None
        out: Dict[int, Tuple[int, int]] = {}
        for part in spec.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            a, b = part.split(":", 1)
            try:
                seq = int(a.strip())
            except Exception:
                continue
            pair = _parse_bs_ga_pair(b)
            if pair is None:
                continue
            bs, ga = pair
            out[int(seq)] = (int(bs), int(ga))
        return out if out else None

    def _batch_for_seq(seq_len: int, mapping: Optional[Dict[int, Tuple[int, int]]], *, default_bs: int, default_ga: int) -> Tuple[int, int]:
        """Conservative choice: pick mapping for the smallest seq_key >= current seq_len (or max key)."""
        if not mapping:
            return int(default_bs), int(default_ga)
        keys = sorted(int(k) for k in mapping.keys())
        chosen_key = keys[-1]
        for k in keys:
            if int(k) >= int(seq_len):
                chosen_key = int(k)
                break
        bs, ga = mapping[chosen_key]
        return int(max(1, bs)), int(max(1, ga))

    def estimate_loss(eval_iters: int) -> Tuple[float, float]:
        model.eval()
        losses_tr: List[float] = []
        losses_va: List[float] = []
        eval_seq = int(getattr(args, "eval_seq_len", 0) or 0)
        bs = int(getattr(args, "batch_size", 1))
        if eval_seq <= 0:
            eval_seq = int(getattr(args, "train_seq_len", 0) or 0)
        if eval_seq <= 0:
            eval_seq = int(args.block)
        eval_seq = int(min(max(2, eval_seq), int(cfg.block_size)))

        def _loss_from_features(feats: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Compute CE loss without materializing full (B,T,V) logits at once.
            # This is crucial on MPS where huge logits tensors can be unstable.
            B, T, _D = feats.shape
            _ = B
            Wt = model.tok_emb.weight.t()  # (D, V)
            V = int(Wt.shape[1])
            loss_sum = torch.zeros([], device=feats.device, dtype=torch.float32)
            # Keep chunks small enough for MPSGraph stability.
            chunk = 64 if feats.device.type == "mps" else 256
            chunk = int(max(1, min(int(T), int(chunk))))
            for i in range(0, int(T), int(chunk)):
                j = min(int(T), i + int(chunk))
                lg = feats[:, i:j, :] @ Wt  # (B, chunk, V)
                loss_sum = loss_sum + F.cross_entropy(lg.reshape(-1, V), y[:, i:j].reshape(-1), reduction="sum").float()
            denom = int(y.numel()) if int(y.numel()) > 0 else 1
            return loss_sum / float(denom)

        for _ in range(int(eval_iters)):
            xb, yb = get_batch_any(train_view, batch_size=bs, block_size=eval_seq, device=device)
            with autocast_ctx():
                if device.type == "mps":
                    feats, _ = model(xb, return_features=True)
                    loss = _loss_from_features(feats, yb)
                else:
                    logits, _ = model(xb)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses_tr.append(float(loss.detach().to("cpu").item()))
            xb, yb = get_batch_any(val_view, batch_size=bs, block_size=eval_seq, device=device)
            with autocast_ctx():
                if device.type == "mps":
                    feats, _ = model(xb, return_features=True)
                    loss = _loss_from_features(feats, yb)
                else:
                    logits, _ = model(xb)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses_va.append(float(loss.detach().to("cpu").item()))
        model.train()
        return float(sum(losses_tr) / max(1, len(losses_tr))), float(sum(losses_va) / max(1, len(losses_va)))

    # -------------------------
    # AUTO: steps / seq_schedule / batch plan
    # -------------------------
    # Any time a value would have been provided by an "expert flag", we treat it as AUTO unless
    # explicitly set in the harness (and even then, the runner may override for safety).
    #
    # This is the core "self-optimizing" contract: intent in, runtime plan out.
    try:
        from production.config import infer_dataset_tokens_from_path
    except Exception:
        infer_dataset_tokens_from_path = None  # type: ignore

    # Treat missing/zero as AUTO for these.
    raw_steps = int(getattr(args, "steps", -1) if getattr(args, "steps", None) is not None else -1)
    raw_bs = int(getattr(args, "batch_size", 0) if getattr(args, "batch_size", None) is not None else 0)
    raw_ga = int(getattr(args, "grad_accum", 0) if getattr(args, "grad_accum", None) is not None else 0)
    raw_seq_sched = getattr(args, "seq_schedule", None) if hasattr(args, "seq_schedule") else None

    want_auto_steps = bool(raw_steps < 0)
    want_auto_batch = bool(raw_bs <= 0 or raw_ga <= 0)
    want_auto_sched = bool(raw_seq_sched in (None, "", "auto"))

    # Build stage seq_lens from the resolved runtime training context.
    # This must satisfy the dataset constraints; `args.train_seq_len` is derived above after splitting.
    stage_lens: List[int] = []
    max_train_seq = int(getattr(args, "train_seq_len", 0) or 0)
    if max_train_seq <= 0:
        max_train_seq = int(cfg.block_size)
    max_train_seq = int(min(max_train_seq, int(cfg.block_size)))

    for cand in (1024, 2048, int(max_train_seq)):
        if cand <= int(max_train_seq):
            stage_lens.append(int(cand))
    stage_lens = sorted({int(x) for x in stage_lens if int(x) > 0})
    if not stage_lens:
        stage_lens = [int(max(2, int(max_train_seq)))]

    # If training selfopt is enabled, compute an AUTO batch plan for the stage seqs.
    # This is used both for training throughput and for deriving steps/schedules.
    auto_plan: Optional[Dict[int, Tuple[int, int]]] = _parse_batch_by_seq(getattr(args, "batch_by_seq", None))
    if auto_plan:
        try:
            parts = [f"{k}:{v[0]}x{v[1]}" for k, v in sorted(auto_plan.items())]
            print(f"[selfopt][train] startup plan: " + ", ".join(parts), flush=True)
        except Exception:
            pass
    elif (self_opt_cfg is not None) and str(getattr(self_opt_cfg, "mode", "none")) != "none" and (want_auto_batch or want_auto_steps or want_auto_sched):
        # Should be rare: the controller normally pre-populates `batch_by_seq`.
        try:
            from production.train_tuning import tune_batch_by_seq

            plan = tune_batch_by_seq(
                cache_path=getattr(self_opt_cfg, "cache_path", None),
                device=device,
                cfg=cfg,
                model=model,
                get_batch=lambda bs, sl: get_batch_any(train_view, batch_size=int(bs), block_size=int(sl), device=device),
                seq_lens=list(stage_lens),
                target_gbs=0,
                warmup=1,
                iters=2,
                verbose=False,
                amp_enabled=bool(amp_enabled),
                amp_dtype=amp_dtype,
            )
            auto_plan = {int(k): (int(v[0]), int(v[1])) for k, v in plan.by_seq.items()}
        except Exception:
            auto_plan = None

    # Derive steps and seq_schedule from dataset token budget + derived batch plan.
    #
    # NOTE: We target *train* tokens (i.e., excluding the validation split) so that AUTO steps corresponds
    # more directly to "1 epoch" over the training set.
    if (want_auto_steps or want_auto_sched) and (infer_dataset_tokens_from_path is not None):
        dataset_tokens_total = None
        try:
            dataset_tokens_total = getattr(args, "dataset_tokens", None)
        except Exception:
            dataset_tokens_total = None
        if dataset_tokens_total is None:
            try:
                dataset_tokens_total = infer_dataset_tokens_from_path(str(args.data))
            except Exception:
                dataset_tokens_total = None

        if dataset_tokens_total is not None and int(dataset_tokens_total) > 0:
            val_frac = float(getattr(args, "val_frac", 0.1) or 0.1)
            val_frac = float(min(0.5, max(0.0, val_frac)))
            dataset_tokens_train = int(max(1.0, float(dataset_tokens_total) * (1.0 - float(val_frac))))

            # Token fractions per stage (in token-space, not step-space).
            if len(stage_lens) >= 3:
                fracs = [0.25, 0.35, 0.40]
            elif len(stage_lens) == 2:
                fracs = [0.40, 0.60]
            else:
                fracs = [1.0]

            # Compute tokens-per-step per stage from the (auto) plan.
            tps: List[int] = []
            for s in stage_lens:
                if auto_plan and int(s) in auto_plan:
                    bs_s, ga_s = auto_plan[int(s)]
                else:
                    bs_s, ga_s = (max(1, raw_bs) if raw_bs > 0 else 1), (max(1, raw_ga) if raw_ga > 0 else 1)
                tps.append(int(bs_s) * int(ga_s) * int(s))

            steps_by_stage: List[int] = []
            for frac, tok_per_step in zip(fracs, tps):
                tok_target = int(max(1.0, float(dataset_tokens_train) * float(frac)))
                steps_by_stage.append(int(math.ceil(float(tok_target) / max(1.0, float(tok_per_step)))))

            total_steps_auto = int(sum(int(x) for x in steps_by_stage))
            # Clamp absurdly small totals.
            total_steps_auto = int(max(1, total_steps_auto))

            if want_auto_steps:
                try:
                    args.steps = int(total_steps_auto)
                except Exception:
                    pass

            if want_auto_sched:
                # Build schedule boundaries in optimizer-step space.
                starts: List[int] = [0]
                cur = 0
                for st in steps_by_stage[:-1]:
                    cur += int(st)
                    starts.append(int(cur))
                parts = [f"{int(stage_lens[i])}@{int(starts[i])}" for i in range(len(stage_lens))]
                sched = ",".join(parts)
                try:
                    args.seq_schedule = str(sched)
                except Exception:
                    pass

            try:
                parts_dbg: List[str] = []
                for i, s in enumerate(stage_lens):
                    tok_per_step = int(tps[i]) if i < len(tps) else 0
                    st = int(steps_by_stage[i]) if i < len(steps_by_stage) else 0
                    gbs = None
                    if auto_plan and int(s) in auto_plan:
                        bs_s, ga_s = auto_plan[int(s)]
                        gbs = int(bs_s) * int(ga_s)
                    parts_dbg.append(
                        f"{int(s)}:tok/step={tok_per_step} steps={st}" + ("" if gbs is None else f" gbs={gbs}")
                    )
                seq_sched_s = getattr(args, "seq_schedule", None)
                print(
                    f"[selfopt][train] auto steps={int(getattr(args, 'steps', total_steps_auto))} "
                    f"(train_tokens={int(dataset_tokens_train)} total_tokens={int(dataset_tokens_total)} val_frac={val_frac:g}): "
                    + ", ".join(parts_dbg)
                    + ("" if not seq_sched_s else f" seq_schedule={seq_sched_s}"),
                    flush=True,
                )
            except Exception:
                pass

    # Materialize defaults for the training loop from the derived plan.
    if want_auto_batch and auto_plan:
        base_s = int(stage_lens[0])
        if base_s in auto_plan:
            bs0, ga0 = auto_plan[base_s]
            try:
                args.batch_size = int(bs0)
                args.grad_accum = int(ga0)
            except Exception:
                pass

    # Derive eval/save cadence if missing/auto (0).
    try:
        if int(getattr(args, "eval_every", 0) or 0) <= 0:
            # About ~1000 evals max; cap cost.
            args.eval_every = int(max(1000, min(5000, int(getattr(args, "steps", 1)) // 1000)))
        if int(getattr(args, "save_every", 0) or 0) <= 0:
            args.save_every = int(max(2000, int(getattr(args, "eval_every", 1000))))
        if int(getattr(args, "log_every", 0) or 0) <= 0:
            args.log_every = int(max(50, int(getattr(args, "eval_every", 1000)) // 10))
    except Exception:
        pass

    # Training schedule: interpret schedule steps in optimizer-step space (v29/v30 semantics).
    seq_schedule = _parse_seq_schedule(getattr(args, "seq_schedule", None))
    batch_schedule = _parse_batch_schedule(getattr(args, "batch_schedule", None))
    batch_by_seq = _parse_batch_by_seq(getattr(args, "batch_by_seq", None))
    base_seq_len = int(getattr(args, "train_seq_len", 0) or 0)
    if base_seq_len <= 0:
        base_seq_len = int(cfg.block_size)
    base_seq_len = int(min(max(2, base_seq_len), int(cfg.block_size)))

    grad_accum_default = max(1, int(getattr(args, "grad_accum", 0) or 0))
    micro_bs_default = max(1, int(getattr(args, "batch_size", 0) or 0))

    # Batch policy comes from the controller via `args.batch_by_seq`; just apply the base default.
    if batch_by_seq is not None and int(base_seq_len) in batch_by_seq:
        micro_bs_default, grad_accum_default = batch_by_seq[int(base_seq_len)]

    # Timing accumulators for throughput measurement (optimizer steps)
    dt_acc = 0.0
    tok_acc = 0
    fwd_acc = 0.0
    bwd_acc = 0.0
    opt_acc = 0.0
    steps_in_acc = 0

    # Non-finite handling: default to a self-healing policy (reduce LR and skip the step).
    nan_policy = str(getattr(args, "nan_policy", "reduce_lr")).lower()
    if nan_policy not in ("reduce_lr", "skip", "error"):
        nan_policy = "reduce_lr"
    nan_lr_decay = float(getattr(args, "nan_lr_decay", 0.5) or 0.5)
    try:
        args.nan_policy = str(nan_policy)
        args.nan_lr_decay = float(nan_lr_decay)
    except Exception:
        pass
    lr_mult = 1.0
    consecutive_nonfinite = 0
    max_consecutive_nonfinite = 8  # hard stop to avoid burning time on a broken config/backend

    model.train()
    opt.zero_grad(set_to_none=True)

    legacy_micro = bool(getattr(args, "legacy_micro_steps", False))
    if legacy_micro:
        print("[warn] --legacy-micro-steps enabled: interpreting --steps as micro-steps (legacy behavior).")
        if batch_schedule or batch_by_seq:
            raise ValueError("--batch-schedule/--batch-by-seq are defined in optimizer-step space and are incompatible with --legacy-micro-steps.")

    total_opt_steps = int(args.steps) if not legacy_micro else int(math.ceil(int(args.steps) / max(1, grad_accum_default)))

    # Self-optimizing batch caps (learned online via OOM events). Keys are seq_len.
    auto_bs_cap_by_seq: Dict[int, int] = {}
    warned_auto_caps: set[int] = set()
    warned_mps_logits: set[int] = set()

    # -------------------------
    # Resume support (paper-critical)
    # -------------------------
    resume_path = None
    if bool(getattr(args, "resume", False)):
        resume_path = os.path.join(str(args.out_dir), "last.pt")
    if getattr(args, "resume_path", None):
        resume_path = str(getattr(args, "resume_path"))

    start_opt_step = 1
    if resume_path:
        if not os.path.exists(str(resume_path)):
            raise FileNotFoundError(f"--resume requested but checkpoint not found: {resume_path}")
        ck = torch.load(str(resume_path), map_location=device)
        if not isinstance(ck, dict) or "model" not in ck:
            raise ValueError(f"Invalid resume checkpoint (expected dict with 'model'): {resume_path}")

        # Strict config match by default (paper safety).
        ck_cfg = ck.get("config", None)
        if ck_cfg is not None:
            want = asdict(cfg)
            if isinstance(ck_cfg, dict) and ck_cfg != want and (not bool(getattr(args, "resume_allow_config_mismatch", False))):
                raise ValueError(
                    "Resume config mismatch. Refusing to resume because this can silently invalidate experiments. "
                    "Pass --resume-allow-config-mismatch to override.\n"
                    f"checkpoint={resume_path}"
                )

        try:
            model.load_state_dict(ck["model"], strict=True)
        except Exception:
            model.load_state_dict(ck["model"], strict=False)

        if "opt" not in ck:
            raise ValueError(
                f"Resume checkpoint does not contain optimizer state ('opt'). "
                f"Re-run with a checkpoint produced by the resumable runner. checkpoint={resume_path}"
            )
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:
            if bool(getattr(args, "resume_allow_config_mismatch", False)):
                print(f"[resume] optimizer state incompatible; continuing with fresh optimizer state: {e}")
            else:
                raise

        if scaler is not None and ck.get("scaler", None) is not None:
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception:
                pass

        try:
            best_val = float(ck.get("best_val", best_val))
        except Exception:
            pass
        try:
            lr_mult = float(ck.get("lr_mult", lr_mult))
        except Exception:
            pass
        try:
            auto_bs_cap_by_seq = dict(ck.get("auto_bs_cap_by_seq", auto_bs_cap_by_seq))
        except Exception:
            pass

        try:
            last_done = int(ck.get("opt_step", 0))
            start_opt_step = int(last_done) + 1
        except Exception:
            start_opt_step = 1

        # RNG restore (best-effort).
        try:
            rng = ck.get("rng", None)
            if isinstance(rng, dict):
                import random

                if "py_random" in rng:
                    try:
                        random.setstate(rng["py_random"])
                    except Exception:
                        pass
                if "torch_cpu" in rng:
                    try:
                        torch.set_rng_state(rng["torch_cpu"])
                    except Exception:
                        pass
                if device.type == "cuda" and "torch_cuda" in rng and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(rng["torch_cuda"])
                    except Exception:
                        pass
        except Exception:
            pass

        print(f"[resume] loaded {resume_path} (next_opt_step={start_opt_step}, best_val={best_val:.6g})")

    def _make_full_ckpt(*, opt_step: int) -> Dict[str, Any]:
        # RNG snapshot (best-effort; important for strict reproducibility).
        rng: Dict[str, Any] = {}
        try:
            import random

            rng["py_random"] = random.getstate()
        except Exception:
            pass
        try:
            rng["torch_cpu"] = torch.get_rng_state()
        except Exception:
            pass
        try:
            if device.type == "cuda" and torch.cuda.is_available():
                rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass

        return {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "opt": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "opt_step": int(opt_step),
            "best_val": float(best_val),
            "lr_mult": float(lr_mult),
            "auto_bs_cap_by_seq": dict(auto_bs_cap_by_seq),
            "rng": rng,
        }

    for opt_step in range(int(start_opt_step), total_opt_steps + 1):
        last_step = opt_step

        # LR schedule
        lr = lr_for_step(
            opt_step - 1,
            base_lr=float(args.lr),
            total_steps=int(args.steps),
            schedule=str(args.lr_schedule),
            warmup_steps=int(args.warmup_steps),
            min_lr=float(args.min_lr),
        )
        lr = float(lr) * float(lr_mult)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Determine training seq length for this optimizer step (schedule is in optimizer-step space).
        seq_len = _seq_len_for_step(opt_step - 1, default_seq_len=base_seq_len, schedule=seq_schedule)
        seq_len = int(min(max(2, seq_len), int(cfg.block_size)))

        # Determine batch/accum for this step.
        micro_bs, grad_accum = micro_bs_default, grad_accum_default
        if batch_by_seq is not None:
            micro_bs, grad_accum = _batch_for_seq(seq_len, batch_by_seq, default_bs=micro_bs, default_ga=grad_accum)
        if batch_schedule is not None:
            micro_bs, grad_accum = _batch_for_step(opt_step - 1, batch_schedule, default_bs=micro_bs, default_ga=grad_accum)

        # Desired global batch size for this step (best-effort constant even if we shrink micro_bs).
        gbs_target = int(micro_bs) * int(grad_accum)

        # Apply any learned caps from prior OOMs (caps at smaller seq_len also apply to larger seq_len).
        try:
            cap: Optional[int] = None
            for s, bs_cap in auto_bs_cap_by_seq.items():
                if int(s) <= int(seq_len):
                    cap = int(bs_cap) if cap is None else min(int(cap), int(bs_cap))
            if cap is not None and int(micro_bs) > int(cap):
                micro_bs = int(cap)
                grad_accum = int(math.ceil(float(gbs_target) / max(1.0, float(micro_bs))))
                if int(seq_len) not in warned_auto_caps:
                    print(f"[autobatch] applying learned cap @ seq={seq_len}: bs<={cap} (gbs_target={gbs_target})")
                    warned_auto_caps.add(int(seq_len))
        except Exception:
            pass

        # MPSGraph logits-size guard: automatically reduce micro-batch if (B*T*V) would exceed INT_MAX.
        if device.type == "mps":
            try:
                max_elems = 2_147_483_647  # INT_MAX
                denom = int(seq_len) * int(cfg.vocab_size)
                if denom <= 0:
                    denom = 1
                max_micro = int(max_elems // denom)
                if max_micro < 1:
                    raise ValueError(
                        f"MPSGraph limitation hit: T*vocab_size={denom} > INT_MAX. "
                        f"Reduce --train-seq-len/--block or use a smaller vocab."
                    )
                if int(micro_bs) > int(max_micro):
                    micro_bs = int(max_micro)
                    grad_accum = int(math.ceil(float(gbs_target) / max(1.0, float(micro_bs))))
                    if int(seq_len) not in warned_mps_logits:
                        print(
                            f"[autobatch] reducing batch_size on MPS to satisfy INT_MAX logits: "
                            f"bs<={max_micro} (seq={seq_len} vocab={cfg.vocab_size}, gbs_target={gbs_target})"
                        )
                        warned_mps_logits.add(int(seq_len))
            except Exception:
                pass

        # (batch_size, grad_accum) may be auto-adjusted on the fly on OOM.
        micro_bs_try = int(micro_bs)
        grad_accum_try = int(grad_accum)

        dt_step = 0.0
        fwd_step = 0.0
        bwd_step = 0.0
        opt_step_t = 0.0
        tok_step = 0
        loss_sum_t = torch.zeros([], device=device, dtype=torch.float32)

        while True:
            t_step0 = time.perf_counter()
            loss_sum_t = torch.zeros([], device=device, dtype=torch.float32)
            nonfinite = False
            nonfinite_detail: Optional[str] = None
            tok_step = 0
            fwd_step = 0.0
            bwd_step = 0.0
            opt_step_t = 0.0

            opt.zero_grad(set_to_none=True)

            try:
                # Backward over grad_accum microbatches.
                for _micro in range(grad_accum_try):
                    xb, yb = get_batch_any(train_view, batch_size=micro_bs_try, block_size=seq_len, device=device)
                    tok_step += int(xb.numel())

                    # Fail fast on corrupt token IDs. On GPU backends this can otherwise produce undefined behavior
                    # (instead of a clean "index out of range" error) and quickly lead to NaNs.
                    try:
                        x_oob = int(((xb < 0) | (xb >= int(cfg.vocab_size))).sum().detach().to("cpu").item())
                        y_oob = int(((yb < 0) | (yb >= int(cfg.vocab_size))).sum().detach().to("cpu").item())
                    except Exception:
                        x_oob = 0
                        y_oob = 0
                    if x_oob or y_oob:
                        try:
                            x_min = int(xb.detach().min().to("cpu").item())
                            x_max = int(xb.detach().max().to("cpu").item())
                        except Exception:
                            x_min = x_max = -1
                        try:
                            y_min = int(yb.detach().min().to("cpu").item())
                            y_max = int(yb.detach().max().to("cpu").item())
                        except Exception:
                            y_min = y_max = -1

                        dump_path = None
                        try:
                            os.makedirs(str(args.out_dir), exist_ok=True)
                            dump_path = os.path.join(str(args.out_dir), f"oob_tokens_step{opt_step}_micro{_micro}.pt")
                            torch.save(
                                {
                                    "opt_step": int(opt_step),
                                    "micro": int(_micro),
                                    "seq_len": int(seq_len),
                                    "batch_size": int(micro_bs_try),
                                    "grad_accum": int(grad_accum_try),
                                    "vocab_size": int(cfg.vocab_size),
                                    "x_oob": int(x_oob),
                                    "y_oob": int(y_oob),
                                    "xb": xb.detach().to("cpu"),
                                    "yb": yb.detach().to("cpu"),
                                },
                                dump_path,
                            )
                        except Exception:
                            dump_path = None

                        raise RuntimeError(
                            f"Out-of-range token ids detected at optimizer step {opt_step} micro={_micro+1}/{grad_accum_try}: "
                            f"xb_oob={x_oob} yb_oob={y_oob} xb[min,max]=[{x_min},{x_max}] yb[min,max]=[{y_min},{y_max}]"
                            + (f" dump={dump_path}" if dump_path else "")
                            + ". This usually means --vocab-size is wrong or the batch got corrupted. "
                              "The self-optimizer will adapt runtime policy automatically; if this persists, treat it as a data/vocab mismatch."
                        )

                    t1 = time.perf_counter()
                    with autocast_ctx():
                        if device.type == "mps":
                            feats, _ = model(xb, return_features=True)
                            Wt = model.tok_emb.weight.t()
                            V = int(Wt.shape[1])
                            loss_sum = torch.zeros([], device=device, dtype=torch.float32)
                            chunk = int(max(1, min(int(seq_len), 64)))
                            for i in range(0, int(seq_len), int(chunk)):
                                j = min(int(seq_len), i + int(chunk))
                                lg = feats[:, i:j, :] @ Wt
                                loss_sum = loss_sum + F.cross_entropy(lg.reshape(-1, V), yb[:, i:j].reshape(-1), reduction="sum").float()
                            denom = int(yb.numel()) if int(yb.numel()) > 0 else 1
                            loss = loss_sum / float(denom)
                            # For debug parity
                            logits = None  # type: ignore[assignment]
                        else:
                            logits, _ = model(xb)
                            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                        loss_to_back = loss / grad_accum_try

                    # Accumulate loss without synchronizing to CPU (important for GPU throughput).
                    try:
                        loss_sum_t = loss_sum_t + loss.detach().float()
                    except Exception:
                        pass
                    fwd_step += time.perf_counter() - t1

                    if not torch.isfinite(loss.detach()).all():
                        nonfinite = True
                        # Ensure logs for this step reflect non-finite state even if we break early.
                        try:
                            loss_sum_t = torch.full_like(loss_sum_t, float("nan"))
                        except Exception:
                            pass

                        # Capture debug info for the failing microbatch (do NOT materially affect normal fast-path).
                        try:
                            loss_val = float(loss.detach().to("cpu").item())
                        except Exception:
                            loss_val = float("nan")
                        try:
                            x_min = int(xb.detach().min().to("cpu").item())
                            x_max = int(xb.detach().max().to("cpu").item())
                        except Exception:
                            x_min = x_max = -1
                        try:
                            y_min = int(yb.detach().min().to("cpu").item())
                            y_max = int(yb.detach().max().to("cpu").item())
                        except Exception:
                            y_min = y_max = -1
                        try:
                            x_oob = int(((xb < 0) | (xb >= int(cfg.vocab_size))).sum().detach().to("cpu").item())
                            y_oob = int(((yb < 0) | (yb >= int(cfg.vocab_size))).sum().detach().to("cpu").item())
                        except Exception:
                            x_oob = -1
                            y_oob = -1
                        lg_any_nan = False
                        lg_any_inf = False
                        lg_last_nan = False
                        lg_last_inf = False
                        lg_last_max_abs = float("nan")
                        try:
                            if device.type != "mps":
                                lg = logits.detach()  # type: ignore[union-attr]
                                lg_any_nan = bool(torch.isnan(lg).any().detach().to("cpu").item())
                                lg_any_inf = bool(torch.isinf(lg).any().detach().to("cpu").item())
                                lg_last = lg[0, -1, :].float()
                            else:
                                # When using chunked loss (no full logits), probe last-token logits cheaply.
                                x_last = feats.detach()[0, -1, :].float()
                                lg_last = (x_last @ model.tok_emb.weight.t().detach().float())
                                lg_any_nan = bool(torch.isnan(feats.detach()).any().detach().to("cpu").item())
                                lg_any_inf = bool(torch.isinf(feats.detach()).any().detach().to("cpu").item())

                            lg_last_nan = bool(torch.isnan(lg_last).any().to("cpu").item())
                            lg_last_inf = bool(torch.isinf(lg_last).any().to("cpu").item())
                            lg_last_max_abs = float(torch.nan_to_num(lg_last, nan=0.0, posinf=0.0, neginf=0.0).abs().max().to("cpu").item())
                        except Exception:
                            pass

                        dump_path = None
                        try:
                            os.makedirs(str(args.out_dir), exist_ok=True)
                            dump_path = os.path.join(str(args.out_dir), f"nonfinite_loss_step{opt_step}_micro{_micro}.pt")
                            torch.save(
                                {
                                    "opt_step": int(opt_step),
                                    "micro": int(_micro),
                                    "seq_len": int(seq_len),
                                    "batch_size": int(micro_bs_try),
                                    "grad_accum": int(grad_accum_try),
                                    "vocab_size": int(cfg.vocab_size),
                                    "xb": xb.detach().to("cpu"),
                                    "yb": yb.detach().to("cpu"),
                                },
                                dump_path,
                            )
                        except Exception:
                            dump_path = None

                        nonfinite_detail = (
                            f"micro={_micro+1}/{grad_accum_try} loss={loss_val} "
                            f"xb[min,max]=[{x_min},{x_max}] yb[min,max]=[{y_min},{y_max}] xb_oob={x_oob} yb_oob={y_oob} "
                            f"logits_nan={lg_any_nan} logits_inf={lg_any_inf} "
                            f"logits_last_nan={lg_last_nan} logits_last_inf={lg_last_inf} logits_last_max|x|~{lg_last_max_abs:.3g}"
                            + (f" dump={dump_path}" if dump_path else "")
                        )
                        break

                    t2 = time.perf_counter()
                    if scaler is None:
                        loss_to_back.backward()
                    else:
                        scaler.scale(loss_to_back).backward()
                    bwd_step += time.perf_counter() - t2

                t3 = time.perf_counter()
                if nonfinite:
                    # Clear grads to avoid contaminating future steps.
                    opt.zero_grad(set_to_none=True)
                    # Self-heal on MPS: if autocast is producing NaNs, disable AMP and retry immediately.
                    if device.type == "mps" and bool(amp_state.get("enabled", False)) and (not amp_disabled_due_to_nonfinite):
                        amp_state["enabled"] = False
                        amp_disabled_due_to_nonfinite = True
                        try:
                            args.amp = False
                        except Exception:
                            pass
                        print("[selfopt][train] disabling AMP on MPS due to non-finite loss; retrying step in fp32.")
                        continue
                    consecutive_nonfinite += 1
                    if consecutive_nonfinite >= int(max_consecutive_nonfinite):
                        raise RuntimeError(
                            f"Aborting: {consecutive_nonfinite} consecutive non-finite steps. "
                            f"Last detail: {nonfinite_detail or '(none)'}"
                        )
                    if nan_policy == "reduce_lr":
                        lr_mult = max(1e-6, float(lr_mult) * float(nan_lr_decay))
                        print(f"[warn] non-finite loss detected @ step {opt_step}; skipping step and reducing lr_mult -> {lr_mult:.3g}")
                        if nonfinite_detail:
                            print(f"[warn] non-finite loss detail: {nonfinite_detail}")
                    elif nan_policy == "skip":
                        print(f"[warn] non-finite loss detected @ step {opt_step}; skipping optimizer step")
                        if nonfinite_detail:
                            print(f"[warn] non-finite loss detail: {nonfinite_detail}")
                    else:
                        raise RuntimeError(
                            f"Non-finite loss detected at optimizer step {opt_step}. "
                            "Try: lower the learning rate; the self-optimizer will also adapt runtime policy (precision/compile/batch) automatically."
                            + (f" Detail: {nonfinite_detail}" if nonfinite_detail else "")
                        )
                else:
                    grad_clip = float(getattr(args, "grad_clip", 0.0) or 0.0)
                    if grad_clip > 0:
                        # If using GradScaler, unscale before clipping so norms are meaningful.
                        if scaler is not None:
                            try:
                                scaler.unscale_(opt)
                            except Exception:
                                pass
                        # On MPS + bf16 params, PyTorch's clip_grad_norm_ can produce a non-finite norm
                        # due to bf16 overflow during the norm reduction even when grads are finite.
                        if device.type == "mps":
                            gn = _clip_grad_norm_fp32(model.parameters(), grad_clip)
                        else:
                            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        if not torch.isfinite(gn.detach()).all():
                            nonfinite = True
                    if nonfinite:
                        # IMPORTANT: capture debug info BEFORE clearing grads.
                        extra = _first_nonfinite_grad(model)
                        opt.zero_grad(set_to_none=True)
                        if device.type == "mps" and bool(amp_state.get("enabled", False)) and (not amp_disabled_due_to_nonfinite):
                            amp_state["enabled"] = False
                            amp_disabled_due_to_nonfinite = True
                            try:
                                args.amp = False
                            except Exception:
                                pass
                            print("[selfopt][train] disabling AMP on MPS due to non-finite grads; retrying step in fp32.")
                            continue
                        consecutive_nonfinite += 1
                        if consecutive_nonfinite >= int(max_consecutive_nonfinite):
                            raise RuntimeError(
                                f"Aborting: {consecutive_nonfinite} consecutive non-finite steps. "
                                + (f" First bad grad: {extra}" if extra else "")
                            )
                        if nan_policy == "reduce_lr":
                            lr_mult = max(1e-6, float(lr_mult) * float(nan_lr_decay))
                            print(f"[warn] non-finite grads detected @ step {opt_step}; skipping step and reducing lr_mult -> {lr_mult:.3g}")
                        elif nan_policy == "skip":
                            print(f"[warn] non-finite grads detected @ step {opt_step}; skipping optimizer step")
                        else:
                            raise RuntimeError(
                                f"Non-finite gradients detected at optimizer step {opt_step}. "
                                "Try: lower the learning rate; the self-optimizer will also adapt runtime policy (precision/compile/batch) automatically."
                                + (f" First bad grad: {extra}" if extra else "")
                            )
                    else:
                        if scaler is None:
                            opt.step()
                        else:
                            scaler.step(opt)
                            scaler.update()
                        opt.zero_grad(set_to_none=True)
                        opt_step_t = time.perf_counter() - t3
                        consecutive_nonfinite = 0

                if bool(getattr(args, "sync_timing", False)):
                    device_synchronize(device)
                dt_step = time.perf_counter() - t_step0
                break

            except RuntimeError as e:
                if _is_oom_error(e):
                    if micro_bs_try <= 1:
                        # Last-resort self-healing: try enabling checkpointing/AMP before bailing.
                        try:
                            if not bool(getattr(model, "grad_checkpointing", False)):
                                print(f"[selfopt][train] OOM @ step {opt_step} with bs=1; enabling grad checkpointing and retrying.")
                                try:
                                    args.grad_checkpoint = True
                                except Exception:
                                    pass
                                try:
                                    model.grad_checkpointing = True
                                except Exception:
                                    pass
                                empty_device_cache(device)
                                continue
                        except Exception:
                            pass

                        try:
                            if (not bool(amp_state.get("enabled", False))) and device.type in ("cuda", "mps"):
                                # Prefer bf16 when supported; otherwise fp16.
                                dt: Optional[torch.dtype]
                                if _supports_dtype(device, torch.bfloat16):
                                    dt = torch.bfloat16
                                elif _supports_dtype(device, torch.float16):
                                    dt = torch.float16
                                else:
                                    dt = None
                                if dt is None:
                                    raise RuntimeError("No supported AMP dtype found for device")
                                amp_state["enabled"] = True
                                amp_state["dtype"] = dt
                                try:
                                    args.amp = True
                                    args.amp_dtype = ("bf16" if dt == torch.bfloat16 else "fp16")
                                except Exception:
                                    pass
                                if device.type == "cuda" and dt == torch.float16 and scaler is None:
                                    try:
                                        from torch.cuda.amp import GradScaler  # type: ignore

                                        scaler = GradScaler()
                                    except Exception:
                                        scaler = None
                                print(f"[selfopt][train] OOM @ step {opt_step} with bs=1; enabling AMP ({args.amp_dtype}) and retrying.")
                                empty_device_cache(device)
                                continue
                        except Exception:
                            pass

                        raise RuntimeError(
                            f"Out of memory at optimizer step {opt_step} even with batch_size=1 (seq_len={seq_len}). "
                            "Reduce model size or reduce sequence length (block/train-seq-len)."
                        ) from e
                    new_bs = max(1, int(micro_bs_try // 2))
                    prev = int(auto_bs_cap_by_seq.get(int(seq_len), 1 << 30))
                    auto_bs_cap_by_seq[int(seq_len)] = min(prev, int(new_bs))
                    empty_device_cache(device)
                    micro_bs_try = int(new_bs)
                    grad_accum_try = int(math.ceil(float(gbs_target) / max(1.0, float(micro_bs_try))))
                    print(
                        f"[autobatch] OOM @ step {opt_step} seq={seq_len}; retrying with "
                        f"batch_size={micro_bs_try} grad_accum={grad_accum_try} (gbs_target={gbs_target})"
                    )
                    continue
                raise

        # Update step-level accumulators using the *successful* attempt only.
        tok_count += int(tok_step)
        tok_acc += int(tok_step)
        fwd_acc += float(fwd_step)
        bwd_acc += float(bwd_step)
        opt_acc += float(opt_step_t)
        dt_acc += float(dt_step)
        steps_in_acc += 1

        # Use the actual micro-batch/accum used for this step when logging.
        micro_bs = int(micro_bs_try)
        grad_accum = int(grad_accum_try)

        # Logging (optimizer steps)
        if int(args.log_every) > 0 and (opt_step % int(args.log_every) == 0 or opt_step == 1):
            tok_s = float(tok_acc / max(dt_acc, 1e-9))
            try:
                loss_avg = float((loss_sum_t / max(1, grad_accum)).detach().to("cpu").item())
            except Exception:
                loss_avg = float("nan")
            ppl = float(math.exp(loss_avg)) if loss_avg < 20 else float("inf")
            ev = {
                "type": "train",
                "step": int(opt_step),
                "loss": float(loss_avg),
                "ppl": float(ppl),
                "lr": float(lr),
                "tok_s": float(tok_s),
                "seq_len": int(seq_len),
                "gbs": int(micro_bs * grad_accum),
                "ms_step": float(1000.0 * dt_acc / max(1, steps_in_acc)),
                "ms_fwd": float(1000.0 * fwd_acc / max(1, steps_in_acc)),
                "ms_bwd": float(1000.0 * bwd_acc / max(1, steps_in_acc)),
                "ms_opt": float(1000.0 * opt_acc / max(1, steps_in_acc)),
                **get_device_mem_stats(device),
            }
            if logger is not None:
                logger.log(ev)
            print(
                f"step {opt_step}/{total_opt_steps} loss={ev['loss']:.4f} ppl={ev['ppl']:.2f} "
                f"lr={lr:.3g} tok/s={tok_s:.0f} seq={seq_len} gbs={ev['gbs']} "
                f"ms/step={ev['ms_step']:.0f}",
                flush=True,
            )
            # reset interval accumulators
            dt_acc = 0.0
            tok_acc = 0
            fwd_acc = 0.0
            bwd_acc = 0.0
            opt_acc = 0.0
            steps_in_acc = 0

        if int(args.eval_every) > 0 and (opt_step % int(args.eval_every) == 0 or opt_step == total_opt_steps):
            tr_loss, va_loss = estimate_loss(int(args.eval_iters))
            ev = {"type": "eval", "step": opt_step, "train_loss": tr_loss, "val_loss": va_loss}
            if logger is not None:
                logger.log(ev)
            print(f"[eval] step {opt_step}: train={tr_loss:.4f} val={va_loss:.4f}", flush=True)
            if va_loss < best_val:
                best_val = va_loss
                torch.save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(str(args.out_dir), "best.pt"))

        if int(args.save_every) > 0 and (opt_step % int(args.save_every) == 0):
            torch.save(_make_full_ckpt(opt_step=opt_step), os.path.join(str(args.out_dir), "last.pt"))

    torch.save(_make_full_ckpt(opt_step=last_step), os.path.join(str(args.out_dir), "last.pt"))
    if logger is not None:
        logger.finalize(best_val=best_val if best_val < float("inf") else float("nan"), last_step=last_step)
        logger.close()
