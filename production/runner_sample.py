from __future__ import annotations

import argparse
import os

import torch
from typing import Literal

from production.runtime_tuning import KVCachePolicy, KVSelfOptConfig
from production.run_config import SampleConfig
from production.selfopt_logging import SelfOptLogger
from production.selfopt_cache import as_object_list, as_str_object_dict


KVCacheKind = Literal["fp16", "fp32", "q8_0", "q4_0", "nf4"]


def _torch_load_obj(path: str, *, device: torch.device) -> object:
    # `torch.load` is typed as returning `Any` in stubs; isolate it behind an `object` boundary.
    return torch.load(str(path), map_location=device)  # pyright: ignore[reportAny]


def _as_kvcache_kind_typed(o: object) -> KVCacheKind:
    s = str(o or "").strip()
    if s == "fp32":
        return "fp32"
    if s == "q8_0":
        return "q8_0"
    if s == "q4_0":
        return "q4_0"
    if s == "nf4":
        return "nf4"
    return "fp16"


def _as_kvcache_kind_opt(o: object) -> KVCacheKind | None:
    if o is None:
        return None
    s = str(o).strip()
    if s in ("fp16", "fp32", "q8_0", "q4_0", "nf4"):
        return _as_kvcache_kind_typed(s)
    return None


def _as_state_dict(o: object) -> dict[str, torch.Tensor] | None:
    if not isinstance(o, dict):
        return None
    out: dict[str, torch.Tensor] = {}
    for kv in o.items():  # pyright: ignore[reportUnknownVariableType]
        kv2: tuple[object, object] = kv  # pyright: ignore[reportUnknownVariableType]
        k_obj, v_obj = kv2[0], kv2[1]
        if not isinstance(k_obj, str):
            return None
        if not isinstance(v_obj, torch.Tensor):
            return None
        out[k_obj] = v_obj
    return out


def _as_str_list(o: object) -> list[str]:
    xs = as_object_list(o)
    if xs is None:
        return []
    return [str(x) for x in xs]


def run_sample(*, args: argparse.Namespace, device: torch.device, self_opt: KVSelfOptConfig | None) -> None:
    """Sample/generate from a checkpoint."""
    # Local imports so CLI --help doesn't require torch/tiktoken.
    from production.instrumentation import RunLogger
    from production.model import GPT, ModelConfig

    try:
        import tiktoken  # type: ignore
    except ImportError:
        tiktoken = None  # type: ignore

    cfg_run = SampleConfig.from_args(args)

    if not cfg_run.ckpt:
        raise ValueError("--ckpt is required for --mode sample")

    ckpt_obj = _torch_load_obj(str(cfg_run.ckpt), device=device)
    ckpt = as_str_object_dict(ckpt_obj)
    if ckpt is None:
        raise ValueError("Checkpoint payload must be a dict-like object")

    cfg_dict_obj = ckpt.get("config", None)
    cfg_dict = as_str_object_dict(cfg_dict_obj)
    if cfg_dict is None:
        raise ValueError("Checkpoint missing 'config'. Can't reconstruct model safely.")
    cfg = ModelConfig.from_dict(cfg_dict, device=device)
    model = GPT(cfg).to(device)

    sd = _as_state_dict(ckpt.get("model"))
    if sd is None:
        raise ValueError("Checkpoint missing 'model' state_dict (expected dict[str, Tensor]).")
    inc = model.load_state_dict(sd, strict=False)
    missing = _as_str_list(getattr(inc, "missing_keys", []))
    unexpected = _as_str_list(getattr(inc, "unexpected_keys", []))
    bad_missing = [k for k in missing if "decoupled_gate_logit" not in k]
    bad_unexpected = [k for k in unexpected if "decoupled_gate_logit" not in k]
    if bad_missing or bad_unexpected:
        _ = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"[warn] Non-strict checkpoint load. Missing={missing} Unexpected={unexpected}")
    _ = model.eval()

    # Prompt: either raw token IDs or text (tiktoken only)
    try:
        prompt_ids = [int(t) for t in str(cfg_run.prompt_tokens).strip().split()]
    except ValueError as exc:
        if str(cfg_run.tokenizer) != "tiktoken":
            raise ValueError("Text prompts require --tokenizer tiktoken") from exc
        if tiktoken is None:
            raise ImportError("tiktoken needed for text prompts") from exc
        enc = tiktoken.get_encoding("gpt2")
        prompt_ids = enc.encode_ordinary(str(cfg_run.prompt_tokens))

    prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    # Expert override: force an atomic decoupled KV cache policy from a single string.
    if cfg_run.kv_policy:
        if str(getattr(cfg, "attn_mode", "")) != "decoupled":
            raise ValueError("--kv-policy is only supported for decoupled attention checkpoints")
        pol = KVCachePolicy.parse(str(cfg_run.kv_policy))
        # Apply as per-tensor overrides (so model.generate() stays unchanged).
        args.kv_cache_k_sem = pol.k_sem_kind
        args.kv_cache_k_geo = pol.k_geo_kind
        args.kv_cache_v = pol.v_kind
        args.kv_qblock_k_sem = int(pol.k_sem_qblock)
        args.kv_qblock_k_geo = int(pol.k_geo_qblock)
        args.kv_qblock_v = int(pol.v_qblock)
        args.kv_residual = int(pol.residual_len)

        # If selfopt is enabled, keep decode-plan tuning but disable cache-policy tuning (policy is forced).
        if self_opt is not None:
            try:
                self_opt.scope = "decode"
            except (AttributeError, TypeError, ValueError):
                pass

    logger = None
    if cfg_run.instrument != "off" or bool(cfg_run.live_plot) or bool(cfg_run.tb) or bool(cfg_run.wandb):
        logger = RunLogger(
            str(cfg_run.out_dir or ""),
            instrument=str(cfg_run.instrument),
            cfg=cfg,
            args=args,
            device=device,
            live_plot=bool(cfg_run.live_plot),
            tb=bool(cfg_run.tb),
            wandb=bool(cfg_run.wandb),
        )
    slog = SelfOptLogger(
        jsonl_path=(os.path.join(str(cfg_run.out_dir), "events.jsonl") if cfg_run.out_dir else None),
        run_logger=logger,
        echo=False,
    )

    print(f"Generating {int(cfg_run.max_new_tokens)} tokens...")
    try:
        if cfg_run.draft_ckpt:
            dckpt_obj = _torch_load_obj(str(cfg_run.draft_ckpt), device=device)
            dckpt = as_str_object_dict(dckpt_obj)
            if dckpt is None:
                raise ValueError("Draft checkpoint payload must be a dict-like object")
            dcfg_dict_obj = dckpt.get("config", None)
            dcfg_dict = as_str_object_dict(dcfg_dict_obj)
            if dcfg_dict is None:
                raise ValueError("Draft checkpoint missing 'config'. Can't reconstruct draft model safely.")
            dcfg = ModelConfig.from_dict(dcfg_dict, device=device)
            draft = GPT(dcfg).to(device)
            dsd = _as_state_dict(dckpt.get("model"))
            if dsd is None:
                raise ValueError("Draft checkpoint missing 'model' state_dict (expected dict[str, Tensor]).")
            incompatible_d = draft.load_state_dict(dsd, strict=False)
            mk = _as_str_list(getattr(incompatible_d, "missing_keys", []))
            uk = _as_str_list(getattr(incompatible_d, "unexpected_keys", []))
            bad_missing_d = [k for k in mk if "decoupled_gate_logit" not in k]
            bad_unexpected_d = [k for k in uk if "decoupled_gate_logit" not in k]
            if bad_missing_d or bad_unexpected_d:
                _ = draft.load_state_dict(dsd, strict=True)
            if mk or uk:
                print(f"[warn] Non-strict draft checkpoint load. Missing={mk} Unexpected={uk}")

            # Basic safety: vocab size must match for token IDs to be meaningful.
            if int(dcfg.vocab_size) != int(cfg.vocab_size):
                raise ValueError(f"Draft vocab_size {dcfg.vocab_size} != main vocab_size {cfg.vocab_size}")

            # Match main model's inference behavior (disable dropout, etc.)
            _ = draft.eval()

            out = model.generate_speculative(
                prompt,
                draft_model=draft,
                max_new_tokens=int(cfg_run.max_new_tokens),
                temperature=float(cfg_run.temperature),
                top_k=(None if cfg_run.top_k is None else int(cfg_run.top_k)),
                kv_cache=_as_kvcache_kind_typed(cfg_run.kv_cache),
                kv_qblock=int(cfg_run.kv_qblock),
                kv_residual=int(cfg_run.kv_residual),
                kv_decode_block=int(cfg_run.kv_decode_block),
                kv_fused=str(cfg_run.kv_fused),
                self_opt=self_opt,
                kv_cache_k=_as_kvcache_kind_opt(cfg_run.kv_cache_k),
                kv_cache_v=_as_kvcache_kind_opt(cfg_run.kv_cache_v),
                kv_cache_k_sem=_as_kvcache_kind_opt(cfg_run.kv_cache_k_sem),
                kv_cache_k_geo=_as_kvcache_kind_opt(cfg_run.kv_cache_k_geo),
                kv_qblock_k=cfg_run.kv_qblock_k,
                kv_qblock_v=cfg_run.kv_qblock_v,
                kv_qblock_k_sem=cfg_run.kv_qblock_k_sem,
                kv_qblock_k_geo=cfg_run.kv_qblock_k_geo,
                spec_k=int(cfg_run.spec_k),
                spec_method=str(cfg_run.spec_method),
                spec_extra_token=bool(cfg_run.spec_extra_token),
                spec_disable_below_accept=float(cfg_run.spec_disable_below_accept),
                log_callback=slog.log,
            )
        else:
            out = model.generate(
                prompt,
                max_new_tokens=int(cfg_run.max_new_tokens),
                temperature=float(cfg_run.temperature),
                top_k=(None if cfg_run.top_k is None else int(cfg_run.top_k)),
                kv_cache=_as_kvcache_kind_typed(cfg_run.kv_cache),
                kv_qblock=int(cfg_run.kv_qblock),
                kv_residual=int(cfg_run.kv_residual),
                kv_decode_block=int(cfg_run.kv_decode_block),
                kv_fused=str(cfg_run.kv_fused),
                self_opt=self_opt,
                kv_cache_k=_as_kvcache_kind_opt(cfg_run.kv_cache_k),
                kv_cache_v=_as_kvcache_kind_opt(cfg_run.kv_cache_v),
                kv_cache_k_sem=_as_kvcache_kind_opt(cfg_run.kv_cache_k_sem),
                kv_cache_k_geo=_as_kvcache_kind_opt(cfg_run.kv_cache_k_geo),
                kv_qblock_k=cfg_run.kv_qblock_k,
                kv_qblock_v=cfg_run.kv_qblock_v,
                kv_qblock_k_sem=cfg_run.kv_qblock_k_sem,
                kv_qblock_k_geo=cfg_run.kv_qblock_k_geo,
                log_callback=slog.log,
            )
    finally:
        slog.close()

    # Convert output tokens to list[int] without relying on `.tolist()` typing.
    out0 = out[0].detach().to("cpu")
    out_ids: list[int] = []
    n = int(out0.size(0)) if out0.ndim == 1 else int(out0.size(-1))
    # Handle [T] or [B,T]; sample uses B=1.
    if out0.ndim == 2:
        for i in range(int(out0.size(1))):
            out_ids.append(int(out0[0, i].item()))
    else:
        for i in range(n):
            out_ids.append(int(out0[i].item()))
    if str(cfg_run.tokenizer) == "tiktoken":
        if tiktoken is None:
            raise ImportError("tiktoken not installed")
        enc = tiktoken.get_encoding("gpt2")
        print(enc.decode(out_ids))
    else:
        print(out_ids)


