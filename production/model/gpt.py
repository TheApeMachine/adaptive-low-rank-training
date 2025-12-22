"""
gpt defines the neural architecture and orchestration for generation.
"""
from __future__ import annotations
from collections.abc import Callable
import logging
import math
import sys
from types import ModuleType
from typing import Protocol, cast
import torch
from torch import nn
import torch.utils.checkpoint as torch_checkpoint
from typing_extensions import override

from production.kvcache_backend import (
    DecoupledLayerKVCache,
    KVCacheKind,
    KVCacheTensorConfig,
    LayerKVCache,
)
from production.runtime_tuning import (
    KVCachePolicy,
    KVCachePolicySelfOptimizer,
    KVDecodeSelfOptimizer,
    KVSelfOptConfig,
    load_token_ids_spec,
    policy_quality_reject_reasons,
    warn_policy_quality_reject,
)
from .metrics import Metrics
from .block import Block
from .cache import Cache
from .config import ModelConfig

logger = logging.getLogger(__name__)


def _maybe_public_model_module() -> ModuleType | None:
    """
    Return the already-imported public API module (`production.model`) if present.

    This avoids importing `production.model` from within `production.model.gpt`, which would create an
    import cycle (package `__init__` imports `GPT` from this module).
    """
    mod = sys.modules.get("production.model")
    return mod if isinstance(mod, ModuleType) else None


class _CacheWithPos(Protocol):
    pos: int

def _checkpoint(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Typed wrapper around torch's checkpoint (no ignore comments)."""
    cp = cast(Callable[..., torch.Tensor], torch_checkpoint.checkpoint)
    return cp(fn, x, use_reentrant=False)

class GPT(nn.Module):
    """Neural architecture with speculative and incremental support."""
    def __init__(self, cfg: ModelConfig) -> None:
        """
        init the GPT model.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.cfg: ModelConfig = cfg
        self.tok_emb: nn.Embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.emb_in: nn.Linear | None = (
            nn.Linear(cfg.embed_dim, cfg.d_model, bias=False)
            if cfg.embed_dim != cfg.d_model else None
        )
        self.emb_out: nn.Linear | None = (
            nn.Linear(cfg.d_model, cfg.embed_dim, bias=False)
            if cfg.embed_dim != cfg.d_model else None
        )
        self.drop: nn.Dropout = nn.Dropout(cfg.dropout)
        self.blocks: nn.ModuleList = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f: nn.LayerNorm = nn.LayerNorm(cfg.d_model)

        # Causal mask for prefill
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer(
            "mask",
            mask.view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )
        _ = self.apply(self._init_weights)
        self.grad_checkpointing: bool = False
        # Memoize cache-policy quality checks per-session (avoid re-gating the same policy repeatedly).
        self._policy_quality_ok: set[str] = set()
        self._policy_quality_long_ok: set[str] = set()
        self._policy_quality_needle_ok: set[str] = set()

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            _ = nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @staticmethod
    def _normalize_attn_mode(mode: object) -> str:
        v = getattr(mode, "value", mode)
        # Treat `None` as "unset"; preserve falsy-but-meaningful values (0/False) by not using `v or ""`.
        s = "" if v is None else str(v).strip().lower()
        if s == "":
            return "bottleneck"
        if s in ("baseline", "standard", "base"):
            return "standard"
        if s in ("gqa", "bottleneck", "decoupled"):
            return s
        raise ValueError(
            f'Unknown attn_mode={v!r} (normalized={s!r}). Accepted aliases: ("standard"/"baseline"/"base"), ("gqa"/"bottleneck"/"decoupled").'
        )

    @staticmethod
    def _residual_for(kind: KVCacheKind, residual_len: int) -> int:
        return int(residual_len) if str(kind) not in ("fp16", "fp32") else 0

    @staticmethod
    def _kv_dim_for_mode(cfg: ModelConfig, mode: str) -> int:
        """Compute KV cache per-layer feature dim for non-decoupled attention modes."""
        try:
            if mode == "standard":
                kv_dim = int(getattr(cfg, "d_model", 0) or 0)
            elif mode == "gqa":
                nh = int(getattr(cfg, "n_head", 1) or 1)
                attn_dim = int(getattr(cfg, "attn_dim", 0) or 0)
                head_dim = int(attn_dim // max(1, nh))
                kvh = int(getattr(cfg, "kv_head", None) or nh)
                kv_dim = int(kvh * head_dim)
            else:
                kv_dim = int(getattr(cfg, "attn_dim", 0) or 0)
            return int(max(1, kv_dim))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _resolve_kv_tensor_cfgs(
        *,
        kv_cache: KVCacheKind,
        kv_qblock: int,
        kv_residual: int,
        kv_cache_k: KVCacheKind | None,
        kv_cache_v: KVCacheKind | None,
        kv_cache_k_sem: KVCacheKind | None,
        kv_cache_k_geo: KVCacheKind | None,
        kv_qblock_k: int | None,
        kv_qblock_v: int | None,
        kv_qblock_k_sem: int | None,
        kv_qblock_k_geo: int | None,
    ) -> tuple[KVCacheTensorConfig, KVCacheTensorConfig, KVCacheTensorConfig, KVCacheTensorConfig]:
        """Resolve per-tensor KV configs (K, V, K_sem, K_geo) from CLI/kwargs."""
        k_kind = cast(KVCacheKind, str(kv_cache_k or kv_cache))
        v_kind = cast(KVCacheKind, str(kv_cache_v or kv_cache))
        k_qb = int(kv_qblock_k) if kv_qblock_k is not None else int(kv_qblock)
        v_qb = int(kv_qblock_v) if kv_qblock_v is not None else int(kv_qblock)

        k_sem_kind = cast(KVCacheKind, str(kv_cache_k_sem or kv_cache))
        k_geo_default: KVCacheKind | None = None
        if str(kv_cache) == "q4_0":
            k_geo_default = "q8_0"
        k_geo_kind = cast(KVCacheKind, str(kv_cache_k_geo or k_geo_default or kv_cache))
        k_sem_qb = int(kv_qblock_k_sem) if kv_qblock_k_sem is not None else int(kv_qblock)
        k_geo_qb = int(kv_qblock_k_geo) if kv_qblock_k_geo is not None else int(kv_qblock)

        k_cfg = KVCacheTensorConfig(kind=k_kind, qblock=int(k_qb), residual_len=GPT._residual_for(k_kind, kv_residual))
        v_cfg = KVCacheTensorConfig(kind=v_kind, qblock=int(v_qb), residual_len=GPT._residual_for(v_kind, kv_residual))
        k_sem_cfg = KVCacheTensorConfig(
            kind=k_sem_kind, qblock=int(k_sem_qb), residual_len=GPT._residual_for(k_sem_kind, kv_residual)
        )
        k_geo_cfg = KVCacheTensorConfig(
            kind=k_geo_kind, qblock=int(k_geo_qb), residual_len=GPT._residual_for(k_geo_kind, kv_residual)
        )
        return k_cfg, v_cfg, k_sem_cfg, k_geo_cfg

    def build_layer_caches(
        self,
        *,
        model: "GPT",
        mode: str,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        k_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        promote_layers: int | None,
        kv_decode_block: int,
        kv_fused: str,
    ) -> list[DecoupledLayerKVCache | LayerKVCache]:
        """Build per-layer KV caches with optional layerwise fp16 promotion (decoupled)."""
        # Resolve cache classes via the public `production.model` module *if it's already loaded* so
        # tests can patch `production.model.LayerKVCache` / `production.model.DecoupledLayerKVCache`,
        # without importing `production.model` (which would create an import cycle).
        model_mod = _maybe_public_model_module()
        LayerKVCacheCls = cast(
            type[LayerKVCache],
            getattr(model_mod, "LayerKVCache", LayerKVCache) if model_mod is not None else LayerKVCache,
        )
        DecoupledLayerKVCacheCls = cast(
            type[DecoupledLayerKVCache],
            getattr(model_mod, "DecoupledLayerKVCache", DecoupledLayerKVCache)
            if model_mod is not None
            else DecoupledLayerKVCache,
        )

        caches: list[DecoupledLayerKVCache | LayerKVCache] = []
        n_layer = int(getattr(model.cfg, "n_layer", 0) or 0)

        if mode == "decoupled":
            fp16 = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
            for i in range(n_layer):
                use_fp16 = bool(promote_layers is not None and i < int(promote_layers))
                caches.append(
                    DecoupledLayerKVCacheCls(
                        batch_size=int(batch_size),
                        max_seq_len=int(max_seq_len),
                        k_sem_dim=int(getattr(model.cfg, "sem_dim", 0) or 0),
                        k_geo_dim=int(getattr(model.cfg, "geo_dim", 0) or 0),
                        v_dim=int(getattr(model.cfg, "attn_dim", 0) or 0),
                        k_sem_cfg=(fp16 if use_fp16 else k_sem_cfg),
                        k_geo_cfg=(fp16 if use_fp16 else k_geo_cfg),
                        v_cfg=(fp16 if use_fp16 else v_cfg),
                        device=device,
                    )
                )
        else:
            kv_dim = self._kv_dim_for_mode(model.cfg, str(mode))
            for _ in range(n_layer):
                caches.append(
                    LayerKVCacheCls(
                        batch_size=int(batch_size),
                        max_seq_len=int(max_seq_len),
                        k_dim=int(kv_dim),
                        v_dim=int(kv_dim),
                        k_cfg=k_cfg,
                        v_cfg=v_cfg,
                        device=device,
                    )
                )

        for c in caches:
            setattr(c, "decode_block", int(kv_decode_block))
            setattr(c, "fused", str(kv_fused))
        return caches

    @override
    def forward(
        self,
        idx: torch.Tensor,
        *,
        caches: list[DecoupledLayerKVCache | LayerKVCache] | None = None,
        pos_offset: int = 0,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache] | None]:
        """Forward pass supporting training, prefill, and incremental decode."""
        _, t = idx.shape
        if caches is None and t > self.cfg.block_size:
            raise ValueError(
                f"Sequence length {t} > block_size {self.cfg.block_size}. Increase --block."
            )

        x = cast(torch.Tensor, self.tok_emb(idx))
        if self.emb_in is not None:
            x = cast(torch.Tensor, self.emb_in(x))
        x = cast(torch.Tensor, self.drop(x))

        attn_mask: torch.Tensor | None
        if caches is None:
            if self.cfg.null_attn:
                causal = cast(torch.Tensor, getattr(self, "mask"))
                attn_mask = causal[:, :, :t, :t]
            else:
                attn_mask = None
        else:
            if t > 1:
                prev_len = cast(_CacheWithPos, cast(object, caches[0])).pos if caches else 0
                if prev_len == 0 and (not self.cfg.null_attn):
                    attn_mask = None
                else:
                    seq_len = prev_len + t
                    key_pos = torch.arange(seq_len, device=idx.device).view(1, 1, 1, seq_len)
                    q_pos = (prev_len + torch.arange(t, device=idx.device)).view(1, 1, t, 1)
                    attn_mask = key_pos <= q_pos
            else:
                attn_mask = None

        new_caches: list[DecoupledLayerKVCache | LayerKVCache] | None = [] if caches is not None else None

        if caches is None and self.training and self.grad_checkpointing:
            for m in self.blocks:
                blk = cast(Block, m)

                def _blk_fwd(x_in: torch.Tensor, _blk: Block = blk) -> torch.Tensor:
                    y, _c = cast(
                        tuple[torch.Tensor, object],
                        _blk(x_in, attn_mask=attn_mask, cache=None, pos_offset=pos_offset),
                    )
                    return y

                x = _checkpoint(_blk_fwd, x)
        elif caches is None:
            for m in self.blocks:
                blk = cast(Block, m)
                x, _c = cast(
                    tuple[torch.Tensor, object],
                    blk(x, attn_mask=attn_mask, cache=None, pos_offset=pos_offset),
                )
        else:
            for i, m in enumerate(self.blocks):
                blk = cast(Block, m)
                layer_cache_in = caches[i]
                x, layer_cache_out = cast(
                    tuple[torch.Tensor, DecoupledLayerKVCache | LayerKVCache | None],
                    blk(x, attn_mask=attn_mask, cache=layer_cache_in, pos_offset=pos_offset),
                )
                cast(list[DecoupledLayerKVCache | LayerKVCache], new_caches).append(
                    cast(DecoupledLayerKVCache | LayerKVCache, layer_cache_out)
                )

        x = cast(torch.Tensor, self.ln_f(x))
        x_small = cast(torch.Tensor, self.emb_out(x)) if self.emb_out is not None else x
        if return_features:
            return x_small, new_caches
        logits = x_small @ self.tok_emb.weight.t()
        return logits, new_caches

    def generate(
        self,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",
        kv_cache_k: KVCacheKind | None = None,
        kv_cache_v: KVCacheKind | None = None,
        kv_cache_k_sem: KVCacheKind | None = None,
        kv_cache_k_geo: KVCacheKind | None = None,
        kv_qblock_k: int | None = None,
        kv_qblock_v: int | None = None,
        kv_qblock_k_sem: int | None = None,
        kv_qblock_k_geo: int | None = None,
        self_opt: KVSelfOptConfig | None = None,
        log_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> torch.Tensor:
        """Token sampling/generation using KV caches (optionally self-optimized)."""
        if prompt.ndim != 2:
            raise ValueError(f"prompt must be (B,T) but got shape={tuple(prompt.shape)}")
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")

        B, T0 = int(prompt.size(0)), int(prompt.size(1))
        device = prompt.device

        # Resolve per-tensor KV configs.
        k_cfg, v_cfg, k_sem_cfg, k_geo_cfg = GPT._resolve_kv_tensor_cfgs(
            kv_cache=kv_cache,
            kv_qblock=int(kv_qblock),
            kv_residual=int(kv_residual),
            kv_cache_k=kv_cache_k,
            kv_cache_v=kv_cache_v,
            kv_cache_k_sem=kv_cache_k_sem,
            kv_cache_k_geo=kv_cache_k_geo,
            kv_qblock_k=kv_qblock_k,
            kv_qblock_v=kv_qblock_v,
            kv_qblock_k_sem=kv_qblock_k_sem,
            kv_qblock_k_geo=kv_qblock_k_geo,
        )

        mode = self._normalize_attn_mode(getattr(self.cfg, "attn_mode", ""))
        max_seq = int(T0 + int(max_new_tokens))

        # Optional decoupled cache-policy tuning / quality gating.
        promote_layers = None
        base_policy_s: str | None = None
        chosen_policy_s: str | None = None
        if mode == "decoupled":
            base_k_sem_kind = cast(KVCacheKind, str(k_sem_cfg.kind))
            base_k_geo_kind = cast(KVCacheKind, str(k_geo_cfg.kind))
            base_v_kind = cast(KVCacheKind, str(v_cfg.kind))
            base_k_sem_qb = int(k_sem_cfg.qblock)
            base_k_geo_qb = int(k_geo_cfg.qblock)
            base_v_qb = int(v_cfg.qblock)
            k_sem_cfg2, k_geo_cfg2, v_cfg2, promote_layers, kv_residual2 = self._choose_kv_cache_policy(
                model=self,
                self_opt=self_opt,
                device=device,
                prompt=prompt,
                k_sem_cfg=k_sem_cfg,
                k_geo_cfg=k_geo_cfg,
                v_dec_cfg=v_cfg,
                kv_residual=int(kv_residual),
                kv_decode_block=int(kv_decode_block),
                kv_fused=str(kv_fused),
                max_new_tokens=int(max_new_tokens),
                is_speculative=False,
            )
            k_sem_cfg, k_geo_cfg, v_cfg = k_sem_cfg2, k_geo_cfg2, v_cfg2
            kv_residual = int(kv_residual2)

            try:
                base_policy_s = KVCachePolicy(
                    k_sem_kind=base_k_sem_kind,
                    k_geo_kind=base_k_geo_kind,
                    v_kind=base_v_kind,
                    k_sem_qblock=int(base_k_sem_qb),
                    k_geo_qblock=int(base_k_geo_qb),
                    v_qblock=int(base_v_qb),
                    residual_len=int(kv_residual2),
                ).short()
                chosen_policy_s = KVCachePolicy(
                    k_sem_kind=k_sem_cfg.kind,
                    k_geo_kind=k_geo_cfg.kind,
                    v_kind=v_cfg.kind,
                    k_sem_qblock=int(k_sem_cfg.qblock),
                    k_geo_qblock=int(k_geo_cfg.qblock),
                    v_qblock=int(v_cfg.qblock),
                    residual_len=int(kv_residual2),
                ).short()
            except (TypeError, ValueError):
                base_policy_s = None
                chosen_policy_s = None

            if log_callback is not None:
                try:
                    log_callback(
                        {
                            "type": "analysis",
                            "subtype": "selfopt_cache_policy",
                            "base_policy": str(base_policy_s or ""),
                            "chosen_policy": str(chosen_policy_s or ""),
                            "promote_layers": (int(promote_layers) if promote_layers is not None else 0),
                        }
                    )
                except (TypeError, ValueError, RuntimeError):
                    pass

        caches = self.build_layer_caches(
            model=self,
            mode=str(mode),
            batch_size=int(B),
            max_seq_len=int(max_seq),
            device=device,
            k_cfg=k_cfg,
            v_cfg=v_cfg,
            k_sem_cfg=k_sem_cfg,
            k_geo_cfg=k_geo_cfg,
            promote_layers=promote_layers,
            kv_decode_block=int(kv_decode_block),
            kv_fused=str(kv_fused),
        )

        # Fast path: no generation requested; still return prompt (caches were built for tests/benching).
        if int(max_new_tokens) == 0:
            return prompt

        # Decode-plan tuner (decoupled only).
        decode_tuner: KVDecodeSelfOptimizer | None = None
        if (
            self_opt is not None
            and str(getattr(self_opt, "mode", "none")) != "none"
            and str(getattr(self_opt, "scope", "all")) in ("decode", "all")
            and mode == "decoupled"
        ):
            try:
                decode_tuner = KVDecodeSelfOptimizer(
                    self_opt,
                    device=device,
                    base_fused=str(kv_fused),
                    base_decode_block=int(kv_decode_block),
                    log_callback=log_callback,
                )
            except (TypeError, ValueError, RuntimeError):
                decode_tuner = None

        # Prefill: append the prompt into caches and get logits for the last prompt token.
        with torch.no_grad():
            logits, caches2 = self.forward(prompt, caches=caches, pos_offset=0)
            caches = caches2 or caches

            out = prompt
            pos = int(T0)
            last_logits = logits[:, -1, :]

            for _ in range(int(max_new_tokens)):
                if decode_tuner is not None and caches:
                    try:
                        attn0 = cast(object, cast(Block, self.blocks[0]).attn)
                        plan = decode_tuner.maybe_get_plan(attn=attn0, cache=caches[0], prefix_len=int(caches[0].pos))
                        if plan is not None:
                            for cc in caches:
                                plan.apply_to_cache(cc)
                    except (TypeError, ValueError, RuntimeError, AttributeError):
                        pass

                next_tok = _sample_next_token(last_logits, temperature=float(temperature), top_k=top_k)
                out = torch.cat([out, next_tok], dim=1)

                logits2, caches3 = self.forward(next_tok, caches=caches, pos_offset=pos)
                caches = caches3 or caches
                pos += 1
                last_logits = logits2[:, -1, :]

        return out

    def generate_speculative(
        self,
        prompt: torch.Tensor,
        draft_model: "GPT",
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        spec_k: int = 4,
        spec_method: str = "reject_sampling",
        spec_extra_token: bool = False,
        spec_disable_below_accept: float = 0.0,
        self_opt: KVSelfOptConfig | None = None,
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",
        kv_cache_k: KVCacheKind | None = None,
        kv_cache_v: KVCacheKind | None = None,
        kv_cache_k_sem: KVCacheKind | None = None,
        kv_cache_k_geo: KVCacheKind | None = None,
        kv_qblock_k: int | None = None,
        kv_qblock_v: int | None = None,
        kv_qblock_k_sem: int | None = None,
        kv_qblock_k_geo: int | None = None,
        log_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> torch.Tensor:
        """Speculative decoding (draft proposes, main verifies) with cache-safe rollback."""
        if (not math_isfinite(float(temperature))) or float(temperature) <= 0.0:
            # Greedy / invalid temp: fall back to the standard path (speculative acceptance needs a
            # proper sampling distribution).
            return self.generate(
                prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=top_k,
                kv_cache=kv_cache,
                kv_qblock=kv_qblock,
                kv_residual=kv_residual,
                kv_decode_block=kv_decode_block,
                kv_fused=kv_fused,
                kv_cache_k=kv_cache_k,
                kv_cache_v=kv_cache_v,
                kv_cache_k_sem=kv_cache_k_sem,
                kv_cache_k_geo=kv_cache_k_geo,
                kv_qblock_k=kv_qblock_k,
                kv_qblock_v=kv_qblock_v,
                kv_qblock_k_sem=kv_qblock_k_sem,
                kv_qblock_k_geo=kv_qblock_k_geo,
                self_opt=self_opt,
                log_callback=log_callback,
            )
        if str(spec_method or "reject_sampling").strip().lower() not in ("reject_sampling",):
            raise ValueError(f"Unsupported spec_method: {spec_method!r}")
        if prompt.ndim != 2:
            raise ValueError(f"prompt must be (B,T) but got shape={tuple(prompt.shape)}")
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")

        # Build separate KV runtimes for main + draft (their configs may differ).
        # NOTE: We intentionally enable layerwise promotion for speculative decode (is_speculative=True).
        #       The non-speculative path stays in `generate()`.
        def _build_caches(m: GPT, *, is_speculative: bool) -> list[DecoupledLayerKVCache | LayerKVCache]:
            mode = GPT._normalize_attn_mode(getattr(m.cfg, "attn_mode", ""))
            batch_size = int(prompt.size(0))
            max_seq = int(prompt.size(1) + int(max_new_tokens))

            k_cfg, v_cfg, k_sem_cfg, k_geo_cfg = GPT._resolve_kv_tensor_cfgs(
                kv_cache=kv_cache,
                kv_qblock=int(kv_qblock),
                kv_residual=int(kv_residual),
                kv_cache_k=kv_cache_k,
                kv_cache_v=kv_cache_v,
                kv_cache_k_sem=kv_cache_k_sem,
                kv_cache_k_geo=kv_cache_k_geo,
                kv_qblock_k=kv_qblock_k,
                kv_qblock_v=kv_qblock_v,
                kv_qblock_k_sem=kv_qblock_k_sem,
                kv_qblock_k_geo=kv_qblock_k_geo,
            )

            promote_layers = None
            if mode == "decoupled":
                k_sem_cfg, k_geo_cfg, v_cfg, promote_layers, _ = m.choose_kv_cache_policy(
                    model=m,
                    self_opt=self_opt,
                    device=prompt.device,
                    prompt=prompt,
                    k_sem_cfg=k_sem_cfg,
                    k_geo_cfg=k_geo_cfg,
                    v_dec_cfg=v_cfg,
                    kv_residual=int(kv_residual),
                    kv_decode_block=int(kv_decode_block),
                    kv_fused=str(kv_fused),
                    max_new_tokens=int(max_new_tokens),
                    is_speculative=bool(is_speculative),
                )

            return m.build_layer_caches(
                model=m,
                mode=str(mode),
                batch_size=int(batch_size),
                max_seq_len=int(max_seq),
                device=prompt.device,
                k_cfg=k_cfg,
                v_cfg=v_cfg,
                k_sem_cfg=k_sem_cfg,
                k_geo_cfg=k_geo_cfg,
                promote_layers=promote_layers,
                kv_decode_block=int(kv_decode_block),
                kv_fused=str(kv_fused),
            )

        main_caches = _build_caches(self, is_speculative=True)
        draft_caches = _build_caches(draft_model, is_speculative=True)

        out = prompt
        pos = int(prompt.size(1))

        # Prefill both (writes caches).
        with torch.no_grad():
            main_logits, main_caches2 = self.forward(out, caches=main_caches, pos_offset=0)
            draft_logits, draft_caches2 = draft_model.forward(out, caches=draft_caches, pos_offset=0)
            main_caches = main_caches2 or main_caches
            draft_caches = draft_caches2 or draft_caches

            accept_total = 0
            prop_total = 0

            while int(out.size(1)) < int(prompt.size(1) + int(max_new_tokens)):
                remaining = int(prompt.size(1) + int(max_new_tokens) - int(out.size(1)))
                # Need at least 1 "verified" token per loop.
                k = int(max(1, min(int(spec_k), max(1, remaining - 1))))

                pos0 = int(main_caches[0].pos) if main_caches else int(pos)

                proposed: list[torch.Tensor] = []
                q_probs: list[torch.Tensor] = []
                for _ in range(int(k)):
                    tok, p = Metrics.sample(
                        draft_logits[:, -1, :],
                        temperature=float(temperature),
                        top_k=top_k,
                    )
                    proposed.append(tok)
                    q_probs.append(p)
                    draft_logits, draft_caches3 = draft_model.forward(tok, caches=draft_caches, pos_offset=pos + len(proposed) - 1)
                    draft_caches = draft_caches3 or draft_caches

                proposed_t = torch.cat(proposed, dim=1)
                main_block, main_caches3 = self.forward(proposed_t, caches=main_caches, pos_offset=pos)
                main_caches = main_caches3 or main_caches

                accepted_k, next_tok = Metrics.verify(
                    main_logits[:, -1, :],
                    main_block,
                    proposed_t,
                    q_probs,
                    temperature=float(temperature),
                    top_k=top_k,
                )
                accepted_k = int(accepted_k)

                prop_total += int(k)
                accept_total += int(max(0, accepted_k))

                # Roll back both caches to the pre-proposal position, then replay accepted + next token.
                for c in main_caches:
                    c.truncate(pos0)
                for c in draft_caches:
                    c.truncate(pos0)

                chunk = torch.cat([proposed_t[:, :accepted_k], next_tok], dim=1)
                main_logits, main_caches4 = self.forward(chunk, caches=main_caches, pos_offset=pos)
                draft_logits, draft_caches4 = draft_model.forward(chunk, caches=draft_caches, pos_offset=pos)
                main_caches = main_caches4 or main_caches
                draft_caches = draft_caches4 or draft_caches

                out = torch.cat([out, chunk], dim=1)
                pos = int(out.size(1))

                # Optional guard: disable speculative if acceptance collapses.
                if float(spec_disable_below_accept) > 0.0 and prop_total >= 16:
                    acc = float(accept_total) / float(max(1, prop_total))
                    if acc < float(spec_disable_below_accept):
                        # Fall back to non-speculative for the remainder.
                        rest = int(prompt.size(1) + int(max_new_tokens) - int(out.size(1)))
                        if rest > 0:
                            out = self.generate(
                                out,
                                max_new_tokens=rest,
                                temperature=float(temperature),
                                top_k=top_k,
                                kv_cache=kv_cache,
                                kv_qblock=kv_qblock,
                                kv_residual=kv_residual,
                                kv_decode_block=kv_decode_block,
                                kv_fused=kv_fused,
                                kv_cache_k=kv_cache_k,
                                kv_cache_v=kv_cache_v,
                                kv_cache_k_sem=kv_cache_k_sem,
                                kv_cache_k_geo=kv_cache_k_geo,
                                kv_qblock_k=kv_qblock_k,
                                kv_qblock_v=kv_qblock_v,
                                kv_qblock_k_sem=kv_qblock_k_sem,
                                kv_qblock_k_geo=kv_qblock_k_geo,
                                self_opt=self_opt,
                                log_callback=log_callback,
                            )
                        break

                _ = spec_extra_token  # reserved for future methods (kept for CLI compatibility)

        return out

    def choose_kv_cache_policy(
        self,
        *,
        model: "GPT",
        self_opt: KVSelfOptConfig | None,
        device: torch.device,
        prompt: torch.Tensor,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_dec_cfg: KVCacheTensorConfig,
        kv_residual: int,
        kv_decode_block: int,
        kv_fused: str,
        max_new_tokens: int,
        is_speculative: bool,
    ) -> tuple[KVCacheTensorConfig, KVCacheTensorConfig, KVCacheTensorConfig, int | None, int]:
        """
        Public wrapper for cache-policy selection.

        Some lint configurations treat calling a protected member on an arbitrary instance as a
        "client" access; this wrapper allows internal helpers to call policy selection without
        referencing a protected name.
        """
        return self._choose_kv_cache_policy(
            model=model,
            self_opt=self_opt,
            device=device,
            prompt=prompt,
            k_sem_cfg=k_sem_cfg,
            k_geo_cfg=k_geo_cfg,
            v_dec_cfg=v_dec_cfg,
            kv_residual=kv_residual,
            kv_decode_block=kv_decode_block,
            kv_fused=kv_fused,
            max_new_tokens=max_new_tokens,
            is_speculative=is_speculative,
        )

    def _choose_kv_cache_policy(
        self,
        *,
        model: "GPT",
        self_opt: KVSelfOptConfig | None,
        device: torch.device,
        prompt: torch.Tensor,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_dec_cfg: KVCacheTensorConfig,
        kv_residual: int,
        kv_decode_block: int,
        kv_fused: str,
        max_new_tokens: int,
        is_speculative: bool,
    ) -> tuple[KVCacheTensorConfig, KVCacheTensorConfig, KVCacheTensorConfig, int | None, int]:
        """Choose decoupled KV cache policy, with optional quality gating and layerwise fallback."""
        # Resolve via the public `production.model` module *if already loaded* so tests can patch:
        # - production.model.KVCachePolicySelfOptimizer
        # - production.model.policy_quality_reject_reasons
        # - production.model.warn_policy_quality_reject
        # - production.model.KVCachePolicy
        #
        # Do NOT import `production.model` here (would create an import cycle).
        model_mod = _maybe_public_model_module()
        TunerCls = cast(
            type[KVCachePolicySelfOptimizer],
            getattr(model_mod, "KVCachePolicySelfOptimizer", KVCachePolicySelfOptimizer)
            if model_mod is not None
            else KVCachePolicySelfOptimizer,
        )
        reject_reasons_fn = cast(
            Callable[..., list[str]],
            getattr(model_mod, "policy_quality_reject_reasons", policy_quality_reject_reasons)
            if model_mod is not None
            else policy_quality_reject_reasons,
        )
        warn_reject_fn = cast(
            Callable[..., object],
            getattr(model_mod, "warn_policy_quality_reject", warn_policy_quality_reject)
            if model_mod is not None
            else warn_policy_quality_reject,
        )
        PolicyCls = cast(
            type[KVCachePolicy],
            getattr(model_mod, "KVCachePolicy", KVCachePolicy) if model_mod is not None else KVCachePolicy,
        )

        if self_opt is None:
            return k_sem_cfg, k_geo_cfg, v_dec_cfg, None, int(kv_residual)
        if str(getattr(self_opt, "mode", "none")) == "none":
            return k_sem_cfg, k_geo_cfg, v_dec_cfg, None, int(kv_residual)
        if str(getattr(self_opt, "scope", "all")) not in ("cache", "all"):
            return k_sem_cfg, k_geo_cfg, v_dec_cfg, None, int(kv_residual)
        if GPT._normalize_attn_mode(getattr(model.cfg, "attn_mode", "")) != "decoupled":
            return k_sem_cfg, k_geo_cfg, v_dec_cfg, None, int(kv_residual)

        # Construct base/override policies.
        base_policy = PolicyCls(
            k_sem_kind=k_sem_cfg.kind,
            k_geo_kind=k_geo_cfg.kind,
            v_kind=v_dec_cfg.kind,
            k_sem_qblock=int(k_sem_cfg.qblock),
            k_geo_qblock=int(k_geo_cfg.qblock),
            v_qblock=int(v_dec_cfg.qblock),
            residual_len=int(kv_residual),
        )

        # Instantiate policy tuner.
        attn0 = cast(object, cast(Block, model.blocks[0]).attn)
        tuner = TunerCls(
            self_opt,
            device=device,
            attn=attn0,
            model_cfg=model.cfg,
            batch_size=int(prompt.size(0)),
            max_seq_len=int(prompt.size(1) + int(max_new_tokens)),
            base_policy=base_policy,
            base_decode_block=int(kv_decode_block),
            base_fused=str(kv_fused),
        )

        want_quality = bool(getattr(self_opt, "policy_quality", False))
        want_long = bool(getattr(self_opt, "policy_quality_long", False))
        want_needle = bool(getattr(self_opt, "policy_quality_needle", False))
        allow_layerwise = bool(getattr(self_opt, "layerwise_cache", False))

        def _gate_tokens(*, spec: object | None, total_len: int) -> torch.Tensor:
            """Build a deterministic calibration sequence for quality gates.

            Priority:
            - explicit token spec (self_opt.calib_tokens / calib_long_tokens)
            - prompt (tiled to required length)
            """
            B = int(prompt.size(0))
            total_len = int(max(2, total_len))

            if spec is not None:
                try:
                    ids = load_token_ids_spec(str(spec))
                    if len(ids) >= total_len:
                        ids2 = ids[:total_len]
                        vocab = int(getattr(model.cfg, "vocab_size", 0) or 0)
                        if ids2 and vocab > 0 and (min(ids2) >= 0) and (max(ids2) < vocab):
                            t = torch.tensor(ids2, device=prompt.device, dtype=torch.long).unsqueeze(0)
                            return t.repeat(B, 1)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

            # Fallback: tile the prompt to reach required length.
            T0 = int(prompt.size(1))
            if T0 <= 0:
                # Should not happen; create a tiny deterministic filler.
                return torch.zeros((B, total_len), device=prompt.device, dtype=torch.long)
            if T0 >= total_len:
                return prompt[:, :total_len]
            # If the prompt is very short, tiling creates a degenerate calibration signal.
            # Use a deterministic pseudo-random token stream instead (CPU-generated for reproducibility).
            vocab = int(getattr(model.cfg, "vocab_size", 0) or 0)
            if vocab > 0 and T0 < int(max(16, min(256, total_len // 8))):
                try:
                    gen = torch.Generator(device="cpu")
                    seed = int(vocab) ^ (int(total_len) << 1) ^ (int(getattr(model.cfg, "d_model", 0) or 0) << 2)
                    _ = gen.manual_seed(int(seed) & 0xFFFFFFFF)
                    ids = torch.randint(0, int(vocab), (1, total_len), generator=gen, dtype=torch.long)
                    return ids.to(device=prompt.device).repeat(B, 1)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            reps = (total_len + T0 - 1) // T0
            return prompt.repeat(1, int(reps))[:, :total_len]

        def _make_induction_prompt(*, base: torch.Tensor, prefill: int, pattern_len: int) -> torch.Tensor:
            """Embed an induction/needle pattern into an existing token sequence.

            Structure:
            - early:    [pat][cont]
            - late:     [pat] and we score the next-token prediction for [cont]
            """
            if base.ndim != 2:
                return base
            B, T = int(base.size(0)), int(base.size(1))
            pre = int(prefill)
            if T < pre + 2:
                return base
            pat_len = int(max(2, min(int(pattern_len), pre - 2)))
            # Place the first occurrence early and the second at the end of prefill.
            pos_a = int(max(1, pre // 8))
            pos_b = int(pre - pat_len + 1)
            if pos_a + pat_len >= pos_b:
                pos_a = int(max(1, pos_b - pat_len - 1))
            if pos_a + pat_len >= pre or pos_b < 1:
                return base

            # Use a single shared pattern across the batch (reduces variance, improves repeatability).
            pat = base[0, pos_a : pos_a + pat_len].clone()
            cont = int(base[0, pos_a + pat_len].item())

            out = base.clone()
            out[:, pos_a : pos_a + pat_len] = pat.unsqueeze(0).repeat(B, 1)
            out[:, pos_a + pat_len] = int(cont)
            out[:, pos_b : pos_b + pat_len] = pat.unsqueeze(0).repeat(B, 1)
            # Retrieval target: after the second pattern.
            out[:, pre + 1] = int(cont)
            return out

        # Candidates:
        # - if quality is enabled, probe the cached/best policy first and only expand to a shortlist on failure
        #   (keeps steady-state overhead near-zero while still allowing "try-next-best" behavior).
        try:
            primary = tuner.choose_policy(prompt_len=int(prompt.size(1)))
        except Exception:  # pylint: disable=broad-exception-caught
            primary = base_policy

        def _short_gate(policy: KVCachePolicy) -> list[str]:
            compute_kl = bool(getattr(self_opt, "quality_compute_kl", False)) or (
                getattr(self_opt, "quality_kl_tol", None) is not None
            )
            pre = int(getattr(self_opt, "calib_prefill", 128))
            dec = int(getattr(self_opt, "calib_decode_steps", 32))
            gate_prompt = _gate_tokens(spec=getattr(self_opt, "calib_tokens", None), total_len=int(pre + dec + 1))
            metrics = self._policy_quality_metrics_decoupled(
                model=model,
                prompt=gate_prompt,
                policy=policy,
                prefill=pre,
                decode_steps=dec,
                compute_kl=bool(compute_kl),
            )
            return reject_reasons_fn(
                metrics,
                max_abs_logit_tol=getattr(self_opt, "quality_tol", None),
                delta_nll_tol=getattr(self_opt, "quality_delta_nll_tol", None),
                ppl_ratio_tol=getattr(self_opt, "quality_ppl_ratio_tol", None),
                kl_tol=getattr(self_opt, "quality_kl_tol", None),
            )

        def _long_gate(policy: KVCachePolicy) -> list[str]:
            compute_kl = bool(getattr(self_opt, "quality_long_compute_kl", False)) or (
                getattr(self_opt, "quality_long_kl_tol", None) is not None
            )
            pre = int(getattr(self_opt, "calib_long_prefill", 4096))
            dec = int(getattr(self_opt, "calib_long_decode_steps", 128))
            spec = getattr(self_opt, "calib_long_tokens", None)
            if spec is None:
                spec = getattr(self_opt, "calib_tokens", None)
            gate_prompt = _gate_tokens(spec=spec, total_len=int(pre + dec + 1))
            metrics = self._policy_quality_metrics_decoupled(
                model=model,
                prompt=gate_prompt,
                policy=policy,
                prefill=pre,
                decode_steps=dec,
                compute_kl=bool(compute_kl),
            )
            max_abs_logit_tol = getattr(self_opt, "quality_long_tol", None)
            if max_abs_logit_tol is None:
                max_abs_logit_tol = getattr(self_opt, "quality_tol", None)
            delta_nll_tol = getattr(self_opt, "quality_long_delta_nll_tol", None)
            if delta_nll_tol is None:
                delta_nll_tol = getattr(self_opt, "quality_delta_nll_tol", None)
            ppl_ratio_tol = getattr(self_opt, "quality_long_ppl_ratio_tol", None)
            if ppl_ratio_tol is None:
                ppl_ratio_tol = getattr(self_opt, "quality_ppl_ratio_tol", None)
            kl_tol = getattr(self_opt, "quality_long_kl_tol", None)
            if kl_tol is None:
                kl_tol = getattr(self_opt, "quality_kl_tol", None)
            return reject_reasons_fn(
                metrics,
                max_abs_logit_tol=max_abs_logit_tol,
                delta_nll_tol=delta_nll_tol,
                ppl_ratio_tol=ppl_ratio_tol,
                kl_tol=kl_tol,
            )

        def _needle_gate(policy: KVCachePolicy) -> list[str]:
            # Use long-horizon prefill by default; decode only one step (retrieval point).
            pre = self_opt.calib_needle_prefill
            pre_i = int(pre) if pre is not None else int(self_opt.calib_long_prefill)
            pre_i = int(max(32, pre_i))
            spec = self_opt.calib_needle_tokens or self_opt.calib_long_tokens or self_opt.calib_tokens

            base = _gate_tokens(spec=spec, total_len=int(pre_i + 2))
            pat_len_raw = self_opt.needle_pattern_len
            if pat_len_raw is not None:
                pat_len = int(pat_len_raw)
            else:
                pat_len = int(max(8, min(64, pre_i // 128)))
            gate_prompt = _make_induction_prompt(base=base, prefill=pre_i, pattern_len=pat_len)

            kl_tol = self_opt.quality_needle_kl_tol
            if kl_tol is None:
                kl_tol = self_opt.quality_long_kl_tol
            if kl_tol is None:
                kl_tol = self_opt.quality_kl_tol
            compute_kl = bool(self_opt.quality_needle_compute_kl) or (kl_tol is not None)

            metrics = self._policy_quality_metrics_decoupled(
                model=model,
                prompt=gate_prompt,
                policy=policy,
                prefill=int(pre_i),
                decode_steps=1,
                compute_kl=bool(compute_kl),
            )

            max_abs_logit_tol = self_opt.quality_needle_tol
            if max_abs_logit_tol is None:
                max_abs_logit_tol = self_opt.quality_long_tol
            if max_abs_logit_tol is None:
                max_abs_logit_tol = self_opt.quality_tol

            delta_nll_tol = self_opt.quality_needle_delta_nll_tol
            if delta_nll_tol is None:
                delta_nll_tol = self_opt.quality_long_delta_nll_tol
            if delta_nll_tol is None:
                delta_nll_tol = self_opt.quality_delta_nll_tol

            ppl_ratio_tol = self_opt.quality_needle_ppl_ratio_tol
            if ppl_ratio_tol is None:
                ppl_ratio_tol = self_opt.quality_long_ppl_ratio_tol
            if ppl_ratio_tol is None:
                ppl_ratio_tol = self_opt.quality_ppl_ratio_tol

            return reject_reasons_fn(
                metrics,
                max_abs_logit_tol=max_abs_logit_tol,
                delta_nll_tol=delta_nll_tol,
                ppl_ratio_tol=ppl_ratio_tol,
                kl_tol=kl_tol,
            )

        chosen: KVCachePolicy | None = None
        rejected: set[str] = set()

        def _is_intrinsically_safe_base(pol: KVCachePolicy) -> bool:
            return bool(
                pol.k_sem_kind in ("fp16", "fp32")
                and pol.k_geo_kind in ("fp16", "fp32")
                and pol.v_kind in ("fp16", "fp32")
            )

        def _gate_and_maybe_choose(cand: KVCachePolicy) -> bool:
            nonlocal chosen
            cshort = cand.short()
            if cshort in rejected:
                return False
            if (
                cshort == base_policy.short()
                and (bool(getattr(self_opt, "trust_base_policy", False)) or _is_intrinsically_safe_base(cand))
            ):
                chosen = cand
                return True

            if (
                cshort in self._policy_quality_ok
                and ((not want_long) or (cshort in self._policy_quality_long_ok))
                and ((not want_needle) or (cshort in self._policy_quality_needle_ok))
            ):
                chosen = cand
                return True

            reasons = _short_gate(cand) if want_quality else []
            if reasons:
                rejected.add(cshort)
                try:
                    _ = warn_reject_fn(chosen=cshort, fallback=base_policy.short(), reasons=reasons)
                except (TypeError, ValueError, RuntimeError):
                    pass
                return False

            if want_long:
                reasons2 = _long_gate(cand)
                if reasons2:
                    rejected.add(cshort)
                    try:
                        _ = warn_reject_fn(chosen=cshort, fallback=base_policy.short(), reasons=reasons2)
                    except (TypeError, ValueError, RuntimeError):
                        pass
                    return False
                self._policy_quality_long_ok.add(cshort)

            if want_needle:
                reasons3 = _needle_gate(cand)
                if reasons3:
                    rejected.add(cshort)
                    try:
                        _ = warn_reject_fn(chosen=cshort, fallback=base_policy.short(), reasons=reasons3)
                    except (TypeError, ValueError, RuntimeError):
                        pass
                    return False
                self._policy_quality_needle_ok.add(cshort)

            self._policy_quality_ok.add(cshort)
            chosen = cand
            return True

        # Phase A: try primary only (fast path). Do not immediately fall back to base if it fails;
        # instead expand to a shortlist so we can prefer other global candidates over layerwise hacks.
        if not want_quality:
            chosen = primary
        else:
            _ = _gate_and_maybe_choose(primary)

        # Phase B: on failure, expand to shortlist (slow path).
        if chosen is None and want_quality:
            try:
                shortlist = list(tuner.shortlist_policies(prompt_len=int(prompt.size(1)), max_candidates=8))
            except (TypeError, ValueError, RuntimeError):
                shortlist = []
            # Ensure base is present (last resort).
            if all(p.short() != base_policy.short() for p in shortlist):
                shortlist.append(base_policy)
            for cand in shortlist:
                if _gate_and_maybe_choose(cand):
                    break

        if chosen is None:
            chosen = base_policy
        if want_quality and chosen.short() == base_policy.short():
            # Persist the safe fallback so we don't repeatedly re-probe bad regions.
            try:
                tuner.update_cached_policy(base_policy)
            except (TypeError, ValueError, RuntimeError):
                pass

        promote_layers = None
        if want_quality and allow_layerwise and is_speculative:
            # Only attempt layerwise promotion after global candidates fail.
            if chosen.short() == base_policy.short():
                print("[selfopt] layerwise cache-policy enabled for speculative decode: trying promote={1,2,4,8,n_layer}")
                n_layer = int(getattr(model.cfg, "n_layer", 0) or 0)
                for n in [1, 2, 4, 8, n_layer]:
                    if n <= 0 or n > n_layer:
                        continue
                    metrics = self._policy_quality_metrics_decoupled(
                        model=model,
                        prompt=_gate_tokens(
                            spec=getattr(self_opt, "calib_tokens", None),
                            total_len=int(
                                int(getattr(self_opt, "calib_prefill", 128))
                                + int(getattr(self_opt, "calib_decode_steps", 32))
                                + 1
                            ),
                        ),
                        policy=primary,
                        promote_layers=int(n),
                        prefill=int(getattr(self_opt, "calib_prefill", 128)),
                        decode_steps=int(getattr(self_opt, "calib_decode_steps", 32)),
                        compute_kl=bool(getattr(self_opt, "quality_compute_kl", False))
                        or (getattr(self_opt, "quality_kl_tol", None) is not None),
                    )
                    reasons = reject_reasons_fn(
                        metrics,
                        max_abs_logit_tol=getattr(self_opt, "quality_tol", None),
                        delta_nll_tol=getattr(self_opt, "quality_delta_nll_tol", None),
                        ppl_ratio_tol=getattr(self_opt, "quality_ppl_ratio_tol", None),
                        kl_tol=getattr(self_opt, "quality_kl_tol", None),
                    )
                    if not reasons:
                        promote_layers = int(n)
                        chosen = primary
                        break

        # Apply chosen policy → tensor configs.
        chosen2 = chosen
        k_sem2, k_geo2, v2 = chosen2.to_tensor_cfgs()
        return k_sem2, k_geo2, v2, promote_layers, int(chosen2.residual_len)

    def _policy_quality_metrics_decoupled(
        self,
        *,
        model: "GPT",
        prompt: torch.Tensor,
        policy: "KVCachePolicy",
        promote_layers: int | None = None,
        prefill: int,
        decode_steps: int,
        compute_kl: bool = False,
    ) -> dict[str, float]:
        """Compute quality metrics for a decoupled policy vs fp16 baseline (best-effort)."""
        if prompt.ndim != 2:
            out = {"max_abs_logit": float("nan"), "delta_nll": float("nan"), "ppl_ratio": float("nan")}
            if compute_kl:
                out["kl_base_cand"] = float("nan")
            return out

        pre = int(max(0, prefill))
        dec = int(max(0, decode_steps))
        T = int(prompt.size(1))
        if T < 2:
            out = {"max_abs_logit": float("nan"), "delta_nll": float("nan"), "ppl_ratio": float("nan")}
            if compute_kl:
                out["kl_base_cand"] = float("nan")
            return out
        pre = int(min(pre, T - 1))
        dec = int(min(dec, T - pre - 1))
        if dec <= 0 or pre <= 0:
            out = {"max_abs_logit": float("nan"), "delta_nll": float("nan"), "ppl_ratio": float("nan")}
            if compute_kl:
                out["kl_base_cand"] = float("nan")
            return out

        promote = int(max(0, int(promote_layers or 0)))
        try:
            fp16 = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)
            ks, kg, vv = policy.to_tensor_cfgs()

            B = int(prompt.size(0))
            max_seq = int(pre + dec)
            base_caches = [
                Cache.build_layer(model.cfg, B, max_seq, prompt.device, k_sem=fp16, k_geo=fp16, v=fp16)
                for _ in range(int(getattr(model.cfg, "n_layer", 0) or 0))
            ]
            test_caches: list[DecoupledLayerKVCache | LayerKVCache] = []
            n_layer = int(getattr(model.cfg, "n_layer", 0) or 0)
            if promote <= 0:
                test_caches = [
                    Cache.build_layer(model.cfg, B, max_seq, prompt.device, k_sem=ks, k_geo=kg, v=vv)
                    for _ in range(n_layer)
                ]
            else:
                for i in range(n_layer):
                    use_fp16 = bool(i < promote)
                    test_caches.append(
                        Cache.build_layer(
                            model.cfg,
                            B,
                            max_seq,
                            prompt.device,
                            k_sem=(fp16 if use_fp16 else ks),
                            k_geo=(fp16 if use_fp16 else kg),
                            v=(fp16 if use_fp16 else vv),
                        )
                    )

            with torch.inference_mode():
                lb, base_caches2 = model.forward(prompt[:, :pre], caches=base_caches)
                lt, test_caches2 = model.forward(prompt[:, :pre], caches=test_caches)
                base_caches = base_caches2 or base_caches
                test_caches = test_caches2 or test_caches

                history: list[dict[str, float]] = []
                for i in range(pre, pre + dec):
                    x = prompt[:, i : i + 1]
                    lb, base_caches2 = model.forward(x, caches=base_caches, pos_offset=int(i))
                    lt, test_caches2 = model.forward(x, caches=test_caches, pos_offset=int(i))
                    base_caches = base_caches2 or base_caches
                    test_caches = test_caches2 or test_caches
                    history.append(Metrics.compare(lb, lt, prompt[:, i + 1], compute_kl=bool(compute_kl)))

            dnll = float(sum(float(h.get("delta_nll", 0.0)) for h in history) / float(len(history)))
            mx = float(max(float(h.get("max_abs_logit", 0.0)) for h in history))
            ppl_ratio = float(math.exp(dnll)) if math_isfinite(dnll) else float("nan")
            out: dict[str, float] = {"max_abs_logit": mx, "delta_nll": dnll, "ppl_ratio": ppl_ratio}
            if compute_kl:
                out["kl_base_cand"] = float(
                    sum(float(h.get("kl_base_cand", 0.0)) for h in history) / float(len(history))
                )
            return out
        except (TypeError, ValueError, RuntimeError):
            out = {"max_abs_logit": float("nan"), "delta_nll": float("nan"), "ppl_ratio": float("nan")}
            if compute_kl:
                out["kl_base_cand"] = float("nan")
            return out

def _sample_next_token(logits: torch.Tensor, *, temperature: float, top_k: int | None) -> torch.Tensor:
    """Sample a single next token from logits (B,V) → (B,1)."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be (B,V) but got shape={tuple(logits.shape)}")
    if not math_isfinite(temperature) or float(temperature) <= 0.0:
        # Greedy.
        return torch.argmax(logits, dim=-1, keepdim=True).to(torch.long)

    x = logits.float() / float(temperature)
    if top_k is not None:
        k = int(top_k)
        if k > 0 and k < int(x.size(-1)):
            vals, _idx = torch.topk(x, k, dim=-1)
            cutoff = vals[:, -1].unsqueeze(-1)
            x = torch.where(x >= cutoff, x, torch.full_like(x, -float("inf")))
    p = torch.softmax(x, dim=-1)
    return torch.multinomial(p, 1)


def math_isfinite(x: float) -> bool:
    try:
        return bool(math.isfinite(float(x)))
    except (TypeError, ValueError):
        return False
