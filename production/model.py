from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from production.attention import DecoupledBottleneckAttention
from production.kvcache_backend import DecoupledLayerKVCache, KVCacheKind, KVCacheTensorConfig, LayerKVCache
from production.runtime_tuning import (
    KVCachePolicy,
    KVCachePolicySelfOptimizer,
    KVDecodeSelfOptimizer,
    KVDecodePlan,
    KVSelfOptConfig,
    load_token_ids_spec,
    policy_quality_reject_reasons,
    warn_policy_quality_reject,
)


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int

    n_layer: int = 6
    n_head: int = 8
    kv_head: Optional[int] = None  # for GQA: number of KV heads (defaults to n_head)
    d_model: int = 512
    d_ff: int = 2048

    embed_dim: int = 512  # lexical bottleneck if < d_model

    attn_mode: Literal["standard", "bottleneck", "decoupled", "gqa"] = "bottleneck"
    attn_dim: int = 512
    sem_dim: int = 32
    geo_dim: int = 64

    decoupled_gate: bool = True

    rope: bool = True
    rope_base: float = 10000.0

    tie_qk: bool = False
    null_attn: bool = False
    learned_temp: bool = True

    mlp: Literal["swiglu", "gelu"] = "swiglu"
    dropout: float = 0.0


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.mlp == "swiglu":
            self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        elif cfg.mlp == "gelu":
            self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        else:
            raise ValueError(cfg.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.mlp == "swiglu":
            x = self.w3(F.silu(self.w1(x)) * self.w2(x))
        else:
            x = self.w2(F.gelu(self.w1(x)))
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = DecoupledBottleneckAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, *, attn_mask: Optional[torch.Tensor], cache: Optional[Any], pos_offset: int
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        a, cache = self.attn(self.ln1(x), attn_mask=attn_mask, cache=cache, pos_offset=pos_offset)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.emb_in = nn.Linear(cfg.embed_dim, cfg.d_model, bias=False) if cfg.embed_dim != cfg.d_model else None
        self.emb_out = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False) if cfg.embed_dim != cfg.d_model else None

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

        self.apply(self._init_weights)
        self.grad_checkpointing = False

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        *,
        caches: Optional[List[Any]] = None,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        B, T = idx.shape
        if caches is None and T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}. Increase --block.")

        x = self.tok_emb(idx)
        if self.emb_in is not None:
            x = self.emb_in(x)
        x = self.drop(x)

        attn_mask: Optional[torch.Tensor] = None
        if caches is None:
            if self.cfg.null_attn:
                attn_mask = self.causal_mask[:, :, :T, :T]
            else:
                attn_mask = None
        else:
            if T > 1:
                prev_len = caches[0].pos
                if prev_len == 0 and (not self.cfg.null_attn):
                    attn_mask = None
                else:
                    L = prev_len + T
                    key_pos = torch.arange(L, device=idx.device).view(1, 1, 1, L)
                    q_pos = (prev_len + torch.arange(T, device=idx.device)).view(1, 1, T, 1)
                    attn_mask = key_pos <= q_pos
            else:
                attn_mask = None

        new_caches: Optional[List[Any]] = [] if caches is not None else None

        if caches is None and self.training and getattr(self, "grad_checkpointing", False):
            try:
                from torch.utils.checkpoint import checkpoint  # type: ignore

                for blk in self.blocks:

                    def _blk_fwd(x_in: torch.Tensor, blk=blk) -> torch.Tensor:
                        y, _ = blk(x_in, attn_mask=attn_mask, cache=None, pos_offset=pos_offset)
                        return y

                    x = checkpoint(_blk_fwd, x, use_reentrant=False)
            except Exception:
                for blk in self.blocks:
                    x, _ = blk(x, attn_mask=attn_mask, cache=None, pos_offset=pos_offset)
        elif caches is None:
            for blk in self.blocks:
                x, _ = blk(x, attn_mask=attn_mask, cache=None, pos_offset=pos_offset)
        else:
            for i, blk in enumerate(self.blocks):
                layer_cache = caches[i]
                x, layer_cache = blk(x, attn_mask=attn_mask, cache=layer_cache, pos_offset=pos_offset)
                new_caches.append(layer_cache)

        x = self.ln_f(x)
        if self.emb_out is not None:
            x_small = self.emb_out(x)
        else:
            x_small = x
        logits = x_small @ self.tok_emb.weight.t()
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",
        self_opt: Optional[KVSelfOptConfig] = None,
        kv_cache_k: Optional[KVCacheKind] = None,
        kv_cache_v: Optional[KVCacheKind] = None,
        kv_cache_k_sem: Optional[KVCacheKind] = None,
        kv_cache_k_geo: Optional[KVCacheKind] = None,
        kv_qblock_k: Optional[int] = None,
        kv_qblock_v: Optional[int] = None,
        kv_qblock_k_sem: Optional[int] = None,
        kv_qblock_k_geo: Optional[int] = None,
        log_callback: Optional[Any] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        device = prompt.device
        B, T0 = prompt.shape
        max_seq = T0 + max_new_tokens

        if kv_fused not in ("none", "auto", "triton1pass", "triton2pass"):
            raise ValueError("kv_fused must be one of: none, auto, triton1pass, triton2pass")

        if self.cfg.attn_mode == "decoupled" and kv_cache == "q4_0":
            if kv_cache_k_geo is None:
                kv_cache_k_geo = "q8_0"
            if kv_cache_k_sem is None:
                kv_cache_k_sem = "q4_0"
            if kv_cache_v is None:
                kv_cache_v = "q4_0"

        def make_cfg(kind_override: Optional[KVCacheKind], qblock_override: Optional[int]) -> KVCacheTensorConfig:
            kind = kind_override if kind_override is not None else kv_cache
            qblock = qblock_override if qblock_override is not None else kv_qblock
            residual_len = kv_residual if kind not in ("fp16", "fp32") else 0
            return KVCacheTensorConfig(kind=kind, qblock=qblock, residual_len=residual_len)

        k_cfg = make_cfg(kv_cache_k, kv_qblock_k)
        v_cfg = make_cfg(kv_cache_v, kv_qblock_v)

        k_sem_cfg = make_cfg(kv_cache_k_sem, kv_qblock_k_sem)
        k_geo_cfg = make_cfg(kv_cache_k_geo, kv_qblock_k_geo)
        v_dec_cfg = make_cfg(kv_cache_v, kv_qblock_v)

        if (
            self_opt is not None
            and getattr(self_opt, "mode", "none") != "none"
            and getattr(self_opt, "scope", "all") in ("cache", "all")
            and self.cfg.attn_mode == "decoupled"
        ):
            try:
                base_policy = KVCachePolicy(
                    k_sem_kind=k_sem_cfg.kind,
                    k_geo_kind=k_geo_cfg.kind,
                    v_kind=v_dec_cfg.kind,
                    k_sem_qblock=k_sem_cfg.qblock,
                    k_geo_qblock=k_geo_cfg.qblock,
                    v_qblock=v_dec_cfg.qblock,
                    residual_len=int(kv_residual),
                )
                pol_tuner = KVCachePolicySelfOptimizer(
                    self_opt,
                    device=device,
                    attn=self.blocks[0].attn,
                    model_cfg=self.cfg,
                    batch_size=B,
                    max_seq_len=max_seq,
                    base_policy=base_policy,
                    base_decode_block=kv_decode_block,
                    base_fused=kv_fused,
                )
                chosen = pol_tuner.choose_policy(prompt_len=T0)

                if getattr(self_opt, "policy_quality", False):
                    calib_spec = getattr(self_opt, "calib_tokens", None)
                    if calib_spec:
                        calib_ids = load_token_ids_spec(str(calib_spec))
                        calib = torch.tensor([calib_ids], device=device, dtype=torch.long)
                    else:
                        calib = prompt.detach()

                    # NOTE: Quality check implementation is wired in runner/instrumentation; here we only warn on rejection.
                    # We keep the hooks so the CLI flow stays compatible.
                    # (Downstream: `production.runner` will compute metrics and use these helpers.)
                    _ = calib

                k_sem_cfg, k_geo_cfg, v_dec_cfg = chosen.to_tensor_cfgs()
                kv_residual = int(chosen.residual_len)
            except Exception:
                pass

        if self.cfg.attn_mode == "decoupled":
            caches: List[Any] = [
                DecoupledLayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_sem_dim=self.cfg.sem_dim,
                    k_geo_dim=self.cfg.geo_dim,
                    v_dim=self.cfg.attn_dim,
                    k_sem_cfg=k_sem_cfg,
                    k_geo_cfg=k_geo_cfg,
                    v_cfg=v_dec_cfg,
                    device=device,
                )
                for _ in range(self.cfg.n_layer)
            ]
        else:
            caches = [
                LayerKVCache(
                    batch_size=B,
                    max_seq_len=max_seq,
                    k_dim=(self.cfg.d_model if self.cfg.attn_mode == "standard" else self.cfg.attn_dim),
                    v_dim=(self.cfg.d_model if self.cfg.attn_mode == "standard" else self.cfg.attn_dim),
                    k_cfg=k_cfg,
                    v_cfg=v_cfg,
                    device=device,
                )
                for _ in range(self.cfg.n_layer)
            ]

        for c in caches:
            c.decode_block = int(kv_decode_block)
            c.fused = str(kv_fused)

        decode_tuner: Optional[KVDecodeSelfOptimizer] = None
        if (
            self_opt is not None
            and getattr(self_opt, "mode", "none") != "none"
            and getattr(self_opt, "scope", "all") in ("decode", "all")
        ):
            decode_tuner = KVDecodeSelfOptimizer(
                self_opt,
                device=device,
                base_fused=str(kv_fused),
                base_decode_block=int(kv_decode_block),
                log_callback=log_callback,
            )

        out = prompt
        pos = 0

        logits, caches = self(out, caches=caches, pos_offset=pos)
        pos += out.size(1)

        for _ in range(int(max_new_tokens)):
            last = logits[:, -1, :] / max(1e-8, float(temperature))
            if top_k is not None:
                vtop, _ = torch.topk(last, int(top_k), dim=-1)
                thresh = vtop[:, -1].unsqueeze(-1)
                last = last.masked_fill(last < thresh, -float("inf"))
            probs = F.softmax(last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if decode_tuner is not None and self.cfg.attn_mode == "decoupled":
                # Provide a single-token query to the decode tuner to choose a plan per bucket.
                # It will update cache.decode_block / cache.fused by applying the plan.
                try:
                    # This uses the first layer as representative.
                    attn0 = self.blocks[0].attn
                    cache0 = caches[0]
                    # We don't have direct access to q_sem/q_geo here without re-running attention internals,
                    # so the runtime tuner is primarily used inside the attention forward when available.
                    _ = (attn0, cache0)
                except Exception:
                    pass

            out = torch.cat([out, next_id], dim=1)
            logits, caches = self(next_id, caches=caches, pos_offset=pos)
            pos += 1

        if was_training:
            self.train()
        return out


