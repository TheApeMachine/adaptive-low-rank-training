"""
policy manages KV cache format selection and quality gates.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Protocol, cast
import torch
import torch.nn as nn

from production.kvcache_backend import (
    KVCacheTensorConfig,
    DecoupledLayerKVCache,
    LayerKVCache
)
from production.runtime_tuning import (
    KVCachePolicy,
    KVCachePolicySelfOptimizer,
    policy_quality_reject_reasons
)

from .block import Block
from .config import ModelConfig
from .metrics import Metrics
from .cache import Cache

if TYPE_CHECKING:
    from production.runtime_tuning import KVSelfOptConfig

class Model(Protocol):
    """Minimal protocol for models Policy can manage."""
    cfg: ModelConfig
    blocks: nn.ModuleList
    def forward(
        self,
        idx: torch.Tensor,
        *,
        caches: list[DecoupledLayerKVCache | LayerKVCache] | None = None,
        pos_offset: int = 0,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache] | None]:
        """Model forward used by Policy for calibration/gating."""
        raise NotImplementedError

class Policy:
    """Selection engine for KV cache configurations."""
    def __init__(self, model: Model):
        self.model: Model = model
        self.cfg: ModelConfig = model.cfg

    def select(
        self,
        prompt: torch.Tensor,
        self_opt: KVSelfOptConfig,
        *,
        k_sem: KVCacheTensorConfig,
        k_geo: KVCacheTensorConfig,
        v: KVCacheTensorConfig,
        residual: int,
        decode_block: int,
        fused: str,
        max_new_tokens: int = 0,
    ) -> tuple[KVCacheTensorConfig, KVCacheTensorConfig, KVCacheTensorConfig, int | None, int]:
        """Choose the optimal policy, falling back to layerwise promotion if needed.

        CRITIQUE.md alignment:
        - Phase A: evaluate global candidates (no promotion) and pick the first that passes.
        - Phase B: only if Phase A fails and layerwise is enabled, try promotion on the best candidate(s).
        """
        batch_size, prompt_len = prompt.shape
        max_seq = prompt_len + max_new_tokens

        base = KVCachePolicy(
            k_sem.kind, k_geo.kind, v.kind, k_sem.qblock, k_geo.qblock, v.qblock, residual
        )

        tuner = KVCachePolicySelfOptimizer(
            self_opt, device=prompt.device, attn=cast(Block, self.model.blocks[0]).attn,
            model_cfg=self.cfg, batch_size=batch_size, max_seq_len=max_seq,
            base_policy=base, base_decode_block=decode_block,
            base_fused=fused
        )

        if not getattr(self_opt, "policy_quality", False):
            chosen = tuner.choose_policy(prompt_len=prompt_len)
            ks, kg, vv = chosen.to_tensor_cfgs()
            return ks, kg, vv, None, chosen.residual_len

        # Phase A: global candidate search (no promotion).
        candidates = tuner.shortlist_policies(prompt_len=prompt_len, max_candidates=8)
        want_long = bool(getattr(self_opt, "policy_quality_long", False))
        for cand in candidates:
            if not self._gate(prompt, cand, self_opt, long=False):
                continue
            if want_long and (not self._gate(prompt, cand, self_opt, long=True)):
                continue
            ks, kg, vv = cand.to_tensor_cfgs()
            return ks, kg, vv, None, cand.residual_len

        # Phase B: layerwise promotion only after global candidates fail.
        if getattr(self_opt, "layerwise_cache", False) and candidates:
            cand0 = candidates[0]
            for n in [1, 2, 4, 8, self.cfg.n_layer]:
                if n > self.cfg.n_layer:
                    break
                if self._gate(prompt, cand0, self_opt, promote=n, long=False):
                    ks, kg, vv = cand0.to_tensor_cfgs()
                    return ks, kg, vv, n, cand0.residual_len

        ks, kg, vv = base.to_tensor_cfgs()
        return ks, kg, vv, None, base.residual_len

    def _gate(
        self,
        tokens: torch.Tensor,
        cand: KVCachePolicy,
        self_opt: KVSelfOptConfig,
        promote: int | None = None
        ,
        long: bool = False,
    ) -> bool:
        """Judge a candidate policy over a calibration window."""
        T = int(tokens.size(1))
        if T < 2:
            return False

        if long:
            pre = int(getattr(self_opt, "calib_long_prefill", 4096))
            dec = int(getattr(self_opt, "calib_long_decode_steps", 128))
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
            compute_kl = bool(getattr(self_opt, "quality_long_compute_kl", False)) or (kl_tol is not None)
        else:
            pre = int(getattr(self_opt, "calib_prefill", 128))
            dec = int(getattr(self_opt, "calib_decode_steps", 32))
            max_abs_logit_tol = getattr(self_opt, "quality_tol", None)
            delta_nll_tol = getattr(self_opt, "quality_delta_nll_tol", None)
            ppl_ratio_tol = getattr(self_opt, "quality_ppl_ratio_tol", None)
            kl_tol = getattr(self_opt, "quality_kl_tol", None)
            compute_kl = bool(getattr(self_opt, "quality_compute_kl", False)) or (kl_tol is not None)

        pre = int(min(max(0, pre), T - 1))
        dec = int(min(max(0, dec), T - pre - 1))
        if pre <= 0 or dec <= 0:
            return False
        ks, kg, vv = cand.to_tensor_cfgs()
        fp16 = KVCacheTensorConfig(kind="fp16", qblock=32, residual_len=0)

        def _build(cfg: KVCacheTensorConfig, l_idx: int) -> KVCacheTensorConfig:
            return fp16 if promote and l_idx < promote else cfg

        base_c = [
            Cache.build_layer(
                self.cfg,
                1,
                pre + dec,
                tokens.device,
                k_sem=fp16,
                k_geo=fp16,
                v=fp16,
            )
            for _ in range(self.cfg.n_layer)
        ]
        test_c = [
            Cache.build_layer(
                self.cfg,
                1,
                pre + dec,
                tokens.device,
                k_sem=_build(ks, i),
                k_geo=_build(kg, i),
                v=_build(vv, i),
            )
            for i in range(self.cfg.n_layer)
        ]

        history: list[dict[str, float]] = []
        # Return of forward is tuple[Tensor, list[Any] | None]
        res_b = cast(
            tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache]],
            self.model.forward(tokens[:, :pre], caches=base_c)
        )
        lb = res_b[0]
        res_t = cast(
            tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache]],
            self.model.forward(tokens[:, :pre], caches=test_c)
        )
        lt = res_t[0]

        for i in range(pre, pre + dec):
            x = tokens[:, i:i+1]
            res_b = cast(
                tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache]],
                self.model.forward(x, caches=base_c, pos_offset=i)
            )
            lb, base_c = res_b[0], res_b[1]
            res_t = cast(
                tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache]],
                self.model.forward(x, caches=test_c, pos_offset=i)
            )
            lt, test_c = res_t[0], res_t[1]
            history.append(Metrics.compare(lb, lt, tokens[:, i+1], compute_kl=bool(compute_kl)))

        agg = {
            "delta_nll": sum(h["delta_nll"] for h in history) / len(history),
            "max_abs_logit": max(h["max_abs_logit"] for h in history),
            "ppl_ratio": float(math.exp(sum(h["delta_nll"] for h in history) / len(history))),
        }
        if compute_kl:
            agg["kl_base_cand"] = sum(h.get("kl_base_cand", 0.0) for h in history) / len(history)
        return not bool(policy_quality_reject_reasons(
            agg,
            max_abs_logit_tol=max_abs_logit_tol,
            delta_nll_tol=delta_nll_tol,
            ppl_ratio_tol=ppl_ratio_tol,
            kl_tol=kl_tol,
        ))
