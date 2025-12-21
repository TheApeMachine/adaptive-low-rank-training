from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from production.selfopt_cache import set_cache_entry
from production.selfopt_logging import append_jsonl
from production.selfopt_utils import device_sig, hash_cfg
from production.train_tuning import TrainBatchPlan, TrainCompilePlan, tune_batch_by_seq, tune_torch_compile


@dataclass(frozen=True)
class RuntimePlan:
    """Self-optimized runtime execution plan (no user overrides)."""

    # Numeric/precision plan
    param_dtype: torch.dtype
    amp_enabled: bool
    amp_dtype: torch.dtype

    # Sequence plan (architectural max lives in cfg.block_size; these are runtime feasible)
    train_seq_len: int
    eval_seq_len: int

    # Batch/compile plans
    batch_plan: TrainBatchPlan
    compile_plan: TrainCompilePlan

    # Optional debug metrics
    metrics: Dict[str, float]


class SelfOptController:
    """Always-on self-optimizer for runtime policy.

    This controller deliberately exposes *no* config surface: it derives decisions from device/model/data.
    """

    def __init__(self, *, cache_path: Optional[str], log_path: Optional[str], device: torch.device, cfg: Any) -> None:
        self.cache_path = str(cache_path) if cache_path else None
        self.log_path = str(log_path) if log_path else None
        self.device = device
        self.cfg = cfg

    # -------------------------
    # Capability probing
    # -------------------------
    @staticmethod
    def _supports_dtype(dev: torch.device, dt: torch.dtype) -> bool:
        try:
            x = torch.ones(8, device=dev, dtype=dt)
            y = (x * 1.0001).sum()
            _ = float(y.detach().to("cpu").item())
            return True
        except Exception:
            return False

    def choose_param_dtype(self) -> torch.dtype:
        # Conservative, practical defaults. (Advanced exploration can be added later without exposing knobs.)
        if self.device.type == "cuda":
            if self._supports_dtype(self.device, torch.bfloat16):
                return torch.bfloat16
            if self._supports_dtype(self.device, torch.float16):
                return torch.float16
            return torch.float32
        if self.device.type == "mps":
            return torch.float32
        return torch.float32

    def choose_amp(self) -> Tuple[bool, torch.dtype]:
        # Compute dtype for autocast.
        if self.device.type not in ("cuda", "mps"):
            return False, torch.bfloat16
        # Prefer bf16 when supported, else fp16.
        if self._supports_dtype(self.device, torch.bfloat16):
            dt = torch.bfloat16
        elif self._supports_dtype(self.device, torch.float16):
            dt = torch.float16
        else:
            return False, torch.bfloat16
        return True, dt

    # -------------------------
    # Planning
    # -------------------------
    def plan_runtime(
        self,
        *,
        model: torch.nn.Module,
        train_view: Any,
        val_view: Any,
        get_batch: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor]],
        train_seq_len_cap: int,
        eval_seq_len_cap: int,
    ) -> Tuple[torch.nn.Module, RuntimePlan]:
        """Return (possibly wrapped) model + the chosen runtime plan.

        `train_seq_len_cap` / `eval_seq_len_cap` should already reflect dataset split feasibility.
        """
        # 1) Precision plan
        t0 = time.perf_counter()
        param_dtype = self.choose_param_dtype()
        amp_enabled, amp_dtype = self.choose_amp()
        if param_dtype != torch.float32:
            try:
                model = model.to(dtype=param_dtype)
            except Exception:
                # If casting fails, stay fp32.
                param_dtype = torch.float32
        t_prec = time.perf_counter() - t0
        append_jsonl(
            self.log_path,
            {
                "type": "selfopt_precision",
                "device": str(self.device),
                "param_dtype": str(param_dtype),
                "amp_enabled": bool(amp_enabled),
                "amp_dtype": str(amp_dtype),
                "dt_s": float(t_prec),
            },
        )

        # 2) Seq plan (runtime feasible; architectural max is cfg.block_size)
        train_seq_len = int(max(2, min(int(getattr(self.cfg, "block_size", 0) or train_seq_len_cap), int(train_seq_len_cap))))
        eval_seq_len = int(max(2, min(int(getattr(self.cfg, "block_size", 0) or eval_seq_len_cap), int(eval_seq_len_cap), int(train_seq_len))))

        # 3) Batch plan by seq (throughput objective)
        seq_lens = sorted({int(x) for x in (train_seq_len, 1024, 2048) if int(x) > 0 and int(x) <= int(train_seq_len)})
        if not seq_lens:
            seq_lens = [int(train_seq_len)]
        t1 = time.perf_counter()
        batch_plan = tune_batch_by_seq(
            cache_path=self.cache_path,
            device=self.device,
            cfg=self.cfg,
            model=model,
            get_batch=get_batch,
            seq_lens=list(seq_lens),
            target_gbs=0,  # auto
            warmup=1,
            iters=2,
            verbose=False,
            amp_enabled=bool(amp_enabled),
            amp_dtype=amp_dtype,
        )
        t_batch = time.perf_counter() - t1
        append_jsonl(
            self.log_path,
            {
                "type": "selfopt_batch_plan",
                "device": str(self.device),
                "train_seq_len": int(train_seq_len),
                "seq_lens": list(seq_lens),
                "plan": {int(k): [int(v[0]), int(v[1])] for k, v in batch_plan.by_seq.items()},
                "dt_s": float(t_batch),
            },
        )

        # 4) Compile plan (training) — only after we know base shape.
        bs0, ga0 = batch_plan.by_seq.get(int(train_seq_len), next(iter(batch_plan.by_seq.values())))
        t2 = time.perf_counter()
        model2, compile_plan = tune_torch_compile(
            cache_path=self.cache_path,
            device=self.device,
            cfg=self.cfg,
            model=model,
            get_batch=get_batch,
            batch_size=int(bs0),
            grad_accum=int(ga0),
            seq_len=int(train_seq_len),
            mode="reduce-overhead",
            warmup=1,
            iters=2,
            hysteresis=0.03,
            verbose=False,
            amp_enabled=bool(amp_enabled),
            amp_dtype=amp_dtype,
        )
        t_compile = time.perf_counter() - t2
        append_jsonl(
            self.log_path,
            {
                "type": "selfopt_compile_plan",
                "device": str(self.device),
                "enabled": bool(getattr(compile_plan, "enabled", False)),
                "mode": str(getattr(compile_plan, "mode", "")),
                "train_seq_len": int(train_seq_len),
                "batch_size": int(bs0),
                "grad_accum": int(ga0),
                "dt_s": float(t_compile),
            },
        )

        metrics = {
            "t_precision_s": float(t_prec),
            "t_batch_tune_s": float(t_batch),
            "t_compile_tune_s": float(t_compile),
        }

        plan = RuntimePlan(
            param_dtype=param_dtype,
            amp_enabled=bool(amp_enabled),
            amp_dtype=amp_dtype,
            train_seq_len=int(train_seq_len),
            eval_seq_len=int(eval_seq_len),
            batch_plan=batch_plan,
            compile_plan=compile_plan,
            metrics=metrics,
        )

        # Persist the selected plan for observability (separate from decision logging todo).
        if self.cache_path:
            try:
                set_cache_entry(
                    self.cache_path,
                    section="runtime_plan",
                    key=f"{device_sig(self.device)}|train_runtime|cfg={hash_cfg(self.cfg)}|seq={int(train_seq_len)}",
                    value={"plan": asdict(plan), "ts": float(time.time())},
                )
            except Exception:
                pass

        append_jsonl(
            self.log_path,
            {
                "type": "selfopt_runtime_plan",
                "device": str(self.device),
                "train_seq_len": int(train_seq_len),
                "eval_seq_len": int(eval_seq_len),
                "plan": asdict(plan),
            },
        )

        return model2, plan


