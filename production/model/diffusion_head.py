"""
Optional diffusion-based next-token head (adapter) conditioned on transformer features.

This is intentionally dependency-gated:
- If `diffusers` is not installed, the module still imports, but instantiation fails
  with a clear error message.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import math
from typing import cast

from typing_extensions import override

import torch
from torch import nn
import torch.nn.functional as F

DIFFUSERS_AVAILABLE: bool = importlib.util.find_spec("diffusers") is not None


@dataclass(frozen=True)
class DiffusionHeadConfig:
    enabled: bool = False
    num_train_timesteps: int = 1000
    num_infer_steps: int = 12
    time_embed_dim: int = 128
    mlp_mult: int = 4
    cfg_dropout_p: float = 0.10
    cfg_guidance_scale: float = 1.5
    scheduler: str = "ddim"  # "ddpm" | "ddim" | "dpm"
    loss_weight: float = 0.10


@dataclass(frozen=True)
class StepOutput:
    prev_sample: torch.Tensor


class DiffusersSchedulerAdapter:
    def __init__(self, inner: object) -> None:
        self._inner: object = inner

    @property
    def timesteps(self) -> torch.Tensor:
        ts = getattr(self._inner, "timesteps", None)
        if not isinstance(ts, torch.Tensor):
            raise TypeError("diffusers scheduler timesteps must be a torch.Tensor")
        return ts

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        fn = getattr(self._inner, "add_noise", None)
        if not callable(fn):
            raise TypeError("diffusers scheduler missing add_noise")
        out = fn(original_samples, noise, timesteps)
        if not isinstance(out, torch.Tensor):
            raise TypeError("diffusers scheduler add_noise must return torch.Tensor")
        return out

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device) -> None:
        fn = getattr(self._inner, "set_timesteps", None)
        if not callable(fn):
            raise TypeError("diffusers scheduler missing set_timesteps")
        _ = fn(int(num_inference_steps), device=device)

    def step(self, model_output: torch.Tensor, timestep: object, sample: torch.Tensor) -> StepOutput:
        fn = getattr(self._inner, "step", None)
        if not callable(fn):
            raise TypeError("diffusers scheduler missing step")
        out = fn(model_output, timestep, sample)
        prev = getattr(out, "prev_sample", None)
        if not isinstance(prev, torch.Tensor):
            raise TypeError("diffusers scheduler step output missing prev_sample torch.Tensor")
        return StepOutput(prev_sample=prev)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.dim: int = int(dim)

    @override
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            t = t.view(-1)
        half = int(self.dim // 2)
        device = t.device
        dtype = torch.float32
        denom = float(max(1, half))
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device, dtype=dtype) / denom)
        args = t.to(dtype=dtype).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class PerTokenDenoiser(nn.Module):
    def __init__(self, *, embed_dim: int, time_embed_dim: int, mlp_mult: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.embed_dim: int = int(embed_dim)
        in_dim = 2 * int(embed_dim) + int(time_embed_dim)
        hid = int(max(64, int(mlp_mult) * int(embed_dim)))
        self.in_ln: nn.LayerNorm = nn.LayerNorm(in_dim)
        self.fc1: nn.Linear = nn.Linear(in_dim, hid)
        self.fc2: nn.Linear = nn.Linear(hid, hid)
        self.fc3: nn.Linear = nn.Linear(hid, int(embed_dim))

    @override
    def forward(self, *, x_t: torch.Tensor, cond: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_t, cond, t_embed], dim=-1)
        h = cast(torch.Tensor, self.in_ln(h))
        h = F.silu(cast(torch.Tensor, self.fc1(h)))
        h = F.silu(cast(torch.Tensor, self.fc2(h)))
        return cast(torch.Tensor, self.fc3(h))


class DiffusionNextTokenHead(nn.Module):
    def __init__(self, *, embed_dim: int, cfg: DiffusionHeadConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError(
                "DiffusionNextTokenHead requires `diffusers` but it is not installed. "
                + "Install with: pip install diffusers"
            )
        self.embed_dim: int = int(embed_dim)
        self.cfg: DiffusionHeadConfig = cfg

        self.time_embed: SinusoidalTimeEmbedding = SinusoidalTimeEmbedding(int(cfg.time_embed_dim))
        self.time_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(int(cfg.time_embed_dim), int(cfg.time_embed_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.time_embed_dim), int(cfg.time_embed_dim)),
        )
        self.denoiser: PerTokenDenoiser = PerTokenDenoiser(
            embed_dim=self.embed_dim,
            time_embed_dim=int(cfg.time_embed_dim),
            mlp_mult=int(cfg.mlp_mult),
        )

        self._sched: DiffusersSchedulerAdapter = self._make_scheduler(cfg)

    @staticmethod
    def _make_scheduler(cfg: DiffusionHeadConfig) -> DiffusersSchedulerAdapter:
        mod = importlib.import_module("diffusers")
        sched = str(cfg.scheduler or "ddim").strip().lower()
        if sched == "ddpm":
            cls_obj: object | None = getattr(mod, "DDPMScheduler", None)
        elif sched == "dpm":
            cls_obj = getattr(mod, "DPMSolverMultistepScheduler", None)
        else:
            cls_obj = getattr(mod, "DDIMScheduler", None)
        if cls_obj is None or (not callable(cls_obj)):
            raise RuntimeError(f"Could not resolve diffusers scheduler class for scheduler={sched!r}")
        inner = cls_obj(num_train_timesteps=int(cfg.num_train_timesteps))
        return DiffusersSchedulerAdapter(inner=inner)

    def _maybe_drop_cond(self, cond: torch.Tensor) -> torch.Tensor:
        p = float(self.cfg.cfg_dropout_p)
        if p <= 0.0 or (not self.training):
            return cond
        B = int(cond.size(0))
        mask = (torch.rand((B, 1, 1), device=cond.device) >= p).to(dtype=cond.dtype)
        return cond * mask

    def diffusion_loss(self, *, cond: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        if target_emb.shape != cond.shape:
            raise ValueError(f"shape mismatch: cond={tuple(cond.shape)} target_emb={tuple(target_emb.shape)}")

        B = int(cond.size(0))
        device = cond.device
        t = torch.randint(0, int(self.cfg.num_train_timesteps), (B,), device=device, dtype=torch.int64)
        noise = torch.randn_like(target_emb)

        x_t = self._sched.add_noise(target_emb, noise, t)
        cond2 = self._maybe_drop_cond(cond)

        t_emb = cast(torch.Tensor, self.time_mlp(self.time_embed(t))).to(dtype=cond.dtype)
        t_emb = t_emb.unsqueeze(1).expand(-1, target_emb.size(1), -1)

        eps_hat = cast(torch.Tensor, self.denoiser(x_t=x_t, cond=cond2, t_embed=t_emb))
        return F.mse_loss(eps_hat, noise)



