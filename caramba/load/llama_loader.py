"""
llama_loader provides weight-loading utilities for Llama-style checkpoints.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from caramba.weight.attention_decoupled import DecoupledAttentionWeight
from caramba.weight.attention_llama import LlamaAttentionWeight


def load_torch_state_dict(path: Path) -> dict[str, Tensor]:
    """
    load_torch_state_dict loads a torch state_dict saved via torch.save().
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict state_dict, got {type(obj)!r}")
    if not all(isinstance(k, str) for k in obj.keys()):
        raise ValueError("Expected state_dict with string keys")
    if not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ValueError("Expected state_dict with Tensor values")
    return obj


def load_state_dict_mapped(
    model: nn.Module,
    *,
    state_dict: dict[str, Tensor],
    mapping: dict[str, str],
    strict: bool = True,
) -> None:
    """
    load_state_dict_mapped loads a state_dict into a model using an explicit key mapping.
    """
    if not mapping:
        raise ValueError("mapping must be non-empty")

    mapped: dict[str, Tensor] = {}
    for src, dst in mapping.items():
        if src not in state_dict:
            raise ValueError(f"Missing source key in state_dict: {src!r}")
        mapped[dst] = state_dict[src]

    missing, unexpected = model.load_state_dict(mapped, strict=bool(strict))
    if missing or unexpected:
        raise ValueError(
            "load_state_dict_mapped failed: "
            f"missing={list(missing)!r}, unexpected={list(unexpected)!r}"
        )


def init_decoupled_from_llama_attention(
    *,
    student: DecoupledAttentionWeight,
    teacher: LlamaAttentionWeight,
) -> None:
    """
    init_decoupled_from_llama_attention initializes DBA weights from teacher attention.
    """
    init_decoupled_from_qkvo(
        student=student,
        teacher_q=teacher.q_proj.weight,
        teacher_k=teacher.k_proj.weight,
        teacher_v=teacher.v_proj.weight,
        teacher_o=teacher.o_proj.weight,
        teacher_q_bias=teacher.q_proj.bias,
        teacher_k_bias=teacher.k_proj.bias,
        teacher_v_bias=teacher.v_proj.bias,
        teacher_o_bias=teacher.o_proj.bias,
    )


def init_decoupled_from_qkvo(
    *,
    student: DecoupledAttentionWeight,
    teacher_q: Tensor,
    teacher_k: Tensor,
    teacher_v: Tensor,
    teacher_o: Tensor,
    teacher_q_bias: Tensor | None = None,
    teacher_k_bias: Tensor | None = None,
    teacher_v_bias: Tensor | None = None,
    teacher_o_bias: Tensor | None = None,
) -> None:
    """
    init_decoupled_from_qkvo initializes decoupled attention weights from teacher Q/K/V/O.

    This supports only truncation/splitting when the student's per-head combined
    Q/K dimension does not exceed the teacher's head_dim.
    """
    _copy_dense(student.v_proj, teacher_v, teacher_v_bias)
    _copy_dense(student.o_proj, teacher_o, teacher_o_bias)

    head_dim = int(student.head_dim)
    sem_h = int(student.sem_head_dim)
    geo_h = int(student.geo_head_dim)
    combined = sem_h + geo_h
    if combined > head_dim:
        raise ValueError(
            "Cannot init decoupled Q/K from teacher: "
            f"sem_head_dim+geo_head_dim={combined} exceeds head_dim={head_dim}"
        )

    q_w = teacher_q.view(int(student.n_heads), head_dim, int(student.d_model))
    q_b = _view_bias(teacher_q_bias, int(student.n_heads), head_dim)
    _init_split(
        out_sem=student.q_sem,
        out_geo=student.q_geo,
        teacher_w=q_w,
        teacher_b=q_b,
        sem_h=sem_h,
        geo_h=geo_h,
    )

    k_w = teacher_k.view(int(student.n_kv_heads), head_dim, int(student.d_model))
    k_b = _view_bias(teacher_k_bias, int(student.n_kv_heads), head_dim)
    _init_split(
        out_sem=student.k_sem,
        out_geo=student.k_geo,
        teacher_w=k_w,
        teacher_b=k_b,
        sem_h=sem_h,
        geo_h=geo_h,
    )

    if student.gate_logit is not None:
        student.gate_logit.data.zero_()


def _copy_dense(dst: nn.Module, w: Tensor, b: Tensor | None) -> None:
    if not hasattr(dst, "weight"):
        raise ValueError(f"Expected DenseWeight-like dst, got {type(dst)!r}")
    if not isinstance(getattr(dst, "weight"), torch.Tensor):
        raise ValueError(
            f"Expected tensor weight, got {type(getattr(dst, 'weight'))!r}"
        )
    dst_w: Tensor = getattr(dst, "weight")
    if dst_w.shape != w.shape:
        raise ValueError(f"Weight shape mismatch: dst={dst_w.shape}, src={w.shape}")
    dst_w.data.copy_(w)

    dst_b = getattr(dst, "bias", None)
    if (dst_b is None) != (b is None):
        raise ValueError("Bias presence mismatch between dst and src")
    if dst_b is not None:
        if b is None:
            raise ValueError("Expected source bias to be present")
        if not isinstance(dst_b, torch.Tensor):
            raise ValueError(f"Expected tensor bias, got {type(dst_b)!r}")
        if dst_b.shape != b.shape:
            raise ValueError(f"Bias shape mismatch: dst={dst_b.shape}, src={b.shape}")
        dst_b.data.copy_(b)


def _view_bias(b: Tensor | None, heads: int, head_dim: int) -> Tensor | None:
    if b is None:
        return None
    if b.ndim != 1:
        raise ValueError(f"Expected 1D bias, got {b.shape}")
    if int(b.shape[0]) != int(heads * head_dim):
        raise ValueError(
            f"Expected bias of shape ({heads * head_dim},), got {b.shape}"
        )
    return b.view(int(heads), int(head_dim))


def _init_split(
    *,
    out_sem: nn.Module,
    out_geo: nn.Module,
    teacher_w: Tensor,
    teacher_b: Tensor | None,
    sem_h: int,
    geo_h: int,
) -> None:
    if not hasattr(out_sem, "weight") or not hasattr(out_geo, "weight"):
        raise ValueError("Expected DenseWeight-like out_sem/out_geo")

    w_sem: Tensor = getattr(out_sem, "weight")
    w_geo: Tensor = getattr(out_geo, "weight")

    # teacher_w: (H, head_dim, d_model)
    w_sem_src = teacher_w[:, : int(sem_h), :].reshape_as(w_sem)
    w_geo_src = teacher_w[:, int(sem_h) : int(sem_h + geo_h), :].reshape_as(w_geo)

    w_sem.data.copy_(w_sem_src)
    w_geo.data.copy_(w_geo_src)

    b_sem = getattr(out_sem, "bias", None)
    b_geo = getattr(out_geo, "bias", None)
    if (b_sem is None) != (teacher_b is None) or (b_geo is None) != (teacher_b is None):
        raise ValueError("Bias presence mismatch between student and teacher")
    if teacher_b is not None:
        if b_sem is None or b_geo is None:
            raise ValueError("Expected student biases to be present")
        if not isinstance(b_sem, torch.Tensor) or not isinstance(b_geo, torch.Tensor):
            raise ValueError("Expected tensor student biases")

        b_sem_src = teacher_b[:, : int(sem_h)].reshape_as(b_sem)
        b_geo_src = teacher_b[:, int(sem_h) : int(sem_h + geo_h)].reshape_as(b_geo)
        b_sem.data.copy_(b_sem_src)
        b_geo.data.copy_(b_geo_src)


