"""
attention_math provides small, pure attention helpers.
"""

from __future__ import annotations

import torch
from torch import Tensor


def decoupled_qk_cat(
    *,
    q_sem: Tensor,
    q_geo: Tensor,
    k_sem: Tensor,
    k_geo: Tensor,
    sem_scale: float,
    geo_scale: float,
) -> tuple[Tensor, Tensor]:
    """
    Build composite (q_cat, k_cat) for decoupled attention.

    Guarantees score equivalence:
        (q_cat @ k_cat^T) == (q_sem @ k_sem^T) * sem_scale
                       + (q_geo @ k_geo^T) * geo_scale
    """
    q_cat = torch.cat(
        [q_sem * float(sem_scale), q_geo * float(geo_scale)],
        dim=-1,
    )
    k_cat = torch.cat([k_sem, k_geo], dim=-1)
    return q_cat, k_cat


