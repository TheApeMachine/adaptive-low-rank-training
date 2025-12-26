"""Runtime utilities for device-dependent planning and execution.

These helpers encapsulate logic that depends on the execution environment
(device type, torch version) while keeping the rest of the codebase manifest-driven.
"""

from __future__ import annotations

from caramba.runtime.activation import exceeds_activation_threshold, tensor_nbytes, tensors_nbytes
from caramba.runtime.plan import RuntimePlan, load_plan, make_plan_key, save_plan

__all__ = [
    "exceeds_activation_threshold",
    "RuntimePlan",
    "load_plan",
    "make_plan_key",
    "save_plan",
    "tensor_nbytes",
    "tensors_nbytes",
]

