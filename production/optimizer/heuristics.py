"""Heuristics for intent → concrete architecture/training defaults."""

from __future__ import annotations

import math


class HeuristicPlanner:
    """Pure heuristics for selecting architecture/training defaults."""

    @staticmethod
    def derive_layers_from_params(target_params: int) -> int:
        """
        Deterministic depth from budget.

        Why:
        - Keep depth selection invariant-ish (budget-driven, not device quirks).
        - Must stay consistent with `ModelConfig.optimize()` which ultimately owns
          the architecture sizing.
        """
        tp = int(max(1, target_params))
        log_params = float(math.log10(float(tp)))
        # Budget-driven depth: slow, smooth growth with model size.
        return int(max(4, int(2.0 + 2.0 * log_params - 11.0)))

    @staticmethod
    def derive_lr_from_params(target_params: int) -> float:
        ref: float = 1_000_000_000.0
        base: float = 3e-4
        exp: float = 0.10
        tp: float = float(target_params)
        ratio: float = tp / ref
        scale: float = float(math.pow(ratio, -exp))
        return float(base * scale)

    @staticmethod
    def choose_layers(*, target_params: int, device_type: str) -> int:
        """Choose model depth. Kept for backward compatibility with older optimizer graphs."""
        _ = device_type  # depth selection is budget-driven, not device-driven
        return int(HeuristicPlanner.derive_layers_from_params(int(target_params)))
