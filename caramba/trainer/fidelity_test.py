"""Unit tests for short-context fidelity checks (delta NLL / PPL ratio).

These tests run on CPU with tiny dummy models to ensure fidelity math and
threshold behavior are stable and fast in CI.
"""

from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from caramba.config.verify import FidelityVerifyConfig
from caramba.trainer.fidelity import (
    assert_fidelity_thresholds,
    compute_short_context_fidelity,
)


class _UniformLM(nn.Module):
    """A dummy LM that emits uniform logits (all zeros).

    Why this exists:
    - Uniform logits give a deterministic NLL of log(V) regardless of targets.
    - Using the same model for teacher and student should yield delta_nll ~ 0.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        b, t = x.shape
        return torch.zeros((b, t, self.vocab_size), dtype=torch.float32)


class _ConstantWinnerLM(nn.Module):
    """A dummy LM that always prefers a single token ID.

    Why this exists:
    - Lets us construct an obvious "teacher is better than student" scenario
      with controlled targets.
    """

    def __init__(self, vocab_size: int, winner_token: int, logit: float = 5.0) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.winner_token = int(winner_token)
        self.logit = float(logit)

    def forward(self, x: Tensor) -> Tensor:
        b, t = x.shape
        out = torch.zeros((b, t, self.vocab_size), dtype=torch.float32)
        out[..., self.winner_token] = self.logit
        return out


class FidelityMathTest(unittest.TestCase):
    """Tests for fidelity metric computation."""

    def test_identical_models_delta_near_zero(self) -> None:
        """Teacher==student should produce delta_nll≈0 and ppl_ratio≈1."""
        vocab = 11
        teacher = _UniformLM(vocab)
        student = _UniformLM(vocab)

        x = torch.randint(0, vocab, (2, 8), dtype=torch.long)
        y = torch.randint(0, vocab, (2, 8), dtype=torch.long)

        result = compute_short_context_fidelity(
            teacher=teacher,
            student=student,
            batches=[(x, y)],
        )

        self.assertAlmostEqual(result.delta_nll, 0.0, places=6)
        self.assertAlmostEqual(result.ppl_ratio, 1.0, places=6)

    def test_student_worse_increases_delta_and_ratio(self) -> None:
        """A worse student should have positive delta_nll and ppl_ratio>1."""
        vocab = 7
        teacher = _ConstantWinnerLM(vocab_size=vocab, winner_token=0, logit=6.0)
        student = _ConstantWinnerLM(vocab_size=vocab, winner_token=1, logit=6.0)

        x = torch.zeros((2, 8), dtype=torch.long)
        y = torch.zeros((2, 8), dtype=torch.long)  # targets always token 0

        result = compute_short_context_fidelity(
            teacher=teacher,
            student=student,
            batches=[(x, y)],
        )

        self.assertGreater(result.delta_nll, 0.0)
        self.assertGreater(result.ppl_ratio, 1.0)


class FidelityThresholdsTest(unittest.TestCase):
    """Tests for threshold enforcement behavior."""

    def test_fail_fast_raises(self) -> None:
        """With fail_fast=True, the first violation should raise."""
        vocab = 5
        teacher = _ConstantWinnerLM(vocab_size=vocab, winner_token=0, logit=6.0)
        student = _ConstantWinnerLM(vocab_size=vocab, winner_token=1, logit=6.0)
        x = torch.zeros((1, 4), dtype=torch.long)
        y = torch.zeros((1, 4), dtype=torch.long)

        result = compute_short_context_fidelity(
            teacher=teacher,
            student=student,
            batches=[(x, y)],
        )

        with self.assertRaises(ValueError):
            _ = assert_fidelity_thresholds(
                result=result,
                max_delta_nll=0.0,
                max_ppl_ratio=None,
                fail_fast=True,
            )

    def test_non_fatal_collects_violations(self) -> None:
        """With fail_fast=False, we should return violations without raising."""
        vocab = 5
        teacher = _ConstantWinnerLM(vocab_size=vocab, winner_token=0, logit=6.0)
        student = _ConstantWinnerLM(vocab_size=vocab, winner_token=1, logit=6.0)
        x = torch.zeros((1, 4), dtype=torch.long)
        y = torch.zeros((1, 4), dtype=torch.long)

        result = compute_short_context_fidelity(
            teacher=teacher,
            student=student,
            batches=[(x, y)],
        )

        violations = assert_fidelity_thresholds(
            result=result,
            max_delta_nll=0.0,
            max_ppl_ratio=1.0,
            fail_fast=False,
        )
        self.assertGreaterEqual(len(violations), 1)


class FidelityConfigValidationTest(unittest.TestCase):
    """Tests for FidelityVerifyConfig validation rules."""

    def test_requires_at_least_one_threshold(self) -> None:
        """Config should require max_delta_nll or max_ppl_ratio."""
        with self.assertRaises(ValueError):
            _ = FidelityVerifyConfig.model_validate(
                {
                    "type": "fidelity",
                    "batches": 2,
                }
            )

