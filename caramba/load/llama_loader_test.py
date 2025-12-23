"""
llama_loader_test provides tests for llama_loader utilities.
"""
from __future__ import annotations

import unittest
import torch

from caramba.load.llama_loader import init_decoupled_from_llama_attention
from caramba.weight.attention_decoupled import DecoupledAttentionWeight
from caramba.weight.attention_llama import LlamaAttentionWeight


class LlamaLoaderTest(unittest.TestCase):
    """
    LlamaLoaderTest provides tests for initialization from attention.
    """
    def test_init_decoupled_from_llama_attention_splits_qk(self) -> None:
        """
        test that Q/K are split per-head from teacher into sem/geo weights.
        """
        teacher = LlamaAttentionWeight(
            d_model=8,
            n_heads=2,
            n_kv_heads=1,
            rope_base=10000.0,
            rope_dim=4,
            bias=False,
        )
        student = DecoupledAttentionWeight(
            d_model=8,
            n_heads=2,
            n_kv_heads=1,
            sem_dim=4,
            geo_dim=4,
            rope_base=10000.0,
            rope_dim=2,
            bias=False,
            gate=True,
        )

        teacher.q_proj.weight.data.copy_(
            torch.arange(0, teacher.q_proj.weight.numel())
            .view_as(teacher.q_proj.weight)
            .float()
        )
        teacher.k_proj.weight.data.copy_(
            torch.arange(0, teacher.k_proj.weight.numel())
            .view_as(teacher.k_proj.weight)
            .float()
            + 1_000.0
        )
        teacher.v_proj.weight.data.copy_(torch.ones_like(teacher.v_proj.weight) * 2.0)
        teacher.o_proj.weight.data.copy_(torch.ones_like(teacher.o_proj.weight) * 3.0)

        init_decoupled_from_llama_attention(student=student, teacher=teacher)

        q = teacher.q_proj.weight.view(2, 4, 8)
        self.assertTrue(
            torch.equal(
                student.q_sem.weight,
                q[:, :2, :].reshape_as(student.q_sem.weight),
            )
        )
        self.assertTrue(
            torch.equal(
                student.q_geo.weight,
                q[:, 2:4, :].reshape_as(student.q_geo.weight),
            )
        )

        k = teacher.k_proj.weight.view(1, 4, 8)
        self.assertTrue(
            torch.equal(
                student.k_sem.weight,
                k[:, :2, :].reshape_as(student.k_sem.weight),
            )
        )
        self.assertTrue(
            torch.equal(
                student.k_geo.weight,
                k[:, 2:4, :].reshape_as(student.k_geo.weight),
            )
        )

        self.assertTrue(torch.equal(student.v_proj.weight, teacher.v_proj.weight))
        self.assertTrue(torch.equal(student.o_proj.weight, teacher.o_proj.weight))

        self.assertIsNotNone(student.gate_logit)
        self.assertTrue(torch.all(student.gate_logit == 0))


if __name__ == "__main__":
    unittest.main()


