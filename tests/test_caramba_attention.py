from __future__ import annotations

import unittest
from typing import cast

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(
        f"torch is required for these tests but is not available: {e}"
    )

from caramba.config.layer import AttentionLayerConfig, LayerType
from caramba.config.operation import AttentionOperationConfig, OperationType
from caramba.config.weight import (
    DecoupledAttentionWeightConfig,
    LlamaAttentionWeightConfig,
    WeightType,
)
from caramba.layer.attention import Attention
from caramba.operation.rope import RotaryEmbedding
from caramba.weight.attention_decoupled import DecoupledAttentionWeight
from caramba.weight.attention_llama import LlamaAttentionWeight


class TestCarambaAttention(unittest.TestCase):
    def test_llama_gqa_shapes_and_hooks(self) -> None:
        torch.manual_seed(0)

        cfg = AttentionLayerConfig(
            type=LayerType.ATTENTION,
            operation=AttentionOperationConfig(
                type=OperationType.ATTENTION,
                is_causal=True,
                dropout_p=0.0,
            ),
            weight=LlamaAttentionWeightConfig(
                type=WeightType.LLAMA_ATTENTION,
                d_model=16,
                n_heads=4,
                n_kv_heads=2,
                rope_base=10_000.0,
                rope_dim=4,
                bias=False,
            ),
        )

        layer = Attention(cfg)
        self.assertIsInstance(layer.weight, LlamaAttentionWeight)

        op_out: list[torch.Tensor] = []

        def hook_op(
            _mod: object,
            _inp: tuple[object, ...],
            out: object,
        ) -> None:
            self.assertIsInstance(out, torch.Tensor)
            op_out.append(cast(torch.Tensor, out))

        rope_inputs: list[torch.Tensor] = []

        def hook_rope(
            _mod: object,
            inp: tuple[object, ...],
            _out: object,
        ) -> None:
            self.assertEqual(len(inp), 1)
            x = inp[0]
            self.assertIsInstance(x, torch.Tensor)
            rope_inputs.append(cast(torch.Tensor, x))

        layer.operation.register_forward_hook(hook_op)
        w = cast(LlamaAttentionWeight, layer.weight)
        rope = w.rope
        self.assertIsInstance(rope, RotaryEmbedding)
        rope.register_forward_hook(hook_rope)

        x = torch.randn(2, 5, 16)
        y = layer(x)

        self.assertEqual(tuple(y.shape), (2, 5, 16))

        self.assertEqual(len(op_out), 1)
        self.assertEqual(tuple(op_out[0].shape), (2, 4, 5, 4))

        self.assertEqual(len(rope_inputs), 2)
        self.assertEqual(int(rope_inputs[0].shape[-1]), 4)
        self.assertEqual(int(rope_inputs[1].shape[-1]), 4)

    def test_decoupled_rope_only_geo_shapes_and_hooks(self) -> None:
        torch.manual_seed(0)

        cfg = AttentionLayerConfig(
            type=LayerType.ATTENTION,
            operation=AttentionOperationConfig(
                type=OperationType.ATTENTION,
                is_causal=True,
                dropout_p=0.0,
            ),
            weight=DecoupledAttentionWeightConfig(
                type=WeightType.DECOUPLED_ATTENTION,
                d_model=16,
                n_heads=4,
                n_kv_heads=2,
                sem_dim=8,
                geo_dim=8,
                rope_base=10_000.0,
                rope_dim=2,
                bias=False,
                gate=True,
            ),
        )

        layer = Attention(cfg)
        self.assertIsInstance(layer.weight, DecoupledAttentionWeight)

        op_out: list[torch.Tensor] = []

        def hook_op(
            _mod: object,
            _inp: tuple[object, ...],
            out: object,
        ) -> None:
            self.assertIsInstance(out, torch.Tensor)
            op_out.append(cast(torch.Tensor, out))

        rope_inputs: list[torch.Tensor] = []

        def hook_rope(
            _mod: object,
            inp: tuple[object, ...],
            _out: object,
        ) -> None:
            self.assertEqual(len(inp), 1)
            x = inp[0]
            self.assertIsInstance(x, torch.Tensor)
            rope_inputs.append(cast(torch.Tensor, x))

        layer.operation.register_forward_hook(hook_op)
        w = cast(DecoupledAttentionWeight, layer.weight)
        rope = w.rope
        self.assertIsInstance(rope, RotaryEmbedding)
        rope.register_forward_hook(hook_rope)

        x = torch.randn(2, 5, 16)
        y = layer(x)

        self.assertEqual(tuple(y.shape), (2, 5, 16))

        self.assertEqual(len(op_out), 1)
        self.assertEqual(tuple(op_out[0].shape), (2, 4, 5, 4))

        self.assertEqual(len(rope_inputs), 2)
        self.assertEqual(tuple(rope_inputs[0].shape), (2, 4, 5, 2))
        self.assertEqual(tuple(rope_inputs[1].shape), (2, 2, 5, 2))


if __name__ == "__main__":
    unittest.main()


