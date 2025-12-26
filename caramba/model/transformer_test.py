"""
transformer_test provides tests to validate the
transformer model.
"""
from __future__ import annotations

import unittest

import torch

from caramba.model.transformer import TransformerModel
from caramba.config.topology import StackedTopologyConfig, NodeConfig
from caramba.config.layer import (
    LinearLayerConfig,
    LayerNormLayerConfig,
    DropoutLayerConfig,
    LayerType,
    AttentionLayerConfig,
)


class TransformerTest(unittest.TestCase):
    """
    TransformerTest provides tests to validate the
    transformer model.
    """
    def test_forward(self) -> None:
        """
        test the forward pass of the transformer model.
        """
        layers: list[NodeConfig] = [
            LinearLayerConfig(
                type=LayerType.LINEAR,
                d_in=128,
                d_out=128,
                bias=True,
            ),
            LayerNormLayerConfig(
                type=LayerType.LAYER_NORM,
                d_model=128,
                eps=1e-5,
            ),
            AttentionLayerConfig(
                type=LayerType.ATTENTION,
                d_model=128,
                n_heads=4,
                dropout_p=0.1,
            ),
            DropoutLayerConfig(
                type=LayerType.DROPOUT,
                p=0.1,
            ),
        ]
        transformer = TransformerModel(StackedTopologyConfig(layers=layers))

        x: torch.Tensor = torch.randn(1, 10, 128)
        self.assertEqual(transformer(x).shape, (1, 10, 128))

    def test_forward_with_activation_checkpointing(self) -> None:
        """Forward works when activation checkpointing is enabled on the topology."""
        layers: list[NodeConfig] = [
            LinearLayerConfig(type=LayerType.LINEAR, d_in=32, d_out=32, bias=True),
            AttentionLayerConfig(type=LayerType.ATTENTION, d_model=32, n_heads=4, dropout_p=0.0),
        ]
        transformer = TransformerModel(StackedTopologyConfig(layers=layers))
        # Topology is a StackedTopology; enable checkpointing.
        topo = transformer.topology
        if hasattr(topo, "activation_checkpointing"):
            setattr(topo, "activation_checkpointing", True)
            # Use a small positive threshold so >0 checks pass.
            setattr(topo, "activation_checkpoint_threshold_mb", 0.1)
        x = torch.randn(1, 8, 32, requires_grad=True)
        y = transformer(x)
        self.assertEqual(y.shape, (1, 8, 32))
        # Perform a backward pass to confirm autograd works with checkpointing.
        y.sum().backward()
        self.assertIsNotNone(x.grad, "Input should have gradients after backward pass")

    def test_gradient_equivalence_with_and_without_checkpointing(self) -> None:
        """Gradients should be numerically equivalent with and without activation checkpointing."""
        torch.manual_seed(42)

        layers: list[NodeConfig] = [
            LinearLayerConfig(type=LayerType.LINEAR, d_in=32, d_out=32, bias=True),
            AttentionLayerConfig(type=LayerType.ATTENTION, d_model=32, n_heads=4, dropout_p=0.0),
        ]

        # Create two identical models (same random seed for parameter initialization).
        torch.manual_seed(42)
        model_with_ckpt = TransformerModel(StackedTopologyConfig(layers=layers))
        torch.manual_seed(42)
        model_without_ckpt = TransformerModel(StackedTopologyConfig(layers=layers))

        # Enable checkpointing on the first model.
        topo_with = model_with_ckpt.topology
        if hasattr(topo_with, "activation_checkpointing"):
            setattr(topo_with, "activation_checkpointing", True)
            setattr(topo_with, "activation_checkpoint_threshold_mb", 0.1)

        # Ensure checkpointing is disabled on the second model.
        topo_without = model_without_ckpt.topology
        if hasattr(topo_without, "activation_checkpointing"):
            setattr(topo_without, "activation_checkpointing", False)

        # Use deterministic input for reproducibility.
        torch.manual_seed(123)
        x_with = torch.randn(1, 8, 32, requires_grad=True)
        x_without = x_with.detach().clone().requires_grad_(True)

        # Forward and backward with checkpointing.
        y_with = model_with_ckpt(x_with)
        y_with.sum().backward()

        # Forward and backward without checkpointing.
        y_without = model_without_ckpt(x_without)
        y_without.sum().backward()

        # Verify gradients exist and are non-zero.
        self.assertIsNotNone(x_with.grad)
        self.assertIsNotNone(x_without.grad)

        # After assertIsNotNone, we know these are not None, but use local variables
        # to satisfy the type checker.
        grad_with = x_with.grad
        grad_without = x_without.grad
        assert grad_with is not None, "x_with.grad should not be None after backward"
        assert grad_without is not None, "x_without.grad should not be None after backward"

        self.assertTrue(grad_with.norm().item() > 0, "Gradient norm should be positive")
        self.assertTrue(grad_without.norm().item() > 0, "Gradient norm should be positive")

        # Compare gradients for numerical equivalence.
        self.assertTrue(
            torch.allclose(grad_with, grad_without, atol=1e-5, rtol=1e-5),
            "Gradients should be numerically equivalent with and without checkpointing",
        )

        # Also compare model parameter gradients.
        for (name_with, param_with), (name_without, param_without) in zip(
            model_with_ckpt.named_parameters(), model_without_ckpt.named_parameters()
        ):
            self.assertEqual(name_with, name_without)
            if param_with.grad is not None and param_without.grad is not None:
                self.assertTrue(
                    torch.allclose(param_with.grad, param_without.grad, atol=1e-5, rtol=1e-5),
                    f"Parameter gradients for {name_with} should match",
                )