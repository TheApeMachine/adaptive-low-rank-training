from __future__ import annotations

import torch

from caramba.runtime.activation import exceeds_activation_threshold, tensor_nbytes, tensors_nbytes


def test_tensor_nbytes() -> None:
    t = torch.zeros((2, 3), dtype=torch.float32)
    assert tensor_nbytes(t) == 2 * 3 * 4


def test_tensors_nbytes_sum() -> None:
    a = torch.zeros((2, 3), dtype=torch.float16)  # 2 bytes
    b = torch.zeros((1, 4), dtype=torch.int32)  # 4 bytes
    assert tensors_nbytes([a, b]) == (2 * 3 * 2) + (1 * 4 * 4)


def test_exceeds_activation_threshold() -> None:
    # 1MB threshold; tensor is ~2MB.
    t = torch.zeros((1024, 512), dtype=torch.float32)  # 1024*512*4 = 2MB
    assert exceeds_activation_threshold(tensors=[t], threshold_mb=1.0) is True
    assert exceeds_activation_threshold(tensors=[t], threshold_mb=3.0) is False

