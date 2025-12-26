from __future__ import annotations

import torch

from caramba.infer.token_view import TokenView


def test_token_view_append_slice_and_rollback() -> None:
    tv = TokenView.allocate(batch_size=2, max_len=8, device=torch.device("cpu"), dtype=torch.long)

    tv.append(torch.tensor([[1, 2], [3, 4]], dtype=torch.long))
    assert tv.length == 2
    assert tv.as_tensor().tolist() == [[1, 2], [3, 4]]

    tv.append(torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long))
    assert tv.length == 5
    assert tv.slice(1, 4).tolist() == [[2, 5, 6], [4, 8, 9]]

    tv.rollback(2)
    assert tv.length == 3
    assert tv.as_tensor().tolist() == [[1, 2, 5], [3, 4, 8]]

