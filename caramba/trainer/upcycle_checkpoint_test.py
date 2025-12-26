from __future__ import annotations

from pathlib import Path

import pytest

from caramba.trainer.upcycle import Upcycle


def test_validate_checkpoint_state_rejects_missing_keys(tmp_path: Path) -> None:
    bad: dict[str, object] = {"run_id": "r", "phase": "global", "step": 1}
    with pytest.raises(ValueError):
        Upcycle._validate_checkpoint_state(bad)

