from __future__ import annotations

from pathlib import Path

import pytest

from caramba.instrumentation.analysis import generate_analysis_png


def test_generate_analysis_png_best_effort(tmp_path: Path) -> None:
    plt = pytest.importorskip("matplotlib")
    _ = plt

    train = tmp_path / "train.jsonl"
    train.write_text(
        "\n".join(
            [
                '{"type":"metrics","ts":0,"pid":1,"run_id":"r","phase":"blockwise","step":1,"data":{"metrics":{"loss":1.0}}}',
                '{"type":"metrics","ts":0,"pid":1,"run_id":"r","phase":"global","step":1,"data":{"metrics":{"loss":0.9}}}',
                '{"type":"metrics","ts":0,"pid":1,"run_id":"r","phase":"verify_eval","step":0,"data":{"metrics":{"student_accuracy":0.5}}}',
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "analysis.png"
    generate_analysis_png(train, out)
    assert out.exists()

