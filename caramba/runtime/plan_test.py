from __future__ import annotations

from pathlib import Path

from caramba.runtime.plan import RuntimePlan, load_plan, make_plan_key, save_plan


def test_plan_roundtrip(tmp_path: Path) -> None:
    payload = {"a": 1, "b": {"c": 2}}
    key = make_plan_key(payload)
    plan = RuntimePlan(
        key=key,
        device="cpu",
        torch_version="x",
        dtype="float32",
        use_amp=False,
        amp_dtype="bfloat16",
        batch_size=4,
        compile=False,
        compile_mode="reduce-overhead",
    )
    path = tmp_path / "plan.json"
    save_plan(path, plan, payload=payload)
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.key == key
    assert loaded.dtype == "float32"
    assert loaded.batch_size == 4
