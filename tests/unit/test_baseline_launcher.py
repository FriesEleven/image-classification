from pathlib import Path

from scripts.run_baselines import build_plan, load_sweep

ROOT = Path(__file__).resolve().parents[2]


def test_baseline_plan_contains_two_datasets_and_three_seeds():
    sweep = load_sweep(ROOT / "configs/sweeps/baselines.yaml")
    plan = build_plan(sweep)

    assert len(plan) == 6
    assert {run["seed"] for run in plan} == {42, 43, 44}
    assert {run["resolved_config"]["dataset"] for run in plan} == {"cifar10", "cifar100"}
    assert len({run["experiment_id"] for run in plan}) == 6
    assert all(run["resolved_config"]["epochs"] == 200 for run in plan)
