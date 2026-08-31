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
    assert all(run["resolved_config"]["batch_size"] == 128 for run in plan)
    assert all(run["resolved_config"]["accumulation_steps"] == 1 for run in plan)
    assert all(run["resolved_config"]["num_workers"] == 8 for run in plan)


def test_plan_experiment_tag_creates_new_ids_without_changing_protocol():
    sweep = load_sweep(ROOT / "configs/sweeps/baselines.yaml")
    original = build_plan(sweep)
    tagged = build_plan(sweep, "retry1")
    assert all("_retry1_seed" in run["experiment_id"] for run in tagged)
    assert [run["seed"] for run in tagged] == [run["seed"] for run in original]
    assert [run["resolved_config"]["dataset"] for run in tagged] == [
        run["resolved_config"]["dataset"] for run in original
    ]
