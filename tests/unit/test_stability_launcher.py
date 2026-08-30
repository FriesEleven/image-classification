from scripts.launch_cifar10_stability import (
    SWEEP_PATH,
    build_command,
    validated_plan,
)


def test_stability_launcher_contains_three_variants_and_two_missing_seeds():
    plan = validated_plan()

    assert build_command("--dry-run")[-3:] == ["--sweep", str(SWEEP_PATH), "--dry-run"]
    assert len(plan) == 6
    assert {run["seed"] for run in plan} == {43, 44}
    assert len({run["experiment_id"] for run in plan}) == 6
    assert all(run["resolved_config"]["dataset"] == "cifar10" for run in plan)
    assert all(run["resolved_config"]["evaluate_test"] is False for run in plan)
    assert all(run["resolved_config"]["epochs"] == 200 for run in plan)
    csgha_runs = [run for run in plan if run["resolved_config"]["model_type"] == "csgha"]
    assert len(csgha_runs) == 2
    assert all("csgha_v3" in run["experiment_id"] for run in csgha_runs)
