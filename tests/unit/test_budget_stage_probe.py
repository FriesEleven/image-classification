from scripts.launch_budget_stage_probe import EXPERIMENT_TAG, JOBS, SEEDS, validated_plan


def test_budget_probe_plan_is_exact_paired_serial_validation_matrix():
    plan = validated_plan()

    assert len(plan) == 30
    assert len({run["experiment_id"] for run in plan}) == 30
    assert {run["seed"] for run in plan} == set(SEEDS) == {45, 46, 47}
    assert all(run["resolved_config"]["model_type"] == "stage_sparse" for run in plan)
    assert all(run["resolved_config"]["evaluate_test"] is False for run in plan)
    assert all(run["resolved_config"]["cuda_graph"] is True for run in plan)
    assert all(run["resolved_config"]["measure_inference"] is False for run in plan)
    assert all(f"_{EXPERIMENT_TAG}_seed" in run["experiment_id"] for run in plan)
    assert JOBS == 1


def test_budget_probe_retry_tag_produces_fresh_ids():
    original = {run["experiment_id"] for run in validated_plan()}
    retry = {run["experiment_id"] for run in validated_plan("probe2")}

    assert len(retry) == 30
    assert original.isdisjoint(retry)
    assert all("_probe2_seed" in experiment_id for experiment_id in retry)
