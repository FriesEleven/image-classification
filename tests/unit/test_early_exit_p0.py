from scripts.launch_early_exit_p0 import EXPERIMENT_TAG, JOBS, SEEDS, validated_plan


def test_early_exit_p0_is_exact_paired_serial_validation_matrix():
    plan = validated_plan()

    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert {run["seed"] for run in plan} == set(SEEDS) == {51, 52, 53}
    assert {run["resolved_config"]["model_type"] for run in plan} == {
        "mobilenetv2", "multi_exit",
    }
    assert all(run["resolved_config"]["evaluate_test"] is False for run in plan)
    assert all(run["resolved_config"]["cuda_graph"] is True for run in plan)
    assert all(run["resolved_config"]["measure_inference"] is False for run in plan)
    assert all(f"_{EXPERIMENT_TAG}_seed" in run["experiment_id"] for run in plan)
    assert JOBS == 1


def test_early_exit_p0_retry_tag_produces_fresh_ids():
    original = {run["experiment_id"] for run in validated_plan()}
    retry = {run["experiment_id"] for run in validated_plan("p0b")}

    assert len(retry) == 6
    assert original.isdisjoint(retry)
    assert all("_p0b_seed" in experiment_id for experiment_id in retry)
