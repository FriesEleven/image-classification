import numpy as np

from image_classification.selection.early_exit import select_shared_single_exit_policy
from scripts.launch_early_exit_p1 import (
    EXPERIMENT_TAG,
    JOBS,
    SEEDS,
    SPLIT_SEED,
    validated_plan,
)


def test_early_exit_p1_is_exact_disjoint_matched_serial_matrix():
    plan = validated_plan()

    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert {run["seed"] for run in plan} == set(SEEDS) == {54, 55, 56}
    assert {run["resolved_config"]["model_type"] for run in plan} == {
        "mobilenetv2", "multi_exit",
    }
    assert all(run["resolved_config"]["split_seed"] == SPLIT_SEED for run in plan)
    assert all(run["resolved_config"]["validation_size"] == 5000 for run in plan)
    assert all(run["resolved_config"]["calibration_size"] == 5000 for run in plan)
    assert all(run["resolved_config"]["evaluate_test"] is False for run in plan)
    assert all(run["resolved_config"]["cuda_graph"] is True for run in plan)
    assert all(f"_{EXPERIMENT_TAG}_seed" in run["experiment_id"] for run in plan)
    assert JOBS == 1


def test_early_exit_p1_retry_tag_produces_fresh_ids():
    original = {run["experiment_id"] for run in validated_plan()}
    retry = {run["experiment_id"] for run in validated_plan("p1b")}

    assert len(retry) == 6
    assert original.isdisjoint(retry)
    assert all("_p1b_seed" in experiment_id for experiment_id in retry)


def _logits(predictions):
    values = np.full((len(predictions), 2), -4.0)
    values[np.arange(len(predictions)), predictions] = 4.0
    return values


def test_shared_policy_requires_one_threshold_to_pass_every_seed():
    labels = np.array([0, 0, 1, 1])
    final_logits = _logits(labels)
    safe_exit = _logits(labels)
    unsafe_exit = safe_exit.copy()
    unsafe_exit[0] = [-4.0, 0.0]
    datasets = [
        (labels, safe_exit, final_logits),
        (labels, unsafe_exit, final_logits),
    ]

    selected = select_shared_single_exit_policy(
        datasets,
        path_costs=[0.4, 1.0],
        thresholds=[0.5, 0.99],
        min_early_fraction=0.5,
        max_early_fraction=1.0,
    )

    assert selected is not None
    assert selected["confidence_threshold"] == 0.99
    assert [value["route_fractions"][0] for value in selected["calibration_metrics"]] == [1.0, 0.75]
    assert all(value["worst_class_accuracy_drop"] <= 0 for value in selected["calibration_metrics"])


def test_shared_policy_returns_none_when_dynamic_route_cannot_be_safe():
    labels = np.array([0, 0, 1, 1])
    final_logits = _logits(labels)
    wrong_exit = _logits(1 - labels)

    selected = select_shared_single_exit_policy(
        [(labels, wrong_exit, final_logits)],
        path_costs=[0.4, 1.0],
        thresholds=[0.5, 1.1],
        min_early_fraction=0.25,
        max_early_fraction=0.95,
    )

    assert selected is None
