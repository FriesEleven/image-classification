import numpy as np

from scripts.analysis.analyze_early_exit_p2_transfer import evaluate_frozen_policy, frozen_gates
from scripts.launch_early_exit_p2 import (
    EXPERIMENT_TAG,
    JOBS,
    LOCKED_THRESHOLD,
    SEEDS,
    SOURCE_SEEDS,
    SPLIT_SEED,
    validate_source_policy,
    validated_plan,
)


def _logits(predictions, confidences):
    values = np.zeros((len(predictions), 2), dtype=np.float64)
    for index, (prediction, confidence) in enumerate(zip(predictions, confidences)):
        values[index, prediction] = np.log(confidence)
        values[index, 1 - prediction] = np.log(1 - confidence)
    return values


def test_early_exit_p2_is_exact_unseen_seed_serial_matrix():
    plan = validated_plan()

    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert set(SOURCE_SEEDS).isdisjoint(SEEDS)
    assert {run["seed"] for run in plan} == set(SEEDS) == {57, 58, 59}
    assert {run["resolved_config"]["model_type"] for run in plan} == {
        "mobilenetv2",
        "multi_exit",
    }
    assert all(run["resolved_config"]["split_seed"] == SPLIT_SEED for run in plan)
    assert all(run["resolved_config"]["validation_size"] == 5000 for run in plan)
    assert all(run["resolved_config"]["calibration_size"] == 5000 for run in plan)
    assert all(run["resolved_config"]["evaluate_test"] is False for run in plan)
    assert all(run["resolved_config"]["cuda_graph"] is True for run in plan)
    assert all(f"_{EXPERIMENT_TAG}_seed" in run["experiment_id"] for run in plan)
    assert JOBS == 1


def test_early_exit_p2_source_policy_is_hash_locked():
    selection = validate_source_policy()

    assert selection["locked_policy"]["confidence_threshold"] == LOCKED_THRESHOLD == 0.984
    assert selection["locked_policy"]["shared_across_training_seeds"] == [54, 55, 56]


def test_p2_transfer_applies_supplied_threshold_without_fitting():
    labels = np.array([0, 0, 1, 1])
    final = _logits(labels, [0.9] * 4)
    exit8 = _logits([0, 1, 1, 0], [0.99, 0.7, 0.99, 0.7])

    metrics, predictions, paths = evaluate_frozen_policy(
        labels,
        exit8,
        final,
        threshold=0.984,
        path_costs=[0.4, 1.0],
    )

    np.testing.assert_array_equal(predictions, labels)
    np.testing.assert_array_equal(paths, [0, 1, 0, 1])
    assert metrics["accuracy_drop"] == 0.0
    assert metrics["worst_class_accuracy_drop"] == 0.0
    assert metrics["route_fractions"] == [0.5, 0.5]
    assert np.isclose(metrics["cost_saving_fraction"], 0.3)
    assert metrics["decision_changes_vs_final"] == 0
    assert metrics["harmed_vs_final"] == 0


def test_p2_frozen_gates_reject_any_target_worst_class_regression():
    safe = {
        "accuracy_drop": 0.0,
        "balanced_accuracy_drop": 0.0,
        "worst_class_accuracy_drop": 0.0,
        "route_fractions": [0.65, 0.35],
        "cost_saving_fraction": 0.36,
    }
    regressed = {**safe, "worst_class_accuracy_drop": 0.002}

    gates = frozen_gates([0.001, 0.002, 0.0], [safe, safe, regressed])

    assert gates["each_transfer_worst_class_drop_at_most_0"] is False
    assert all(passed for name, passed in gates.items() if name != "each_transfer_worst_class_drop_at_most_0")


def test_early_exit_p2_retry_tag_produces_fresh_ids():
    original = {run["experiment_id"] for run in validated_plan()}
    retry = {run["experiment_id"] for run in validated_plan("p2b")}

    assert len(retry) == 6
    assert original.isdisjoint(retry)
    assert all("_p2b_seed" in experiment_id for experiment_id in retry)
