import numpy as np

from scripts.analysis.analyze_early_exit_p0 import _cost_profile
from scripts.analysis.analyze_early_exit_p3 import frozen_gates
from scripts.analysis.evaluate_early_exit_p3_locked_test import locked_test_gates
from scripts.launch_early_exit_p3 import (
    EXPERIMENT_TAG,
    JOBS,
    SEEDS,
    SOURCE_SEEDS,
    SPLIT_SEED,
    TARGET_SEEDS,
    validated_plan,
)


def _policy(**overrides):
    return {
        "accuracy_drop": 0.0,
        "balanced_accuracy_drop": 0.0,
        "worst_class_accuracy_drop": 0.0,
        "route_fractions": [0.5, 0.5],
        "cost_saving_fraction": 0.28,
        **overrides,
    }


def test_early_exit_p3_is_exact_cifar100_source_target_serial_matrix():
    plan = validated_plan()

    assert len(plan) == 12
    assert len({run["experiment_id"] for run in plan}) == 12
    assert set(SOURCE_SEEDS).isdisjoint(TARGET_SEEDS)
    assert {run["seed"] for run in plan} == set(SEEDS) == set(range(60, 66))
    assert {run["resolved_config"]["model_type"] for run in plan} == {
        "mobilenetv2",
        "multi_exit",
    }
    assert all(run["resolved_config"]["dataset"] == "cifar100" for run in plan)
    assert all(run["resolved_config"]["split_seed"] == SPLIT_SEED for run in plan)
    assert all(run["resolved_config"]["calibration_size"] == 5000 for run in plan)
    assert all(run["resolved_config"]["evaluate_test"] is False for run in plan)
    assert all(run["resolved_config"]["cuda_graph"] is True for run in plan)
    assert all(f"_{EXPERIMENT_TAG}_seed" in run["experiment_id"] for run in plan)
    assert JOBS == 1


def test_early_exit_p3_retry_tag_produces_fresh_ids():
    original = {run["experiment_id"] for run in validated_plan()}
    retry = {run["experiment_id"] for run in validated_plan("p3b")}

    assert len(retry) == 12
    assert original.isdisjoint(retry)
    assert all("_p3b_seed" in experiment_id for experiment_id in retry)


def test_p3_frozen_gates_require_every_unseen_target_to_pass():
    gains = [0.0] * 6
    source = [_policy() for _ in SOURCE_SEEDS]
    target = [_policy() for _ in TARGET_SEEDS]
    target[-1] = _policy(worst_class_accuracy_drop=0.02)

    gates = frozen_gates(gains, source, target)

    assert gates["each_target_worst_class_drop_at_most_0"] is False
    assert all(passed for name, passed in gates.items() if name != "each_target_worst_class_drop_at_most_0")


def test_p3_frozen_gates_accept_complete_safe_replication():
    gates = frozen_gates(
        [0.001, -0.001, 0.0, 0.002, -0.002, 0.001],
        [_policy() for _ in SOURCE_SEEDS],
        [_policy() for _ in TARGET_SEEDS],
    )

    assert all(gates.values())


def test_cifar100_cost_profile_counts_larger_classification_heads():
    cifar10 = _cost_profile("cifar10")
    cifar100 = _cost_profile("cifar100")

    assert cifar100["num_classes"] == 100
    assert cifar100["path_macs"][0] > cifar10["path_macs"][0]
    assert cifar100["path_macs"][-1] > cifar10["path_macs"][-1]
    assert np.isclose(cifar100["path_cost_fractions"][-1], 1.0)


def _test_row(seed, **overrides):
    policy = {
        "accuracy_drop": 0.0,
        "balanced_accuracy_drop": 0.0,
        "worst_class_accuracy_drop": 0.0,
        "route_fractions": [0.5, 0.5],
        "cost_saving_fraction": 0.28,
        **overrides,
    }
    return {"seed": seed, "locked_policy": policy}


def test_p3_locked_test_gates_accept_preregistered_boundary():
    assert all(locked_test_gates([_test_row(seed) for seed in SEEDS]).values())


def test_p3_locked_test_gates_reject_excess_mean_drop():
    rows = [_test_row(seed, accuracy_drop=0.003) for seed in SEEDS]

    gates = locked_test_gates(rows)

    assert gates["each_accuracy_drop_at_most_0_005"] is True
    assert gates["mean_accuracy_drop_at_most_0_002"] is False
