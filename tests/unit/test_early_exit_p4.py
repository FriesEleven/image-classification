from scripts.analysis.analyze_early_exit_p4_confirmation import frozen_gates
from scripts.analysis.audit_early_exit_p4 import _serial_timeline_overlaps
from scripts.analysis.evaluate_early_exit_p4_locked_test import locked_test_gates
from scripts.launch_early_exit_p4 import (
    LOCKED_THRESHOLD,
    SEEDS,
    validate_policy_lock,
    validated_plan,
)


def _policy(**overrides):
    value = {
        "accuracy_drop": 0.0,
        "balanced_accuracy_drop": 0.0,
        "worst_class_accuracy_drop": 0.04,
        "route_fractions": [0.4, 0.6],
        "cost_saving_fraction": 0.22,
    }
    value.update(overrides)
    return value


def test_p4_policy_lock_and_plan_are_frozen():
    lock = validate_policy_lock()
    plan = validated_plan()

    assert lock["frozen_policy"]["confidence_threshold"] == LOCKED_THRESHOLD
    assert lock["frozen_policy"]["p4_threshold_candidates"] == 0
    assert len(plan) == 6
    assert {(run["resolved_config"]["model_type"], run["seed"]) for run in plan} == {
        (model_type, seed) for model_type in ("mobilenetv2", "multi_exit") for seed in SEEDS
    }
    assert all("_p4a_seed" in run["experiment_id"] for run in plan)
    assert all(run["resolved_config"]["evaluate_test"] is False for run in plan)


def test_p4_audit_timeline_allows_launch_gap_and_rejects_overlap():
    previous_finish = "2026-09-03T15:55:24+08:00"

    assert not _serial_timeline_overlaps(previous_finish, "2026-09-03T15:55:24+08:00")
    assert not _serial_timeline_overlaps(previous_finish, "2026-09-03T15:55:25+08:00")
    assert _serial_timeline_overlaps(previous_finish, "2026-09-03T15:55:23+08:00")


def test_p4_confirmation_gates_accept_frozen_boundary():
    gates = frozen_gates([0.0, 0.01, -0.005], [_policy(), _policy(), _policy()])

    assert all(gates.values())


def test_p4_confirmation_gates_reject_extra_class_drop():
    policies = [_policy(), _policy(), _policy(worst_class_accuracy_drop=0.06)]
    gates = frozen_gates([0.0, 0.01, -0.005], policies)

    assert gates["each_worst_class_drop_at_most_0_04"] is False
    assert all(passed for name, passed in gates.items() if name != "each_worst_class_drop_at_most_0_04")


def test_p4_official_test_gates_are_stricter_for_overall_drop():
    rows = [{"seed": seed, "locked_policy": _policy(accuracy_drop=0.005)} for seed in SEEDS]
    gates = locked_test_gates(rows)

    assert gates["each_accuracy_drop_at_most_0_005"] is True
    assert gates["mean_accuracy_drop_at_most_0_002"] is False
