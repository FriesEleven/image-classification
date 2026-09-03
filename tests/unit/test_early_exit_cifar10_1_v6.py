import pytest

from scripts.analysis.evaluate_early_exit_cifar10_1_v6 import ALL_SEEDS, frozen_external_gates


def _row(seed, **policy_overrides):
    policy = {
        "accuracy_drop": 0.0,
        "balanced_accuracy_drop": 0.0,
        "worst_class_accuracy_drop": 0.0,
        "route_fractions": [0.65, 0.35],
        "cost_saving_fraction": 0.36,
        **policy_overrides,
    }
    return {"seed": seed, "locked_policy": policy}


def test_external_gates_require_every_source_and_target_model_to_pass():
    rows = [_row(seed) for seed in ALL_SEEDS]
    rows[-1] = _row(ALL_SEEDS[-1], worst_class_accuracy_drop=0.005)

    gates = frozen_external_gates(rows)

    assert gates["each_external_worst_class_drop_at_most_0"] is False
    assert all(passed for name, passed in gates.items() if name != "each_external_worst_class_drop_at_most_0")


def test_external_gates_accept_safe_dynamic_policy_for_all_six_models():
    assert all(frozen_external_gates([_row(seed) for seed in ALL_SEEDS]).values())


def test_external_gates_reject_wrong_seed_order():
    with pytest.raises(ValueError, match="ordered source and target seeds"):
        frozen_external_gates([_row(seed) for seed in reversed(ALL_SEEDS)])
