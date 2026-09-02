import numpy as np

from scripts.analysis.diagnose_early_exit_p0_class_risk import (
    CLASS_NAMES,
    _apply_guard,
    _metrics,
    _preaggregate_thresholds,
)


def test_predicted_class_guard_falls_back_only_for_protected_predictions():
    exit_logits = np.array([[8, 0], [0, 8], [8, 0]], dtype=float)
    final_logits = np.array([[0, 8], [8, 0], [0, 8]], dtype=float)

    predictions, paths = _apply_guard(exit_logits, final_logits, 0.5, (1,))

    np.testing.assert_array_equal(predictions, [0, 0, 0])
    np.testing.assert_array_equal(paths, [0, 1, 0])


def test_preaggregated_drop_and_route_count_match_direct_policy_metrics():
    labels = np.arange(len(CLASS_NAMES)).repeat(2)
    exit_logits = np.full((len(labels), len(CLASS_NAMES)), -4.0)
    final_logits = np.full_like(exit_logits, -4.0)
    for index, label in enumerate(labels):
        exit_logits[index, (label + index % 2) % len(CLASS_NAMES)] = 4.0
        final_logits[index, label] = 4.0
    value = {
        "labels": labels,
        "exit_logits": exit_logits,
        "final_logits": final_logits,
        "calibration": np.ones(len(labels), dtype=bool),
    }
    threshold = 0.5
    protected = (1, 3, 5)
    direct = _metrics(labels, exit_logits, final_logits, threshold, protected, [0.4, 1.0])
    summary = _preaggregate_thresholds(value, [threshold])[threshold]
    active = np.ones(len(CLASS_NAMES), dtype=bool)
    active[list(protected)] = False
    early_count = summary["route_counts_by_predicted_class"][active].sum()
    differences = summary["correctness_difference_by_true_and_predicted_class"][:, active].sum(axis=1)

    assert early_count / len(labels) == direct["route_fractions"][0]
    assert differences.sum() / len(labels) == direct["accuracy_drop"]
    assert (differences / summary["class_counts"]).max() == direct["worst_class_accuracy_drop"]
