import numpy as np
import pytest

from image_classification.selection.early_exit import (
    apply_policy,
    policy_metrics,
    select_policy,
    stratified_calibration_mask,
)


def test_stratified_calibration_mask_is_balanced_and_deterministic():
    labels = np.repeat(np.arange(3), 10)

    first = stratified_calibration_mask(labels, seed=17)
    second = stratified_calibration_mask(labels, seed=17)

    np.testing.assert_array_equal(first, second)
    assert [int(first[labels == class_id].sum()) for class_id in range(3)] == [5, 5, 5]


def test_policy_uses_first_confident_exit_then_second_then_final():
    final = np.array([[0, 4], [4, 0], [0, 4]], dtype=float)
    first = np.array([[0, 8], [0, 0], [0, 0]], dtype=float)
    second = np.array([[8, 0], [8, 0], [0, 0]], dtype=float)

    predictions, paths = apply_policy([first, second], final, [0.9, 0.9])

    np.testing.assert_array_equal(predictions, [1, 0, 1])
    np.testing.assert_array_equal(paths, [0, 1, 2])


def test_policy_selection_keeps_final_only_as_feasible_fallback():
    labels = np.tile([0, 1], 20)
    final = np.where(labels[:, None] == np.array([0, 1]), 4.0, 0.0)
    wrong = final[:, ::-1]

    selected = select_policy(
        labels,
        [wrong, wrong],
        final,
        [0.3, 0.7, 1.0],
        max_accuracy_drop=0,
        max_balanced_accuracy_drop=0,
        max_worst_class_drop=0,
        grid_points=5,
    )

    assert selected["calibration_metrics"]["route_fractions"] == [0.0, 0.0, 1.0]
    assert selected["calibration_metrics"]["cost_saving_fraction"] == 0


def test_policy_metrics_reject_missing_costs():
    values = np.array([0, 1])
    with pytest.raises(ValueError, match="path_costs"):
        policy_metrics(values, values, values, np.array([0, 2]), [0.2, 0.5])
