import numpy as np

from scripts.analysis.diagnose_early_exit_p3_boundary import evaluate_threshold


def test_boundary_metrics_count_harmed_rescued_and_neutral_changes():
    labels = np.array([0, 0, 1, 2])
    final_logits = np.array(
        [
            [5.0, 0.0, 0.0],  # correct, then harmed
            [0.0, 5.0, 0.0],  # wrong, then rescued
            [0.0, 5.0, 0.0],  # correct, then harmed
            [5.0, 0.0, 0.0],  # wrong, changed but still wrong
        ]
    )
    exit_logits = np.array(
        [
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    result = evaluate_threshold(
        labels,
        exit_logits,
        final_logits,
        threshold=0.5,
        path_costs=[0.4, 1.0],
    )

    assert result["decision_changes"] == 4
    assert result["harmed"] == 2
    assert result["rescued"] == 1
    assert result["neutral_changes"] == 1
    assert result["route_fractions"] == [1.0, 0.0]
    assert np.isclose(result["cost_saving_fraction"], 0.6)


def test_boundary_metrics_fallback_only_matches_reference():
    labels = np.array([0, 1])
    final_logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    exit_logits = np.array([[0.0, 2.0], [2.0, 0.0]])

    result = evaluate_threshold(
        labels,
        exit_logits,
        final_logits,
        threshold=1.1,
        path_costs=[0.4, 1.0],
    )

    assert result["accuracy_drop"] == 0.0
    assert result["worst_class_accuracy_drop"] == 0.0
    assert result["decision_changes"] == 0
    assert result["route_fractions"] == [0.0, 1.0]
