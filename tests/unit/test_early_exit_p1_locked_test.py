import numpy as np

from scripts.analysis.evaluate_early_exit_p1_locked_test import summarize_seed


def _logits(predictions, confidences):
    values = np.zeros((len(predictions), 2), dtype=np.float64)
    for index, (prediction, confidence) in enumerate(zip(predictions, confidences)):
        other = 1 - prediction
        values[index, prediction] = np.log(confidence)
        values[index, other] = np.log(1 - confidence)
    return values


def test_locked_test_summary_applies_only_frozen_exit8_threshold():
    labels = np.array([0, 0, 1, 1])
    baseline = _logits([0, 1, 1, 1], [0.9] * 4)
    final = _logits([0, 0, 1, 1], [0.9] * 4)
    exit8 = _logits([0, 1, 1, 0], [0.99, 0.7, 0.99, 0.7])
    exit16 = _logits([1, 1, 0, 0], [0.99] * 4)

    result, predictions, paths = summarize_seed(
        labels,
        baseline,
        [final, exit8, exit16],
        seed=54,
        threshold=0.98,
        path_costs=[0.4, 1.0],
        class_names=("zero", "one"),
    )

    np.testing.assert_array_equal(predictions, labels)
    np.testing.assert_array_equal(paths, [0, 1, 0, 1])
    assert result["locked_policy"]["accuracy"] == 1.0
    assert result["locked_policy"]["route_fractions"] == [0.5, 0.5]
    assert np.isclose(result["locked_policy"]["cost_saving_fraction"], 0.3)
    assert result["exit16_test_accuracy"] == 0.0
    assert result["decision_changes_vs_final"]["changed"] == 0
