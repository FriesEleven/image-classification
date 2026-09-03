import numpy as np

from scripts.analysis.diagnose_early_exit_p3_class_guard import (
    _prepare,
    evaluate_guarded_threshold,
)


def test_predicted_class_guard_routes_only_allowed_confident_samples():
    labels = np.array([0, 0, 1, 2])
    final_logits = np.array(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
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

    result = evaluate_guarded_threshold(
        _prepare(labels, exit_logits, final_logits),
        threshold=0.5,
        allowed_predicted_classes=(0,),
        path_costs=[0.4, 1.0],
    )

    assert result["route_fractions"] == [0.5, 0.5]
    assert result["decision_changes"] == 2
    assert result["harmed"] == 1
    assert result["rescued"] == 1
    assert np.isclose(result["cost_saving_fraction"], 0.3)
