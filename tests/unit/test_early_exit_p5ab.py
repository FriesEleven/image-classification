import multiprocessing as mp

import numpy as np

from image_classification.selection.early_exit import policy_metrics
from scripts.analysis.run_early_exit_p5ab import (
    _initialize_worker,
    _shared_strategy_task,
    binary_auc,
    calibration_metrics,
    macro_f1,
    reproduction_row_passes,
    route_metrics,
    select_shared,
)


def _record(seed=1):
    labels = np.array([0, 0, 1, 1])
    final = np.array([[4, 0], [4, 0], [0, 4], [0, 4]], dtype=float)
    exit8 = np.array([[5, 0], [0, 1], [0, 5], [1, 0]], dtype=float)
    return {"seed": seed, "labels": labels, "final_logits": final, "exit8_logits": exit8, "exit_cost": 0.4}


def test_metrics_helpers_are_finite():
    record = _record()
    metrics = calibration_metrics(record["exit8_logits"], record["labels"])
    assert 0 <= metrics["ece_15"] <= 1
    assert macro_f1(record["labels"], record["final_logits"].argmax(1)) == 1
    assert binary_auc(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])) == 1


def test_route_metrics_counts_harm_and_rescue():
    values = route_metrics(_record(), np.array([True, True, True, True]))
    assert values["premature_count"] == 2
    assert values["rescue_count"] == 0
    assert values["cost_saving_fraction"] == 0.6


def test_vectorized_route_metrics_match_reference_metrics():
    record = _record()
    early = np.array([True, False, True, False])
    predictions = np.where(
        early,
        record["exit8_logits"].argmax(axis=1),
        record["final_logits"].argmax(axis=1),
    )
    reference = policy_metrics(
        record["labels"],
        predictions,
        record["final_logits"].argmax(axis=1),
        np.where(early, 0, 1),
        [record["exit_cost"], 1.0],
    )
    actual = route_metrics(record, early)
    for key in reference:
        np.testing.assert_allclose(actual[key], reference[key], atol=1e-12)


def test_shared_selector_keeps_final_only_when_zero_risk_requires_it():
    selected = select_shared([_record()], "msp", {"overall_drop": 0.0, "balanced_drop": 0.0, "worst_class_drop": 0.0})
    assert selected is not None
    assert selected["source_metrics"][0]["accuracy_drop"] <= 0


def test_p1_replay_tolerance_never_relaxes_risk_metrics():
    amendment = {
        "unchanged_requirements": {
            "accuracy_drop_absolute_difference_max": 1e-12,
            "balanced_accuracy_drop_absolute_difference_max": 1e-12,
            "worst_class_accuracy_drop_absolute_difference_max": 1e-12,
            "all_non_p1_cohorts_absolute_difference_max": 1e-12,
        },
        "p1_cross_hardware_replay_disclosure": {
            "absolute_accuracy_difference_max": 0.000600000001,
            "cost_saving_fraction_difference_max": 0.000230000001,
        },
    }
    acceptable = {"accuracy": 0.0006, "accuracy_drop": 0.0, "balanced_accuracy_drop": 0.0, "worst_class_accuracy_drop": 0.0, "cost_saving_fraction": 0.000225}
    assert reproduction_row_passes("cifar10_source", acceptable, amendment)
    unacceptable = {**acceptable, "worst_class_accuracy_drop": 0.02}
    assert not reproduction_row_passes("cifar10_source", unacceptable, amendment)


def test_shared_strategy_runs_in_forked_worker_pool():
    cohorts = {"source": [_record(1)], "target": [_record(2)]}
    payload = ("design", "source", "target", "shared_msp_full", "msp", {"overall_drop": 0.0, "balanced_drop": 0.0, "worst_class_drop": 0.0}, 1.0)
    with mp.get_context("fork").Pool(2, initializer=_initialize_worker, initargs=(cohorts,)) as pool:
        design, method, result = pool.map(_shared_strategy_task, [payload])[0]
    assert design == "design"
    assert method == "shared_msp_full"
    assert result["source"]["feasible"]
