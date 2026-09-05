import multiprocessing as mp

import numpy as np

from image_classification.selection.early_exit import policy_metrics
from scripts.analysis.run_early_exit_p5ab import (
    _initialize_worker,
    _shared_strategy_task,
    binary_auc,
    calibration_metrics,
    gate_a,
    macro_f1,
    reproduction_row_passes,
    risk_feasible,
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


def _summary(saving, overall=0.0, balanced=0.0, worst=0.0):
    return {
        "feasible": True,
        "cost_saving_fraction_mean": saving,
        "accuracy_drop_mean": overall,
        "accuracy_drop_max": overall,
        "balanced_accuracy_drop_max": balanced,
        "worst_class_accuracy_drop_max": worst,
    }


def test_risk_feasibility_uses_worst_seed_not_mean():
    budget = {"overall_drop": 0.0, "balanced_drop": 0.0, "worst_class_drop": 0.04}
    assert risk_feasible(_summary(0.2, overall=-0.01, worst=0.04), budget)
    assert not risk_feasible(_summary(0.3, overall=-0.01, worst=0.06), budget)


def test_gate_a_does_not_count_risk_violator_as_dominating():
    checks = {"all_passed": True, "p3_stop_without_test_preserved": True}
    protocol = {
        "risk_budgets": {
            "cifar10": {"overall_drop": 0.0, "balanced_drop": 0.0, "worst_class_drop": 0.0},
            "cifar100_relaxed_boundary": {"overall_drop": 0.0, "balanced_drop": 0.0, "worst_class_drop": 0.04},
        }
    }
    comparisons = {}
    for design in ("cifar10", "cifar100_relaxed"):
        proposed = _summary(0.20)
        violating = _summary(0.30, worst=0.06 if design == "cifar100_relaxed" else 0.01)
        entropy = _summary(0.19)
        comparisons[design] = {
            "shared_msp_full": {"source": proposed, "target": proposed},
            "shared_msp_overall": {"source": violating, "target": violating},
            "shared_entropy_full": {"source": entropy, "target": entropy},
        }
    matched = []
    for design in comparisons:
        for method, saving in (("shared_msp", 0.20), ("shared_entropy", 0.19)):
            summary = _summary(saving)
            matched.append({"design": design, "method": method, "target_saving": 0.2, "status": "feasible", **{f"target_{key}": value for key, value in summary.items()}})
    decision = gate_a(checks, comparisons, matched, protocol)
    assert decision["status"] == "go_p5c"
    assert decision["gates"]["proposed_not_pareto_dominated_on_primary_designs"]
