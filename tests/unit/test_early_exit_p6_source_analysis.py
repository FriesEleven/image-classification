import numpy as np

from scripts.analysis.analyze_early_exit_p6_source import route_metrics, select_shared_threshold


def record(seed, exit_correct=True):
    labels = np.array([0, 1, 0, 1])
    final = np.array([[5, 0], [0, 4], [3, 0], [0, 2]], dtype=float)
    exit_logits = final.copy()
    if not exit_correct:
        exit_logits[0] = [0, 5]
    return {
        "seed": seed, "labels": labels, "final_logits": final, "exit_logits": exit_logits,
        "num_classes": 2, "exit_cost_fraction": 0.4, "fallback_cost_fraction": 1.01,
    }


def test_route_metrics_includes_fallback_exit_head_overhead():
    metrics = route_metrics(record(1), np.array([True, False, False, False]))
    assert metrics["expected_cost_fraction"] == 0.25 * 0.4 + 0.75 * 1.01
    assert metrics["accuracy_drop"] == 0


def test_shared_selection_uses_all_seeds_and_frozen_tie_break():
    records = [record(seed) for seed in (71, 72, 73)]
    budget = {"overall_drop": 0, "balanced_drop": 0, "worst_class_drop": 0}
    selection = {
        "minimum_early_fraction_each_seed": 0.15,
        "maximum_early_fraction_each_seed": 0.95,
        "minimum_mac_saving_each_seed": 0.15,
    }
    selected, frontier = select_shared_threshold(records, budget, selection)
    assert len(frontier) == 1001
    assert selected is not None
    assert selected["threshold_candidates_evaluated"] == 1001
    assert all(metrics["accuracy_drop"] == 0 for metrics in selected["source_metrics"])
