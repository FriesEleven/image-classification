from __future__ import annotations

import math

from scripts.analysis.build_early_exit_paper_tables import (
    PATH_MACS,
    build_main_results,
    build_risk_boundary,
    load_sources,
    percent,
    summarize,
)


def test_sample_standard_deviation_and_pp_conversion() -> None:
    result = summarize([0.01, 0.02, 0.03])
    assert math.isclose(result["mean"], 0.02)
    assert math.isclose(result["sample_std"], 0.01)
    assert math.isclose(percent(result["mean"]), 2.0)


def test_path_costs_match_frozen_ledger() -> None:
    assert PATH_MACS["cifar10"] == {"exit8": 2_676_864, "exit16": 5_234_688, "final": 6_124_928}
    assert PATH_MACS["cifar100"] == {"exit8": 2_682_624, "exit16": 5_249_088, "final": 6_240_128}


def test_main_table_uses_locked_test_aggregates() -> None:
    rows = build_main_results(load_sources())
    cifar10_policy = next(
        row for row in rows if row["dataset"] == "CIFAR-10" and row["method"] == "Locked shared policy"
    )
    cifar100_policy = next(
        row for row in rows if row["dataset"] == "CIFAR-100" and row["method"] == "Locked shared policy"
    )
    assert math.isclose(cifar10_policy["accuracy_mean_pct"], 87.03666666666666)
    assert math.isclose(cifar10_policy["mac_saving_mean_pct"], 36.509562879215335)
    assert math.isclose(cifar100_policy["accuracy_mean_pct"], 58.09666666666667)
    assert math.isclose(cifar100_policy["worst_class_drop_mean_pp"], 3.0)


def test_risk_boundary_keeps_post_hoc_and_confirmation_distinct() -> None:
    rows = build_risk_boundary(load_sources())
    assert any(row["risk_budget_pp"] == 0.0 and row["outcome"] == "Below 15% saving" for row in rows)
    p4 = next(row for row in rows if row["evidence_type"] == "Independent confirmation")
    assert p4["threshold"] == 0.903
    assert p4["risk_budget_pp"] == 4.0
