import pytest

from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from image_classification.selection import (
    attention_operation_profile,
    candidate_from_positions,
    enumerate_stage_candidates,
    score_candidates,
    select_candidates_for_budgets,
)


def test_stage_candidate_space_is_complete_and_unique():
    candidates = enumerate_stage_candidates()

    assert len(candidates) == 64
    assert len({candidate["candidate_id"] for candidate in candidates}) == 64
    assert sum(candidate["active_stages"] <= 1 for candidate in candidates) == 10


def test_candidate_positions_require_complete_stage_packets():
    candidate = candidate_from_positions((1, 2), (7, 8), (15, 16))
    assert candidate["choices"] == {"shallow": "eca", "middle": "se", "deep": "cbam"}
    with pytest.raises(ValueError, match="complete position packet"):
        candidate_from_positions((1,), (), ())


def test_attention_operation_proxy_is_stage_sensitive():
    shallow = build_model(ExperimentConfig(model_type="stage_sparse", eca_positions=(1, 2)))
    deep = build_model(ExperimentConfig(model_type="stage_sparse", eca_positions=(15, 16)))

    shallow_profile = attention_operation_profile(shallow)
    deep_profile = attention_operation_profile(deep)

    assert shallow_profile["operations_estimate"] > deep_profile["operations_estimate"]
    assert sum(row["parameters"] for row in shallow_profile["modules"]) == 6
    assert sum(row["parameters"] for row in deep_profile["modules"]) == 6


def test_selector_can_return_no_attention_when_all_robust_gains_are_negative():
    profiles = []
    for candidate in enumerate_stage_candidates():
        profiles.append(
            {
                **candidate,
                "parameter_delta": candidate["active_stages"] * 100,
                "attention_operations_estimate": candidate["active_stages"] * 1000,
                "latency_overhead_percent": candidate["active_stages"] * 2.0,
            }
        )
    evidence = {
        f"{stage}_{attention}": {"mean_gain_pp": -0.1, "sample_std_pp": 0.05}
        for stage in ("shallow", "middle", "deep")
        for attention in ("eca", "se", "cbam")
    }
    scored = score_candidates(profiles, evidence, risk_penalty=0.5)
    selections = select_candidates_for_budgets(
        scored,
        [{"name": "test", "max_active_stages": 3}],
    )

    assert selections[0]["selected"]["candidate_id"] == (
        "shallow_none__middle_none__deep_none"
    )


def test_selector_prefers_no_attention_when_utility_is_exactly_tied():
    profiles = []
    for candidate in enumerate_stage_candidates():
        profiles.append(
            {
                **candidate,
                "parameter_delta": candidate["active_stages"] * 100,
                "attention_operations_estimate": candidate["active_stages"] * 1000,
                "latency_overhead_percent": -candidate["active_stages"] * 2.0,
            }
        )
    evidence = {
        f"{stage}_{attention}": {"mean_gain_pp": 0.0, "sample_std_pp": 0.0}
        for stage in ("shallow", "middle", "deep")
        for attention in ("eca", "se", "cbam")
    }
    scored = score_candidates(profiles, evidence, risk_penalty=0.5)
    selections = select_candidates_for_budgets(
        scored,
        [{"name": "test", "max_active_stages": 3}],
    )

    assert selections[0]["selected"]["candidate_id"] == (
        "shallow_none__middle_none__deep_none"
    )


def test_selector_obeys_budget_before_maximizing_utility():
    profiles = []
    for candidate in enumerate_stage_candidates():
        profiles.append(
            {
                **candidate,
                "parameter_delta": candidate["active_stages"] * 100,
                "attention_operations_estimate": candidate["active_stages"] * 1000,
                "latency_overhead_percent": candidate["active_stages"] * 2.0,
            }
        )
    evidence = {
        f"{stage}_{attention}": {
            "mean_gain_pp": 0.8 if (stage, attention) == ("middle", "eca") else -0.2,
            "sample_std_pp": 0.0,
        }
        for stage in ("shallow", "middle", "deep")
        for attention in ("eca", "se", "cbam")
    }
    scored = score_candidates(profiles, evidence, risk_penalty=0.5)
    selections = select_candidates_for_budgets(
        scored,
        [{"name": "one_stage", "max_active_stages": 1, "max_parameter_delta": 100}],
    )

    assert selections[0]["selected"]["candidate_id"] == (
        "shallow_none__middle_eca__deep_none"
    )
