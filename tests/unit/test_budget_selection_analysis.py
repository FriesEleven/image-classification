import pytest

from scripts.analysis.select_budget_stage_attention import (
    _confirmation_plan,
    _mean,
    _sample_std,
)


def test_selection_statistics_do_not_depend_on_shadowed_stdlib_module():
    values = [-0.12, -0.14, -0.20]

    assert _mean(values) == pytest.approx(-0.15333333333333335)
    assert _sample_std(values) == pytest.approx(0.04163331998932266)


def test_selection_statistics_reject_insufficient_values():
    with pytest.raises(ValueError, match="empty"):
        _mean([])
    with pytest.raises(ValueError, match="at least two"):
        _sample_std([1.0])


def test_all_none_selection_activates_stop_rule():
    baseline = "shallow_none__middle_none__deep_none"

    plan = _confirmation_plan([baseline], baseline)

    assert plan["required"] is False
    assert plan["selected_candidate_ids"] == []
    assert plan["recommended_confirmation_seeds"] == []
    assert plan["retrain_from_scratch"] is False
    assert "stop rule" in plan["stop_reason"]


def test_attention_selection_keeps_only_nonbaseline_candidates():
    baseline = "shallow_none__middle_none__deep_none"
    selected = "shallow_se__middle_none__deep_none"

    plan = _confirmation_plan([baseline, selected], baseline)

    assert plan["required"] is True
    assert plan["selected_candidate_ids"] == [selected]
    assert plan["recommended_confirmation_seeds"] == [48, 49, 50]
    assert plan["retrain_from_scratch"] is True
