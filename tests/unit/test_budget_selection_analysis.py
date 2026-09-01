import pytest

from scripts.analysis.select_budget_stage_attention import _mean, _sample_std


def test_selection_statistics_do_not_depend_on_shadowed_stdlib_module():
    values = [-0.12, -0.14, -0.20]

    assert _mean(values) == pytest.approx(-0.15333333333333335)
    assert _sample_std(values) == pytest.approx(0.04163331998932266)


def test_selection_statistics_reject_insufficient_values():
    with pytest.raises(ValueError, match="empty"):
        _mean([])
    with pytest.raises(ValueError, match="at least two"):
        _sample_std([1.0])
