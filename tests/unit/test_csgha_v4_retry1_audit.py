import math

import pytest

from scripts.analysis.audit_csgha_v4_retry1 import sample_stats


def test_retry1_audit_uses_sample_standard_deviation():
    result = sample_stats([87.88, 87.82, 88.12])
    assert result["n"] == 3
    assert result["mean_percent"] == pytest.approx(87.94)
    assert result["sample_sd_percent"] == pytest.approx(0.15874507866388146)


def test_retry1_paired_values_match_manifest_results():
    control = {42: 87.88, 43: 87.82, 44: 88.12}
    guided = {42: 88.44, 43: 87.90, 44: 87.90}
    deltas = [guided[seed] - control[seed] for seed in (42, 43, 44)]
    result = sample_stats(deltas)
    assert deltas == pytest.approx([0.56, 0.08, -0.22])
    assert result["mean_percent"] == pytest.approx(0.14)
    assert result["sample_sd_percent"] == pytest.approx(0.39344631145811976)


@pytest.mark.parametrize("values", [[], [math.nan], [math.inf]])
def test_retry1_audit_rejects_invalid_statistics(values):
    with pytest.raises(ValueError, match="finite"):
        sample_stats(values)
