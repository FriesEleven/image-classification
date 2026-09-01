import pytest

from scripts.analysis.audit_csgha_v6_serial import sample_stats


def test_v6_serial_audit_uses_sample_standard_deviation():
    result = sample_stats([87.76, 87.68, 87.92])
    assert result["mean_percent"] == pytest.approx(87.78666666666668)
    assert result["sample_sd_percent"] == pytest.approx(0.12220201853215312)


def test_v6_serial_paired_values_match_manifest_results():
    control = {42: 87.88, 43: 87.82, 44: 88.12}
    guided = {42: 87.76, 43: 87.68, 44: 87.92}
    deltas = [guided[seed] - control[seed] for seed in (42, 43, 44)]
    result = sample_stats(deltas)
    assert deltas == pytest.approx([-0.12, -0.14, -0.20])
    assert result["mean_percent"] == pytest.approx(-0.15333333333332652)
    assert result["sample_sd_percent"] == pytest.approx(0.0416333199893303)
