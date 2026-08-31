import pytest

from scripts.analysis.audit_csgha_v5_serial import sample_stats


def test_v5_serial_audit_uses_sample_standard_deviation():
    result = sample_stats([87.62, 87.18, 88.14])
    assert result["mean_percent"] == pytest.approx(87.64666666666668)
    assert result["sample_sd_percent"] == pytest.approx(0.4805552344250662)


def test_v5_serial_paired_values_match_manifest_results():
    control = {42: 87.88, 43: 87.82, 44: 88.12}
    guided = {42: 87.62, 43: 87.18, 44: 88.14}
    deltas = [guided[seed] - control[seed] for seed in (42, 43, 44)]
    result = sample_stats(deltas)
    assert deltas == pytest.approx([-0.26, -0.64, 0.02])
    assert result["mean_percent"] == pytest.approx(-0.29333333333332706)
    assert result["sample_sd_percent"] == pytest.approx(0.3312602199681206)
