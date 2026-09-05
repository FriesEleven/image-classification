import numpy as np

from scripts.analysis.analyze_early_exit_p5c import mean_sd_ci


def test_mean_sd_ci_uses_sample_sd_and_small_sample_t_interval():
    result = mean_sd_ci([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["sample_sd"] == 1.0
    np.testing.assert_allclose(result["ci95_low"], 2.0 - 4.302652729911275 / np.sqrt(3))
    np.testing.assert_allclose(result["ci95_high"], 2.0 + 4.302652729911275 / np.sqrt(3))
