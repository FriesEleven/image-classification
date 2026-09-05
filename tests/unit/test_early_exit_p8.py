import numpy as np

from scripts.analysis.benchmark_early_exit_p8 import expected_latency, latency_statistics


def test_latency_statistics_and_expected_path_weighting():
    result = latency_statistics(np.array([1.0, 2.0, 3.0]), batch_size=4)
    assert result["latency_mean_ms"] == 2.0
    assert result["latency_median_ms"] == 2.0
    assert result["throughput_samples_per_second"] == 2000.0
    assert expected_latency(0.25, 1.0, 3.0) == 2.5
