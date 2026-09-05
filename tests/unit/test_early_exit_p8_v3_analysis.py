import pytest

from scripts.analysis.analyze_early_exit_p8_v3 import aggregate, summarize


def test_seed_summary_uses_sample_sd_and_t_interval():
    result = summarize([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["sample_sd"] == 1.0
    assert result["ci95_low"] == pytest.approx(-0.4841377117)
    assert result["ci95_high"] == pytest.approx(4.4841377117)


def test_aggregate_averages_rounds_before_three_seed_summary():
    rows = []
    for seed in (1, 2, 3):
        for round_index in range(5):
            for order, (mode, latency) in enumerate((
                ("final_only", 10.0),
                ("actual_dynamic", 10.0 - seed),
                ("exit8_only", 4.0),
                ("forced_fallback", 11.0),
            )):
                rows.append({
                    "device": "cpu", "dataset": "cifar10", "seed": str(seed),
                    "batch_size": "1", "round": str(round_index), "mode": mode,
                    "mode_order": str(order), "latency_mean_ms": str(latency),
                    "actual_dynamic_saving_fraction": str(seed / 10),
                    "timed_early_fraction": "0.5", "mac_saving_fraction": "0.25",
                    "singleton_isolated_expected_ms": "7.5",
                })
    seed_rows, across = aggregate(rows)
    assert len(seed_rows) == 3 and len(across) == 1
    assert across[0]["actual_saving_percent_mean"] == pytest.approx(20.0)
    assert across[0]["actual_saving_percent_sample_sd"] == pytest.approx(10.0)
    assert across[0]["positive_speedup_seeds"] == 3
    assert across[0]["forced_fallback_overhead_percent_mean"] == pytest.approx(10.0)
    assert across[0]["isolated_expected_saving_percent_mean"] == pytest.approx(25.0)
