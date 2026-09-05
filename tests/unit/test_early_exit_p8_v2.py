import numpy as np
import pytest
import torch

from scripts.analysis.benchmark_early_exit_p8_v2 import (
    check_routing,
    mac_saving,
    sample_order,
    time_mode,
    timed_sample_ids,
)


def test_workload_indices_represent_the_exact_repeated_batches():
    ids = np.arange(20, 30)
    np.testing.assert_array_equal(timed_sample_ids(ids, 2, 7),
                                  [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [20, 21], [22, 23]])
    a = sample_order(5000, 32, 0, 54)
    assert len(a) == 4992 and len(np.unique(a)) == 4992
    np.testing.assert_array_equal(a, sample_order(5000, 32, 0, 54))
    assert not np.array_equal(a, sample_order(5000, 32, 1, 54))


def test_mac_accounting_includes_unresolved_exit_head_overhead():
    assert mac_saving(1.0, 40, 100, 5) == pytest.approx(0.6)
    assert mac_saving(0.0, 40, 100, 5) == pytest.approx(-0.05)
    assert mac_saving(0.5, 40, 100, 5) == pytest.approx(0.275)


class TinyRouter:
    def forward_to_exit(self, inputs, position):
        return inputs if position == 8 else inputs.flip(1)

    def forward_with_policy(self, inputs, threshold, exit_position=8):
        early = inputs.float().softmax(1).amax(1) >= threshold
        return torch.where(early[:, None], inputs, inputs.flip(1)), (~early).long()


def test_timed_paths_and_correctness_all_mixed_none():
    batches = [torch.tensor([[8., 0.], [0., 0.1]])]
    model = TinyRouter()
    assert check_routing(model, batches, 0.9) == {"samples_checked": 2, "route_errors": 0, "prediction_errors": 0}
    for threshold, mode, expected in ((0.9, "actual_dynamic", 1), (0., "actual_dynamic", 2), (0.9, "forced_fallback", 0)):
        times, counts, memory = time_mode(model, batches, mode, threshold, 1, 3, torch.device("cpu"))
        np.testing.assert_array_equal(counts, [expected] * 3)
        assert np.isfinite(times).all() and (times > 0).all()
        assert memory["cpu_mode_peak_bytes"] is None
        assert memory["cuda_allocated_peak_bytes"] is None
