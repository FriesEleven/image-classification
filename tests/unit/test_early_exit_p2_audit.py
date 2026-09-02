import pytest

from scripts.analysis.audit_early_exit_p2 import sample_stats, split_fingerprint


def test_p2_audit_uses_sample_standard_deviation():
    assert sample_stats([1.0, 2.0, 3.0]) == {
        "n": 3,
        "mean_percent": 2.0,
        "sample_sd_percent": 1.0,
    }


def test_p2_split_fingerprint_ignores_training_seed_metadata():
    split = {
        "split_seed": 7,
        "training_seed": 57,
        "train_indices": [0, 1],
        "validation_indices": [2],
        "calibration_indices": [3],
    }

    assert split_fingerprint(split) == split_fingerprint({**split, "training_seed": 58})


@pytest.mark.parametrize("values", [[], [1.0], [1.0, float("nan")]])
def test_p2_audit_rejects_invalid_statistics(values):
    with pytest.raises(ValueError):
        sample_stats(values)
