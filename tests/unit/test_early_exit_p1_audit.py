import pytest

from scripts.analysis.audit_early_exit_p1 import sample_stats, split_fingerprint


def test_p1_audit_uses_sample_standard_deviation():
    result = sample_stats([1.0, 2.0, 3.0])

    assert result == {
        "n": 3,
        "mean_percent": 2.0,
        "sample_sd_percent": 1.0,
    }


def test_p1_split_fingerprint_ignores_training_seed_metadata():
    split = {
        "split_seed": 7,
        "training_seed": 54,
        "train_indices": [0, 1],
        "validation_indices": [2],
        "calibration_indices": [3],
    }
    changed_metadata = {**split, "training_seed": 55}

    assert split_fingerprint(split) == split_fingerprint(changed_metadata)


@pytest.mark.parametrize("values", [[], [1.0], [1.0, float("nan")]])
def test_p1_audit_rejects_invalid_statistics(values):
    with pytest.raises(ValueError):
        sample_stats(values)
