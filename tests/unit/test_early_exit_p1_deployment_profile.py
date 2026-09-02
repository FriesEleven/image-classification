from scripts.diagnostics.profile_early_exit_p1_deployment import _median


def test_deployment_profile_uses_median_across_paired_rounds():
    assert _median([4.0, 1.0, 3.0, 2.0, 100.0]) == 3.0
