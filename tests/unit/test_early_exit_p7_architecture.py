from scripts.analysis.profile_early_exit_p7_architecture import select_boundary


def test_select_boundary_uses_target_fraction_and_order():
    candidates = [
        {"position": 1, "fraction": 0.2},
        {"position": 2, "fraction": 0.4},
        {"position": 3, "fraction": 0.8},
        {"position": 4, "fraction": 0.9},
    ]
    assert select_boundary(candidates, 0.425)["position"] == 2
    assert select_boundary(candidates, 0.825, after=2)["position"] == 3
