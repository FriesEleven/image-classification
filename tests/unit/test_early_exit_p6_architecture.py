from scripts.analysis.profile_early_exit_p6_architecture import select_boundary


def test_boundary_selection_uses_nearest_fraction_and_ordered_auxiliary():
    candidates = [
        {"position": 0, "fraction": 0.2},
        {"position": 1, "fraction": 0.4},
        {"position": 2, "fraction": 0.6},
        {"position": 3, "fraction": 0.8},
    ]
    assert select_boundary(candidates, 0.425)["position"] == 1
    assert select_boundary(candidates, 0.75, after=1)["position"] == 3
