from pathlib import Path

from image_classification.config import load_config
from scripts.launch_early_exit_p5c import VARIANT_CONTRACTS, _variant
from scripts.run_baselines import build_plan, load_sweep


def test_p5c_sweeps_are_exactly_eighteen_development_only_runs():
    runs = []
    for path in (
        "configs/sweeps/early_exit_p5c_cifar10.yaml",
        "configs/sweeps/early_exit_p5c_cifar100.yaml",
    ):
        runs.extend(build_plan(load_sweep(Path(path)), "test"))
    assert len(runs) == 18
    assert len({run["experiment_id"] for run in runs}) == 18
    assert all(not run["resolved_config"]["evaluate_test"] for run in runs)
    assert all(run["resolved_config"]["measure_inference"] is False for run in runs)


def test_p5c_effective_ce_weights_match_frozen_plan():
    expected = {"a1": [0.1], "a2": [0.1], "a3": [0.1, 0.15]}
    for dataset in ("cifar10", "cifar100"):
        for variant, values in expected.items():
            path = f"configs/experiments/early_exit_p5c_{dataset}_{variant}.yaml"
            config = load_config(["--config", path])
            actual = [
                weight * (1 - config.exit_distillation_alpha)
                for weight in config.exit_loss_weights
            ]
            assert actual == values
            assert _variant(path) == variant
            contract = VARIANT_CONTRACTS[variant]
            assert list(config.exit_positions) == contract["exit_positions"]
