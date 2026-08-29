from pathlib import Path

import pytest

from image_classification.config import load_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("filename", "cbam_positions"),
    [
        ("position_se_shallow_cbam_shallow.yaml", (1, 2)),
        ("position_se_shallow_cbam_middle.yaml", (7, 8)),
        ("position_se_shallow_cbam_deep.yaml", (15, 16)),
    ],
)
def test_position_screening_configs_are_validation_only(filename, cbam_positions):
    config = load_config(["--config", str(ROOT / "configs/experiments" / filename)])

    assert config.model_type == "hybrid"
    assert config.dataset == "cifar10"
    assert config.se_positions == (1, 2)
    assert config.cbam_positions == cbam_positions
    assert config.evaluate_test is False


def test_csgha_candidate_uses_block_two_descriptor_and_middle_targets():
    config = load_config(
        ["--config", str(ROOT / "configs/experiments/csgha_se_shallow_cbam_middle.yaml")]
    )

    assert config.model_type == "csgha"
    assert config.se_positions == (1, 2)
    assert config.guidance_position == 2
    assert config.guidance_reduction == 4
    assert config.cbam_positions == (7, 8)
    assert config.evaluate_test is False
