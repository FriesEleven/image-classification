import pytest
import torch

from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from image_classification.models.attention import CrossStageChannelAttention
from scripts.launch_csgha_v6 import EXPERIMENT_TAG, JOBS, validated_plan


def test_v6_plan_is_six_new_serial_matched_validation_only_runs():
    plan = validated_plan()
    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert all(not run["resolved_config"]["evaluate_test"] for run in plan)
    assert all(run["resolved_config"]["cuda_graph"] for run in plan)
    assert all(f"_serial_{EXPERIMENT_TAG}_seed" in run["experiment_id"] for run in plan)
    assert {run["resolved_config"]["model_type"] for run in plan} == {
        "hybrid_leaky", "csgha_v6",
    }
    assert JOBS == 1


def test_v6_retry_plan_uses_fresh_ids():
    plan = validated_plan("v6s2")
    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert all("_serial_v6s2_seed" in run["experiment_id"] for run in plan)


def test_v6_changes_only_guidance_scale_cap_from_v5():
    torch.manual_seed(29)
    v5 = CrossStageChannelAttention(
        64, 24, deep_activation="leaky_relu", guidance_output_normalization="rms",
    )
    torch.manual_seed(29)
    v6 = CrossStageChannelAttention(
        64,
        24,
        deep_activation="leaky_relu",
        guidance_output_normalization="rms",
        guidance_scale_cap=0.25,
    )
    v5.guidance_scale.data.fill_(0.8)
    v6.guidance_scale.data.fill_(0.8)
    inputs = torch.randn(2, 64, 4, 4)
    descriptors = torch.randn(2, 24)
    v5_deep, v5_raw, v5_bounded, v5_gated = v5.attention_logits(inputs, descriptors)
    v6_deep, v6_raw, v6_bounded, v6_gated = v6.attention_logits(inputs, descriptors)
    assert torch.equal(v5_deep, v6_deep)
    assert torch.equal(v5_raw, v6_raw)
    assert torch.equal(v5_bounded, v6_bounded)
    assert torch.allclose(v6_gated, 0.25 * v5_gated)
    assert v6_gated.abs().max() <= 0.25
    assert sum(parameter.numel() for parameter in v5.parameters()) == sum(
        parameter.numel() for parameter in v6.parameters()
    )


def test_v5_and_v6_checkpoints_fail_strict_cross_loading():
    v5 = CrossStageChannelAttention(
        64, 24, deep_activation="leaky_relu", guidance_output_normalization="rms",
    )
    v6 = CrossStageChannelAttention(
        64,
        24,
        deep_activation="leaky_relu",
        guidance_output_normalization="rms",
        guidance_scale_cap=0.25,
    )
    with pytest.raises(RuntimeError, match="guidance_scale_cap_version"):
        v6.load_state_dict(v5.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="guidance_scale_cap_version"):
        v5.load_state_dict(v6.state_dict(), strict=True)


@pytest.mark.parametrize("cap", [0.0, -0.1, 1.1])
def test_guidance_scale_cap_is_validated(cap):
    with pytest.raises(ValueError, match="guidance_scale_cap"):
        CrossStageChannelAttention(64, 24, guidance_scale_cap=cap)


def test_v6_model_preserves_output_shape_and_records_versions():
    config = ExperimentConfig(model_type="csgha_v6", se_positions=(1, 2), cbam_positions=(7, 8))
    model = build_model(config).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    state = model.state_dict()
    assert sum("deep_activation_version" in name for name in state) == 2
    assert sum("guidance_output_normalization_version" in name for name in state) == 2
    assert sum("guidance_scale_cap_version" in name for name in state) == 2
    assert "guidance_cap_0.25" in config.architecture_version
