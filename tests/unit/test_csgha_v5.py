import pytest
import torch

from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from image_classification.models.attention import (
    CrossStageChannelAttention,
    _rms_normalize_channels,
)
from scripts.launch_csgha_v5 import EXPERIMENT_TAG, JOBS, validated_plan


def test_v5_plan_is_six_new_matched_validation_only_runs():
    plan = validated_plan()
    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert all(not run["resolved_config"]["evaluate_test"] for run in plan)
    assert all(run["resolved_config"]["cuda_graph"] for run in plan)
    assert all(f"_serial_{EXPERIMENT_TAG}_seed" in run["experiment_id"] for run in plan)
    assert {run["resolved_config"]["model_type"] for run in plan} == {"hybrid_leaky", "csgha_v5"}
    assert JOBS == 1


def test_v5_retry_plan_uses_fresh_ids():
    plan = validated_plan("s2")
    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert all("_serial_s2_seed" in run["experiment_id"] for run in plan)


def test_rms_channel_normalization_is_scale_invariant_and_unit_rms():
    torch.manual_seed(11)
    raw = torch.randn(4, 64, 1, 1)
    normalized = _rms_normalize_channels(raw)
    scaled = _rms_normalize_channels(raw * 100.0)
    rms = normalized.square().mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-6)
    assert torch.allclose(normalized, scaled, atol=2e-6)


def test_rms_normalization_prevents_scale_driven_tanh_saturation():
    torch.manual_seed(19)
    raw = torch.randn(8, 64, 1, 1) * 100.0
    v4_saturation = (raw.tanh().abs() > 0.99).float().mean()
    v5_saturation = (_rms_normalize_channels(raw).tanh().abs() > 0.99).float().mean()
    assert v4_saturation > 0.95
    assert v5_saturation < 0.02


def test_v5_changes_only_guidance_bounding_formula():
    torch.manual_seed(23)
    v4 = CrossStageChannelAttention(64, 24, deep_activation="leaky_relu")
    torch.manual_seed(23)
    v5 = CrossStageChannelAttention(
        64,
        24,
        deep_activation="leaky_relu",
        guidance_output_normalization="rms",
    )
    inputs = torch.randn(2, 64, 4, 4)
    descriptor = torch.randn(2, 24)
    v4_deep, v4_raw, v4_bounded, _ = v4.attention_logits(inputs, descriptor)
    v5_deep, v5_raw, v5_bounded, _ = v5.attention_logits(inputs, descriptor)
    assert torch.equal(v4_deep, v5_deep)
    assert torch.equal(v4_raw, v5_raw)
    assert torch.equal(v4_bounded, v4_raw.tanh())
    assert torch.allclose(v5_bounded, _rms_normalize_channels(v5_raw).tanh())
    assert sum(parameter.numel() for parameter in v4.parameters()) == sum(
        parameter.numel() for parameter in v5.parameters()
    )


def test_v4_and_v5_checkpoints_fail_strict_cross_loading():
    v4 = CrossStageChannelAttention(64, 24, deep_activation="leaky_relu")
    v5 = CrossStageChannelAttention(
        64,
        24,
        deep_activation="leaky_relu",
        guidance_output_normalization="rms",
    )
    with pytest.raises(RuntimeError, match="guidance_output_normalization_version"):
        v5.load_state_dict(v4.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="guidance_output_normalization_version"):
        v4.load_state_dict(v5.state_dict(), strict=True)


def test_v5_model_preserves_output_shape_and_records_versions():
    config = ExperimentConfig(model_type="csgha_v5", se_positions=(1, 2), cbam_positions=(7, 8))
    model = build_model(config).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    state = model.state_dict()
    assert sum("deep_activation_version" in name for name in state) == 2
    assert sum("guidance_output_normalization_version" in name for name in state) == 2
    assert "rms_normalized_guidance" in config.architecture_version
