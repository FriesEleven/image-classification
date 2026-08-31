import torch

from image_classification.config import ExperimentConfig
from image_classification.diagnostics.guidance_v4 import (
    channel_modules,
    projected_guidance,
)
from image_classification.models import build_model
from image_classification.models.attention import _rms_normalize_channels


def test_v4_projected_guidance_matches_model_attention_logits():
    config = ExperimentConfig(
        model_type="csgha_v4", se_positions=(1, 2), cbam_positions=(7, 8),
    )
    model = build_model(config)
    module = next(iter(channel_modules(model, "csgha_v4").values()))
    module.guidance_scale.data.fill_(0.4)
    features = torch.randn(3, 64, 4, 4)
    descriptors = torch.randn(3, 24)
    deep, raw_4d, bounded_4d, contribution_4d = module.attention_logits(features, descriptors)
    raw, bounded, contribution = projected_guidance(module, descriptors)
    assert deep.shape == (3, 64, 1, 1)
    assert torch.equal(raw, raw_4d.flatten(1))
    assert torch.equal(bounded, bounded_4d.flatten(1))
    assert torch.equal(contribution, contribution_4d.flatten(1))


def test_v4_and_control_select_exactly_two_matched_modules():
    control_config = ExperimentConfig(
        model_type="hybrid_leaky", se_positions=(1, 2), cbam_positions=(7, 8),
    )
    guided_config = ExperimentConfig(
        model_type="csgha_v4", se_positions=(1, 2), cbam_positions=(7, 8),
    )
    control = channel_modules(build_model(control_config), "hybrid_leaky")
    guided = channel_modules(build_model(guided_config), "csgha_v4")
    assert sorted(control) == [
        "model.features.7.cbam.channel_attention",
        "model.features.8.cbam.channel_attention",
    ]
    assert sorted(guided) == [
        "model.features.7.guided_cbam.channel_attention",
        "model.features.8.guided_cbam.channel_attention",
    ]


def test_v4_formula_is_bounded_for_new_version():
    config = ExperimentConfig(
        model_type="csgha_v4", se_positions=(1, 2), cbam_positions=(7, 8),
    )
    module = next(iter(channel_modules(build_model(config), "csgha_v4").values()))
    raw, bounded, contribution = projected_guidance(module, torch.randn(4, 24))
    assert torch.equal(bounded, raw.tanh())
    assert torch.count_nonzero(contribution) == 0


def test_v5_diagnostic_formula_matches_rms_normalized_model_path():
    config = ExperimentConfig(
        model_type="csgha_v5", se_positions=(1, 2), cbam_positions=(7, 8),
    )
    module = next(iter(channel_modules(build_model(config), "csgha_v5").values()))
    raw, bounded, contribution = projected_guidance(module, torch.randn(4, 24))
    assert torch.allclose(bounded, _rms_normalize_channels(raw).tanh())
    assert torch.count_nonzero(contribution) == 0
