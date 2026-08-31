import pytest
import torch

from image_classification.config import ExperimentConfig
from image_classification.diagnostics.guidance import load_historical_model
from image_classification.models import build_model
from image_classification.models.attention import ChannelAttention, CrossStageChannelAttention
from scripts.launch_csgha_v4 import JOBS, validated_plan


def test_v4_plan_is_six_matched_validation_only_runs():
    plan = validated_plan()
    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert all(not run["resolved_config"]["evaluate_test"] for run in plan)
    assert all(run["resolved_config"]["cuda_graph"] for run in plan)
    assert all("_perf2_seed" in run["experiment_id"] for run in plan)
    assert JOBS == 2
    assert all(run["resolved_config"]["torch_num_threads"] == 1 for run in plan)
    assert all(not run["resolved_config"]["measure_inference"] for run in plan)


def test_v4_retry_plan_uses_six_new_matched_ids():
    plan = validated_plan("retry1")
    assert len(plan) == 6
    assert len({run["experiment_id"] for run in plan}) == 6
    assert all("_perf2_retry1_seed" in run["experiment_id"] for run in plan)


@pytest.mark.parametrize("model_type", ["hybrid_leaky", "csgha_v4"])
def test_new_architectures_preserve_shapes_and_record_versions(model_type):
    config = ExperimentConfig(model_type=model_type, se_positions=(1, 2), cbam_positions=(7, 8))
    model = build_model(config).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert "leaky" in config.architecture_version
    assert sum("deep_activation_version" in name for name in model.state_dict()) == 2


def test_leaky_path_has_gradient_when_relu_is_completely_inactive():
    relu = ChannelAttention(16)
    leaky = ChannelAttention(16, deep_activation="leaky_relu")
    for module in (relu, leaky):
        module.fc[0].weight.data.fill_(-0.1)
        module.fc[2].weight.data.fill_(0.1)
        module(torch.ones(2, 16, 2, 2)).sum().backward()
    assert torch.count_nonzero(relu.fc[0].weight.grad) == 0
    assert torch.count_nonzero(leaky.fc[0].weight.grad) > 0


def test_parameterless_activation_change_is_not_silently_checkpoint_compatible():
    original = CrossStageChannelAttention(64, 24)
    revised = CrossStageChannelAttention(64, 24, deep_activation="leaky_relu")
    with pytest.raises(RuntimeError, match="deep_activation_version"):
        revised.load_state_dict(original.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="deep_activation_version"):
        original.load_state_dict(revised.state_dict(), strict=True)
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in revised.parameters())


def test_v4_preserves_guidance_formula_and_zero_initialization():
    attention = CrossStageChannelAttention(64, 24, deep_activation="leaky_relu")
    deep, raw, bounded, gated = attention.attention_logits(torch.randn(2, 64, 4, 4), torch.randn(2, 24))
    assert torch.equal(bounded, raw.tanh())
    assert torch.count_nonzero(gated) == 0
    assert deep.shape == gated.shape


def test_existing_v3_initialization_and_forward_are_unchanged():
    config = ExperimentConfig(model_type="csgha", se_positions=(1, 2), cbam_positions=(7, 8))
    torch.manual_seed(777)
    old, _ = load_historical_model("f11d0af", config.to_dict())
    torch.manual_seed(777)
    current = build_model(config)
    assert old.state_dict().keys() == current.state_dict().keys()
    assert all(torch.equal(value, current.state_dict()[key]) for key, value in old.state_dict().items())
    sample = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(old.eval()(sample), current.eval()(sample))
