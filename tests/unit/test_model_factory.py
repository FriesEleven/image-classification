import pytest
import torch

from image_classification.config import ExperimentConfig
from image_classification.models import build_model


@pytest.mark.parametrize(
    "config",
    [
        ExperimentConfig(model_type="mobilenetv2"),
        ExperimentConfig(model_type="eca"),
        ExperimentConfig(model_type="cbam", aux_positions=(1, 2)),
        ExperimentConfig(model_type="se", aux_positions=(1, 2)),
        ExperimentConfig(model_type="hybrid", se_positions=(1, 2), cbam_positions=(15, 16)),
        ExperimentConfig(
            model_type="stage_sparse",
            eca_positions=(1, 2),
            se_positions=(7, 8),
            cbam_positions=(15, 16),
        ),
        ExperimentConfig(
            model_type="csgha",
            se_positions=(1, 2),
            cbam_positions=(7, 8),
            guidance_position=2,
        ),
    ],
)
def test_model_output_shape(config):
    model = build_model(config).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, config.num_classes)


def test_cifar100_model_has_100_class_output():
    config = ExperimentConfig(model_type="mobilenetv2", dataset="cifar100")
    model = build_model(config).eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 100)


def test_cifar100_csgha_has_100_class_output():
    config = ExperimentConfig(
        model_type="csgha",
        dataset="cifar100",
        se_positions=(1, 2),
        cbam_positions=(7, 8),
        guidance_position=2,
    )
    model = build_model(config).eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 100)


def test_empty_stage_sparse_member_matches_baseline_initialization():
    torch.manual_seed(71)
    baseline = build_model(ExperimentConfig(model_type="mobilenetv2"))
    torch.manual_seed(71)
    sparse_none = build_model(ExperimentConfig(model_type="stage_sparse"))

    assert baseline.state_dict().keys() == sparse_none.state_dict().keys()
    assert all(
        torch.equal(baseline.state_dict()[name], sparse_none.state_dict()[name])
        for name in baseline.state_dict()
    )


def test_multi_exit_outputs_are_ordered_and_prefix_execution_matches():
    config = ExperimentConfig(
        model_type="multi_exit",
        exit_positions=(8, 16),
        exit_loss_weights=(0.2, 0.3),
    )
    model = build_model(config).eval()
    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        final_logits, exit8_logits, exit16_logits = model(inputs)

    assert final_logits.shape == exit8_logits.shape == exit16_logits.shape == (2, 10)
    with torch.no_grad():
        torch.testing.assert_close(model.forward_to_exit(inputs, 8), exit8_logits)
        torch.testing.assert_close(model.forward_to_exit(inputs, 16), exit16_logits)
        torch.testing.assert_close(model.forward_to_exit(inputs, None), final_logits)
    with pytest.raises(ValueError, match="Unknown"):
        model.forward_to_exit(inputs, 7)


def test_multi_exit_dynamic_policy_continues_only_unresolved_samples():
    config = ExperimentConfig(
        model_type="multi_exit",
        exit_positions=(8, 16),
        exit_loss_weights=(0.2, 0.3),
    )
    model = build_model(config).eval()
    inputs = torch.randn(8, 3, 32, 32)
    exit16_calls = []
    handle = model.exit_heads["16"].register_forward_hook(
        lambda _module, _inputs, _output: exit16_calls.append(True)
    )

    with torch.no_grad():
        final_logits, exit8_logits, _exit16_logits = model(inputs)
        confidence = torch.softmax(exit8_logits, dim=1).amax(dim=1)
        ordered = confidence.sort().values
        threshold = float((ordered[3] + ordered[4]) / 2)
        exit16_calls.clear()
        routed_logits, paths = model.forward_with_policy(inputs, threshold)
    handle.remove()

    early = confidence >= threshold
    assert early.any() and (~early).any()
    torch.testing.assert_close(routed_logits[early], exit8_logits[early])
    torch.testing.assert_close(routed_logits[~early], final_logits[~early])
    torch.testing.assert_close(paths, torch.where(early, 0, 1))
    assert exit16_calls == []


def test_multi_exit_dynamic_policy_supports_all_early_and_all_final():
    model = build_model(
        ExperimentConfig(
            model_type="multi_exit",
            exit_positions=(8, 16),
            exit_loss_weights=(0.2, 0.3),
        )
    ).eval()
    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        final_logits, exit8_logits, _exit16_logits = model(inputs)
        all_early_logits, all_early_paths = model.forward_with_policy(inputs, 0.0)
        all_final_logits, all_final_paths = model.forward_with_policy(inputs, 2.0)

    torch.testing.assert_close(all_early_logits, exit8_logits)
    torch.testing.assert_close(all_final_logits, final_logits)
    torch.testing.assert_close(all_early_paths, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(all_final_paths, torch.ones(2, dtype=torch.long))


def test_multi_exit_dynamic_policy_rejects_training_mode():
    model = build_model(
        ExperimentConfig(
            model_type="multi_exit",
            exit_positions=(8, 16),
            exit_loss_weights=(0.2, 0.3),
        )
    )

    with pytest.raises(RuntimeError, match="inference-only"):
        model.forward_with_policy(torch.randn(1, 3, 32, 32), 0.9)


def test_multi_exit_adds_only_heads_to_identically_initialized_backbone():
    torch.manual_seed(73)
    baseline = build_model(ExperimentConfig(model_type="mobilenetv2"))
    torch.manual_seed(73)
    multi_exit = build_model(
        ExperimentConfig(
            model_type="multi_exit",
            exit_positions=(8, 16),
            exit_loss_weights=(0.2, 0.3),
        )
    )

    assert all(
        torch.equal(value, multi_exit.state_dict()[name])
        for name, value in baseline.state_dict().items()
    )
