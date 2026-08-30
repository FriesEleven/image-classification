import torch

from image_classification.diagnostics.guidance import (
    derangement,
    load_historical_model,
    paired_metrics,
    projected_guidance,
)


def test_permutation_preserves_marginal_and_has_no_self_pairs():
    permutation = derangement(5000, 7301)
    assert torch.equal(permutation.sort().values, torch.arange(5000))
    assert not torch.any(permutation == torch.arange(5000))
    assert torch.equal(permutation, derangement(5000, 7301))


def test_paired_metrics_counts_benefit_and_harm():
    labels = torch.tensor([0, 1, 0])
    original = torch.tensor([[2., 1.], [2., 1.], [2., 1.]])
    modified = torch.tensor([[1., 2.], [1., 2.], [2., 1.]])
    result = paired_metrics(modified, labels, original)
    assert result["prediction_changes"] == 2
    assert result["originally_correct_now_wrong"] == 1
    assert result["originally_wrong_now_correct"] == 1
    assert result["delta_accuracy_pp"] == 0


def test_historical_formula_matches_actual_forward_for_each_version():
    config = {"dataset": "cifar10", "se_positions": [1, 2], "cbam_positions": [7, 8],
              "guidance_position": 2, "guidance_reduction": 4}
    for version, revision in (("v1", "6dc4c57"), ("v2", "82625b4"), ("v3", "f11d0af")):
        model, provenance = load_historical_model(revision, config)
        module = model.model.features[7].guided_cbam.channel_attention
        if version != "v1":
            module.guidance_scale.data.fill_(0.7)
        inputs, guide = torch.randn(2, 64, 4, 4), torch.randn(2, 24)
        deep = module.fc(module.avg_pool(inputs)) + module.fc(module.max_pool(inputs))
        contribution = projected_guidance(module, guide, version)[2]
        assert torch.allclose(module(inputs, guide), (deep + contribution[:, :, None, None]).sigmoid())
        assert len(provenance["reference_commit"]) == 40
