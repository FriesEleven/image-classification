from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from image_classification.training.benchmark import model_metrics


def test_csgha_metrics_separate_guided_attention_and_projection_parameters():
    config = ExperimentConfig(
        model_type="csgha",
        se_positions=(1, 2),
        cbam_positions=(7, 8),
        guidance_position=2,
    )

    metrics = model_metrics(build_model(config), config)

    assert metrics["num_guided_cbam_modules"] == 2
    assert metrics["parameters_guided_cbam"] > 0
    assert metrics["parameters_cross_stage_projection"] > 0
    assert metrics["parameters_cross_stage_normalization"] == 48 * 2
    assert metrics["parameters_cross_stage_scale"] == 2
    assert metrics["parameters_cross_stage_guidance"] == (
        metrics["parameters_cross_stage_projection"]
        + metrics["parameters_cross_stage_normalization"]
        + metrics["parameters_cross_stage_scale"]
    )
    assert metrics["guidance_source_channels"] == 24
    assert metrics["guidance_target_channels"] == {7: 64, 8: 64}
    assert metrics["parameters_total"] == (
        metrics["parameters_backbone"]
        + metrics["parameters_classifier"]
        + metrics["parameters_guided_cbam"]
        + metrics["parameters_se"]
    )
