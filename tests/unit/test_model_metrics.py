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


def test_stage_sparse_metrics_count_all_selected_attention_as_main():
    config = ExperimentConfig(
        model_type="stage_sparse",
        eca_positions=(1, 2),
        se_positions=(7, 8),
        cbam_positions=(15, 16),
    )

    metrics = model_metrics(build_model(config), config)

    assert metrics["num_eca_modules"] == 2
    assert metrics["num_se_modules"] == 2
    assert metrics["num_cbam_modules"] == 2
    assert metrics["eca_positions"] == [1, 2]
    assert metrics["parameters_aux_attention"] == 0
    assert metrics["parameters_main_attention"] == (
        metrics["parameters_eca"] + metrics["parameters_se"] + metrics["parameters_cbam"]
    )


def test_multi_exit_metrics_keep_heads_separate_from_final_classifier():
    baseline_config = ExperimentConfig(model_type="mobilenetv2")
    exit_config = ExperimentConfig(
        model_type="multi_exit",
        exit_positions=(8, 16),
        exit_loss_weights=(0.2, 0.3),
    )
    baseline = model_metrics(build_model(baseline_config), baseline_config)
    multi_exit = model_metrics(build_model(exit_config), exit_config)

    assert multi_exit["num_exit_heads"] == 2
    assert multi_exit["exit_positions"] == [8, 16]
    assert multi_exit["parameters_exit_heads"] == (64 + 1) * 10 + (160 + 1) * 10
    assert multi_exit["flops_exit_head_adjustment"] == 64 * 10 + 160 * 10
    assert multi_exit["parameters_classifier"] == baseline["parameters_classifier"]
    assert multi_exit["parameters_backbone"] == baseline["parameters_backbone"]
