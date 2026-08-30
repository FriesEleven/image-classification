import torch

from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from scripts.diagnostics.check_csgha_guidance import summarize_guidance


def test_guidance_diagnostic_reports_both_targets_and_zero_initial_gate():
    config = ExperimentConfig(
        model_type="csgha",
        se_positions=(1, 2),
        cbam_positions=(7, 8),
        guidance_position=2,
    )
    model = build_model(config).eval()

    reports = summarize_guidance(model, torch.randn(2, 3, 32, 32))

    assert len(reports) == 2
    assert all(report["raw_guidance_to_deep_ratio"] >= 0 for report in reports)
    assert all(report["guidance_scale_parameter"] == 0 for report in reports)
    assert all(report["gated_guidance_logits_abs_mean"] == 0 for report in reports)
