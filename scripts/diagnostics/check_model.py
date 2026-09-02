"""Build every architecture and run a small forward pass."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_classification.config import ExperimentConfig
from image_classification.models import build_model

CONFIGS = (
    ExperimentConfig(model_type="mobilenetv2"),
    ExperimentConfig(model_type="mobilenetv2", dataset="cifar100"),
    ExperimentConfig(model_type="eca"),
    ExperimentConfig(model_type="cbam", aux_positions=(1, 2)),
    ExperimentConfig(model_type="se", aux_positions=(1, 2)),
    ExperimentConfig(model_type="hybrid", se_positions=(1, 2), cbam_positions=(15, 16)),
    ExperimentConfig(
        model_type="csgha",
        se_positions=(1, 2),
        cbam_positions=(7, 8),
        guidance_position=2,
    ),
    ExperimentConfig(model_type="hybrid_leaky", se_positions=(1, 2), cbam_positions=(7, 8)),
    ExperimentConfig(model_type="csgha_v4", se_positions=(1, 2), cbam_positions=(7, 8)),
    ExperimentConfig(model_type="csgha_v5", se_positions=(1, 2), cbam_positions=(7, 8)),
    ExperimentConfig(model_type="csgha_v6", se_positions=(1, 2), cbam_positions=(7, 8)),
    ExperimentConfig(
        model_type="stage_sparse",
        eca_positions=(1, 2),
        se_positions=(7, 8),
        cbam_positions=(15, 16),
    ),
    ExperimentConfig(
        model_type="multi_exit",
        exit_positions=(8, 16),
        exit_loss_weights=(0.2, 0.3),
    ),
)


def main() -> int:
    sample = torch.randn(2, 3, 32, 32)
    for config in CONFIGS:
        model = build_model(config).eval()
        with torch.no_grad():
            output = model(sample)
        outputs = output if isinstance(output, tuple) else (output,)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{config.dataset:8s} {config.model_type:12s} "
            f"outputs={[tuple(value.shape) for value in outputs]} parameters={parameters:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
