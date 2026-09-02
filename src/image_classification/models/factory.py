"""Single model-construction entry point."""

from torch import nn

from image_classification.config import ExperimentConfig

from .eca import ECAMobileNetV2
from .mobilenetv2 import (
    BaseMobileNetV2,
    CBAMMobileNetV2,
    CSGHAMobileNetV2,
    HybridAttentionMobileNetV2,
    MultiExitMobileNetV2,
    SEMobileNetV2,
    StageSparseAttentionMobileNetV2,
)


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model_type == "mobilenetv2":
        return BaseMobileNetV2(num_classes=config.num_classes)
    if config.model_type == "eca":
        return ECAMobileNetV2(num_classes=config.num_classes)
    if config.model_type == "multi_exit":
        return MultiExitMobileNetV2(
            num_classes=config.num_classes,
            exit_positions=config.exit_positions,
        )
    if config.model_type == "cbam":
        return CBAMMobileNetV2(num_classes=config.num_classes, aux_positions=config.aux_positions)
    if config.model_type == "se":
        return SEMobileNetV2(num_classes=config.num_classes, aux_positions=config.aux_positions)
    if config.model_type == "stage_sparse":
        return StageSparseAttentionMobileNetV2(
            num_classes=config.num_classes,
            eca_positions=config.eca_positions,
            se_positions=config.se_positions,
            cbam_positions=config.cbam_positions,
        )
    if config.model_type in {"hybrid", "hybrid_leaky"}:
        return HybridAttentionMobileNetV2(
            num_classes=config.num_classes,
            se_positions=config.se_positions,
            cbam_positions=config.cbam_positions,
            deep_activation="leaky_relu" if config.model_type == "hybrid_leaky" else "relu",
        )
    if config.model_type in {"csgha", "csgha_v4", "csgha_v5", "csgha_v6"}:
        return CSGHAMobileNetV2(
            num_classes=config.num_classes,
            se_positions=config.se_positions,
            cbam_positions=config.cbam_positions,
            guidance_position=config.guidance_position,
            guidance_reduction=config.guidance_reduction,
            deep_activation=(
                "leaky_relu"
                if config.model_type in {"csgha_v4", "csgha_v5", "csgha_v6"} else "relu"
            ),
            guidance_output_normalization=(
                "rms" if config.model_type in {"csgha_v5", "csgha_v6"} else "none"
            ),
            guidance_scale_cap=0.25 if config.model_type == "csgha_v6" else 1.0,
        )
    raise ValueError(f"Unsupported model type: {config.model_type}")
