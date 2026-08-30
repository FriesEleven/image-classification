"""Single model-construction entry point."""

from torch import nn

from image_classification.config import ExperimentConfig

from .eca import ECAMobileNetV2
from .mobilenetv2 import (
    BaseMobileNetV2,
    CBAMMobileNetV2,
    CSGHAMobileNetV2,
    HybridAttentionMobileNetV2,
    SEMobileNetV2,
)


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model_type == "mobilenetv2":
        return BaseMobileNetV2(num_classes=config.num_classes)
    if config.model_type == "eca":
        return ECAMobileNetV2(num_classes=config.num_classes)
    if config.model_type == "cbam":
        return CBAMMobileNetV2(num_classes=config.num_classes, aux_positions=config.aux_positions)
    if config.model_type == "se":
        return SEMobileNetV2(num_classes=config.num_classes, aux_positions=config.aux_positions)
    if config.model_type in {"hybrid", "hybrid_leaky"}:
        return HybridAttentionMobileNetV2(
            num_classes=config.num_classes,
            se_positions=config.se_positions,
            cbam_positions=config.cbam_positions,
            deep_activation="leaky_relu" if config.model_type == "hybrid_leaky" else "relu",
        )
    if config.model_type in {"csgha", "csgha_v4"}:
        return CSGHAMobileNetV2(
            num_classes=config.num_classes,
            se_positions=config.se_positions,
            cbam_positions=config.cbam_positions,
            guidance_position=config.guidance_position,
            guidance_reduction=config.guidance_reduction,
            deep_activation="leaky_relu" if config.model_type == "csgha_v4" else "relu",
        )
    raise ValueError(f"Unsupported model type: {config.model_type}")
