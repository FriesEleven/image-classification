"""Single model-construction entry point."""

from torch import nn

from image_classification.config import ExperimentConfig

from .mobilenetv2 import (
    BaseMobileNetV2,
    CBAMMobileNetV2,
    CSGHAMobileNetV2,
    ECAMobileNetV2,
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
    if config.model_type == "hybrid":
        return HybridAttentionMobileNetV2(
            num_classes=config.num_classes,
            se_positions=config.se_positions,
            cbam_positions=config.cbam_positions,
        )
    if config.model_type == "csgha":
        return CSGHAMobileNetV2(
            num_classes=config.num_classes,
            se_positions=config.se_positions,
            cbam_positions=config.cbam_positions,
            guidance_position=config.guidance_position,
            guidance_reduction=config.guidance_reduction,
        )
    raise ValueError(f"Unsupported model type: {config.model_type}")
