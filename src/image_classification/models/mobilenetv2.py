"""MobileNetV2 variants used by the experiments."""

import math

import torch
from torch import nn
from torchvision.models import mobilenet_v2

from .attention import CBAM, CrossStageGuidedCBAM, SEBlock
from .eca import ECALayer


def _output_channels(module: nn.Module) -> int | None:
    convolution = getattr(module, "conv", None)
    if isinstance(convolution, nn.Sequential):
        for layer in reversed(convolution):
            if hasattr(layer, "out_channels"):
                return int(layer.out_channels)
    if hasattr(convolution, "out_channels"):
        return int(convolution.out_channels)
    return None


class BaseMobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        self.model = mobilenet_v2(weights=None, width_mult=width_mult)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class EarlyExitHead(nn.Module):
    """A deliberately small classifier attached to an intermediate feature map."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(inputs).flatten(1))


class MultiExitMobileNetV2(nn.Module):
    """MobileNetV2 with ordered intermediate heads for exploratory early exiting."""

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        exit_positions: tuple[int, ...] = (8, 16),
    ):
        super().__init__()
        if not exit_positions:
            raise ValueError("At least one early-exit position is required")
        self.model = mobilenet_v2(weights=None, width_mult=width_mult)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        self.exit_positions = tuple(exit_positions)
        heads = {}
        for position in self.exit_positions:
            channels = _output_channels(self.model.features[position])
            if channels is None:
                raise ValueError(f"Cannot determine output channels for exit layer {position}")
            heads[str(position)] = EarlyExitHead(channels, num_classes)
        self.exit_heads = nn.ModuleDict(heads)

    def _final_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.classifier(features.mean((2, 3)))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = inputs
        exit_logits = []
        for index, module in enumerate(self.model.features):
            outputs = module(outputs)
            if index in self.exit_positions:
                exit_logits.append(self.exit_heads[str(index)](outputs))
        return (self._final_logits(outputs), *exit_logits)

    def forward_to_exit(
        self, inputs: torch.Tensor, exit_position: int | None,
    ) -> torch.Tensor:
        """Execute only the prefix required by one exit, or the full final path."""

        if exit_position is not None and exit_position not in self.exit_positions:
            raise ValueError(f"Unknown early-exit position: {exit_position}")
        outputs = inputs
        for index, module in enumerate(self.model.features):
            outputs = module(outputs)
            if index == exit_position:
                return self.exit_heads[str(index)](outputs)
        return self._final_logits(outputs)

    def forward_with_policy(
        self,
        inputs: torch.Tensor,
        confidence_threshold: float,
        exit_position: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route each sample at one exit and continue only the unresolved subset.

        This deployment path is intentionally inference-only. It avoids recomputing
        the prefix for samples that fall through to the final classifier and skips
        every auxiliary head other than the selected routing exit.
        """

        if self.training:
            raise RuntimeError("Dynamic early-exit routing is inference-only")
        if inputs.ndim != 4 or inputs.shape[0] == 0:
            raise ValueError("Dynamic early-exit routing requires a non-empty image batch")
        if exit_position not in self.exit_positions:
            raise ValueError(f"Unknown early-exit position: {exit_position}")
        if not math.isfinite(confidence_threshold):
            raise ValueError("confidence_threshold must be finite")

        outputs = inputs
        for index, module in enumerate(self.model.features):
            outputs = module(outputs)
            if index != exit_position:
                continue
            exit_logits = self.exit_heads[str(index)](outputs)
            confidence = torch.softmax(exit_logits.float(), dim=1).amax(dim=1)
            early = confidence >= confidence_threshold
            routed_logits = torch.empty_like(exit_logits)
            paths = torch.ones(inputs.shape[0], dtype=torch.long, device=inputs.device)
            routed_logits[early] = exit_logits[early]
            paths[early] = 0
            unresolved = ~early
            if bool(unresolved.any()):
                remaining = outputs[unresolved]
                for later_module in self.model.features[index + 1 :]:
                    remaining = later_module(remaining)
                routed_logits[unresolved] = self._final_logits(remaining)
            return routed_logits, paths
        raise RuntimeError("Configured early exit was not reached")


class _SparseAttentionMobileNetV2(nn.Module):
    attention_attribute = ""

    def __init__(self, num_classes: int, width_mult: float, positions: tuple[int, ...]):
        super().__init__()
        self.model = mobilenet_v2(weights=None, width_mult=width_mult)
        self.positions = tuple(positions)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def _attach(self, factory) -> None:
        for index, module in enumerate(self.model.features):
            if index not in self.positions:
                continue
            channels = _output_channels(module)
            if channels is None:
                raise ValueError(f"Cannot determine output channels for feature layer {index}")
            setattr(module, self.attention_attribute, factory(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for index, module in enumerate(self.model.features):
            outputs = module(outputs)
            attention = getattr(module, self.attention_attribute, None)
            if index in self.positions and attention is not None:
                outputs = attention(outputs)
        outputs = outputs.mean((2, 3))
        return self.model.classifier(outputs)


class CBAMMobileNetV2(_SparseAttentionMobileNetV2):
    attention_attribute = "cbam"

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, aux_positions: tuple[int, ...] = ()):
        super().__init__(num_classes, width_mult, tuple(aux_positions))
        self.aux_positions = self.positions
        self._attach(CBAM)


class SEMobileNetV2(_SparseAttentionMobileNetV2):
    attention_attribute = "se"

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, aux_positions: tuple[int, ...] = ()):
        super().__init__(num_classes, width_mult, tuple(aux_positions))
        self.aux_positions = self.positions
        self._attach(SEBlock)


class StageSparseAttentionMobileNetV2(nn.Module):
    """MobileNetV2 with one independently selected attention type per feature layer."""

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        eca_positions: tuple[int, ...] = (),
        se_positions: tuple[int, ...] = (),
        cbam_positions: tuple[int, ...] = (),
    ):
        super().__init__()
        position_groups = {
            "eca": tuple(eca_positions),
            "se": tuple(se_positions),
            "cbam": tuple(cbam_positions),
        }
        all_positions = tuple(position for positions in position_groups.values() for position in positions)
        if len(all_positions) != len(set(all_positions)):
            raise ValueError("Stage-sparse attention positions must be disjoint")

        self.model = mobilenet_v2(weights=None, width_mult=width_mult)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        self.eca_positions = position_groups["eca"]
        self.se_positions = position_groups["se"]
        self.cbam_positions = position_groups["cbam"]
        self.attention_kinds: dict[int, str] = {}

        factories = {"eca": ECALayer, "se": SEBlock, "cbam": CBAM}
        for kind, positions in position_groups.items():
            for index in positions:
                module = self.model.features[index]
                channels = _output_channels(module)
                if channels is None:
                    raise ValueError(f"Cannot determine output channels for feature layer {index}")
                module.stage_attention = factories[kind](channels)
                self.attention_kinds[index] = kind

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for index, module in enumerate(self.model.features):
            outputs = module(outputs)
            if index in self.attention_kinds:
                outputs = module.stage_attention(outputs)
        outputs = outputs.mean((2, 3))
        return self.model.classifier(outputs)


class HybridAttentionMobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0,
                 se_positions: tuple[int, ...] = (), cbam_positions: tuple[int, ...] = (),
                 deep_activation: str = "relu"):
        super().__init__()
        self.model = mobilenet_v2(weights=None, width_mult=width_mult)
        self.se_positions = tuple(se_positions)
        self.cbam_positions = tuple(cbam_positions)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        for index, module in enumerate(self.model.features):
            if index not in self.se_positions and index not in self.cbam_positions:
                continue
            channels = _output_channels(module)
            if channels is None:
                raise ValueError(f"Cannot determine output channels for feature layer {index}")
            if index in self.se_positions:
                module.se = SEBlock(channels)
            if index in self.cbam_positions:
                module.cbam = CBAM(channels, deep_activation=deep_activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for index, module in enumerate(self.model.features):
            outputs = module(outputs)
            if index in self.se_positions:
                outputs = module.se(outputs)
            if index in self.cbam_positions:
                outputs = module.cbam(outputs)
        outputs = outputs.mean((2, 3))
        return self.model.classifier(outputs)


class CSGHAMobileNetV2(nn.Module):
    """MobileNetV2 with shallow SE and cross-stage guided CBAM modules."""

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        se_positions: tuple[int, ...] = (),
        cbam_positions: tuple[int, ...] = (),
        guidance_position: int = 2,
        guidance_reduction: int = 4,
        deep_activation: str = "relu",
        guidance_output_normalization: str = "none",
        guidance_scale_cap: float = 1.0,
    ):
        super().__init__()
        self.model = mobilenet_v2(weights=None, width_mult=width_mult)
        self.se_positions = tuple(se_positions)
        self.cbam_positions = tuple(cbam_positions)
        self.guidance_position = guidance_position
        self.guidance_reduction = guidance_reduction
        if guidance_position not in self.se_positions:
            raise ValueError("guidance_position must identify a shallow SE position")
        if any(position <= guidance_position for position in self.cbam_positions):
            raise ValueError("guided CBAM positions must follow guidance_position")
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

        guidance_module = self.model.features[guidance_position]
        guide_channels = _output_channels(guidance_module)
        if guide_channels is None:
            raise ValueError(f"Cannot determine output channels for guidance layer {guidance_position}")
        self.guide_channels = guide_channels
        self.guided_target_channels: dict[int, int] = {}

        for index, module in enumerate(self.model.features):
            if index in self.se_positions:
                channels = _output_channels(module)
                if channels is None:
                    raise ValueError(f"Cannot determine output channels for feature layer {index}")
                module.se = SEBlock(channels)
            if index in self.cbam_positions:
                channels = _output_channels(module)
                if channels is None:
                    raise ValueError(f"Cannot determine output channels for feature layer {index}")
                self.guided_target_channels[index] = channels
                module.guided_cbam = CrossStageGuidedCBAM(
                    channels,
                    guide_channels,
                    guidance_reduction=guidance_reduction,
                    deep_activation=deep_activation,
                    guidance_output_normalization=guidance_output_normalization,
                    guidance_scale_cap=guidance_scale_cap,
                )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        shallow_descriptor = None
        for index, module in enumerate(self.model.features):
            outputs = module(outputs)
            if index in self.se_positions:
                outputs = module.se(outputs)
            if index == self.guidance_position:
                shallow_descriptor = outputs.mean((2, 3))
            if index in self.cbam_positions:
                if shallow_descriptor is None:
                    raise RuntimeError("Shallow descriptor was not created before guided CBAM")
                outputs = module.guided_cbam(outputs, shallow_descriptor)
        outputs = outputs.mean((2, 3))
        return self.model.classifier(outputs)
