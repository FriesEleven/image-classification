"""MobileNetV2 variants used by the experiments."""

import torch
from torch import nn
from torchvision.models import mobilenet_v2

from .attention import CBAM, SEBlock
from .eca import ECAMobileNetV2


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


class HybridAttentionMobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0,
                 se_positions: tuple[int, ...] = (), cbam_positions: tuple[int, ...] = ()):
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
                module.cbam = CBAM(channels)

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
