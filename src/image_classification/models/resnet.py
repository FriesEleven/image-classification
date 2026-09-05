"""CIFAR-stem ResNet-18 models for the P6 backbone-transfer study."""

from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.models import resnet18

from .mobilenetv2 import EarlyExitHead

RESNET18_BLOCK_CHANNELS = (64, 64, 128, 128, 256, 256, 512, 512)


def _cifar_resnet18(num_classes: int) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class BaseCIFARResNet18(nn.Module):
    """Standard ResNet-18 with a 3x3 stride-1 CIFAR stem and no max-pool."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = _cifar_resnet18(num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class MultiExitCIFARResNet18(nn.Module):
    """CIFAR ResNet-18 with exits at numbered residual-block boundaries.

    Blocks are numbered 0..7 in execution order. The P6 protocol uses block 2
    as the deployable exit and block 6 as a training-only auxiliary exit.
    """

    def __init__(self, num_classes: int = 10, exit_positions: tuple[int, ...] = (2, 6)):
        super().__init__()
        if not exit_positions:
            raise ValueError("At least one early-exit position is required")
        if tuple(sorted(set(exit_positions))) != tuple(exit_positions):
            raise ValueError("ResNet exit positions must be unique and increasing")
        if any(position < 0 or position > 6 for position in exit_positions):
            raise ValueError("ResNet exits must be residual-block boundaries 0 through 6")
        self.model = _cifar_resnet18(num_classes)
        self.exit_positions = tuple(exit_positions)
        self.exit_heads = nn.ModuleDict({
            str(position): EarlyExitHead(RESNET18_BLOCK_CHANNELS[position], num_classes)
            for position in self.exit_positions
        })

    def _stem(self, inputs: torch.Tensor) -> torch.Tensor:
        model = self.model
        return model.maxpool(model.relu(model.bn1(model.conv1(inputs))))

    def _blocks(self):
        return tuple(block for layer in (self.model.layer1, self.model.layer2,
                                         self.model.layer3, self.model.layer4) for block in layer)

    def _final_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.fc(torch.flatten(self.model.avgpool(features), 1))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self._stem(inputs)
        exit_logits = []
        for position, block in enumerate(self._blocks()):
            features = block(features)
            if position in self.exit_positions:
                exit_logits.append(self.exit_heads[str(position)](features))
        return (self._final_logits(features), *exit_logits)

    def forward_to_exit(self, inputs: torch.Tensor, exit_position: int | None) -> torch.Tensor:
        if exit_position is not None and exit_position not in self.exit_positions:
            raise ValueError(f"Unknown early-exit position: {exit_position}")
        features = self._stem(inputs)
        for position, block in enumerate(self._blocks()):
            features = block(features)
            if position == exit_position:
                return self.exit_heads[str(position)](features)
        return self._final_logits(features)

    def forward_with_policy(
        self,
        inputs: torch.Tensor,
        confidence_threshold: float,
        exit_position: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            raise RuntimeError("Dynamic early-exit routing is inference-only")
        if inputs.ndim != 4 or inputs.shape[0] == 0:
            raise ValueError("Dynamic early-exit routing requires a non-empty image batch")
        if exit_position not in self.exit_positions:
            raise ValueError(f"Unknown early-exit position: {exit_position}")
        if not math.isfinite(confidence_threshold):
            raise ValueError("confidence_threshold must be finite")

        features = self._stem(inputs)
        blocks = self._blocks()
        for position, block in enumerate(blocks):
            features = block(features)
            if position != exit_position:
                continue
            exit_logits = self.exit_heads[str(position)](features)
            confidence = torch.softmax(exit_logits.float(), dim=1).amax(dim=1)
            early = confidence >= confidence_threshold
            routed_logits = torch.empty_like(exit_logits)
            paths = torch.ones(inputs.shape[0], dtype=torch.long, device=inputs.device)
            routed_logits[early] = exit_logits[early]
            paths[early] = 0
            unresolved = ~early
            if bool(unresolved.any()):
                remaining = features[unresolved]
                for later_block in blocks[position + 1:]:
                    remaining = later_block(remaining)
                routed_logits[unresolved] = self._final_logits(remaining)
            return routed_logits, paths
        raise RuntimeError("Configured early exit was not reached")
