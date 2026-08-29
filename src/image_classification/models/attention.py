"""Reusable attention blocks."""

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(inputs)) + self.fc(self.max_pool(inputs)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = torch.mean(inputs, dim=1, keepdim=True)
        maximum, _ = torch.max(inputs, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.channel_attention(inputs) * inputs
        return self.spatial_attention(outputs) * outputs


class CrossStageChannelAttention(nn.Module):
    """CBAM channel attention augmented by a projected shallow descriptor."""

    def __init__(
        self,
        channels: int,
        guide_channels: int,
        ratio: int = 16,
        guidance_reduction: int = 4,
    ):
        super().__init__()
        hidden = max(1, channels // ratio)
        guidance_hidden = max(1, guide_channels // guidance_reduction)
        self.channels = channels
        self.guide_channels = guide_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.guide_projection = nn.Sequential(
            nn.Linear(guide_channels, guidance_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(guidance_hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, shallow_descriptor: torch.Tensor) -> torch.Tensor:
        expected_shape = (inputs.shape[0], self.guide_channels)
        if tuple(shallow_descriptor.shape) != expected_shape:
            raise ValueError(
                f"Expected shallow descriptor shape {expected_shape}, "
                f"got {tuple(shallow_descriptor.shape)}"
            )
        batch = inputs.shape[0]
        guidance = self.guide_projection(shallow_descriptor).view(batch, self.channels, 1, 1)
        deep_attention = self.fc(self.avg_pool(inputs)) + self.fc(self.max_pool(inputs))
        return self.sigmoid(deep_attention + guidance)


class CrossStageGuidedCBAM(nn.Module):
    """CBAM whose channel gate receives a compact shallow-stage descriptor."""

    def __init__(self, channels: int, guide_channels: int, guidance_reduction: int = 4):
        super().__init__()
        self.channel_attention = CrossStageChannelAttention(
            channels,
            guide_channels,
            guidance_reduction=guidance_reduction,
        )
        self.spatial_attention = SpatialAttention()

    def forward(self, inputs: torch.Tensor, shallow_descriptor: torch.Tensor) -> torch.Tensor:
        outputs = self.channel_attention(inputs, shallow_descriptor) * inputs
        return self.spatial_attention(outputs) * outputs


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = inputs.size()
        weights = self.avg_pool(inputs).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return inputs * weights
