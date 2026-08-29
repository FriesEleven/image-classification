"""ECA-MobileNetV2 adapted from the vendored ECA-Net implementation.

The original implementation and license are kept under ``third_party/eca_net``.
"""

import torch
from torch import nn


class ECALayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(inputs)
        weights = self.conv(weights.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return inputs * self.sigmoid(weights).expand_as(inputs)


class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True),
        )


class ECAInvertedResidual(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int, expand_ratio: int, kernel_size: int):
        super().__init__()
        hidden = int(round(input_channels * expand_ratio))
        self.use_residual = stride == 1 and input_channels == output_channels
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(input_channels, hidden, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden, hidden, stride=stride, groups=hidden),
                nn.Conv2d(hidden, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                ECALayer(output_channels, kernel_size),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        return inputs + outputs if self.use_residual else outputs


class ECAMobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        input_channels = int(32 * width_mult)
        self.last_channel = int(1280 * max(1.0, width_mult))
        settings = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
                    (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))
        features: list[nn.Module] = [ConvBNReLU(3, input_channels, stride=2)]
        for expand_ratio, channels, repeats, first_stride in settings:
            output_channels = int(channels * width_mult)
            for index in range(repeats):
                stride = first_stride if index == 0 else 1
                kernel_size = 1 if channels < 96 else 3
                features.append(ECAInvertedResidual(input_channels, output_channels, stride, expand_ratio, kernel_size))
                input_channels = output_channels
        features.append(ConvBNReLU(input_channels, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.last_channel, num_classes))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = outputs.mean((-1, -2))
        return self.classifier(outputs)
