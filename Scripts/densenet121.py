from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


FEATURE_CHANNELS: Dict[str, int] = {
    "roi_low_feature": 128,
    "roi_mid_feature": 256,
    "roi_high_feature": 512,
}
GLOBAL_FEATURE_CHANNELS: Dict[str, int] = {"global_deep_feature": 512}


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class FeatureAdaptor3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj: nn.Module
        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DenseLayer3D(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = _make_group_norm(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            num_input_features,
            inter_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.norm2 = _make_group_norm(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            inter_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(
                new_features,
                p=self.drop_rate,
                training=self.training,
            )
        return torch.cat([x, new_features], dim=1)


class DenseBlock3D(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        layers = []
        channels = num_input_features
        for _ in range(num_layers):
            layer = DenseLayer3D(channels, growth_rate, bn_size, drop_rate)
            layers.append(layer)
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Transition3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            _make_group_norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )


class DenseNet3DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            _make_group_norm(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        self.block1 = DenseBlock3D(
            block_config[0], num_features, bn_size, growth_rate, drop_rate
        )
        num_features = self.block1.out_channels
        self.trans1 = Transition3D(num_features, num_features // 2)
        num_features //= 2

        self.block2 = DenseBlock3D(
            block_config[1], num_features, bn_size, growth_rate, drop_rate
        )
        num_features = self.block2.out_channels
        self.trans2 = Transition3D(num_features, num_features // 2)
        num_features //= 2

        self.block3 = DenseBlock3D(
            block_config[2], num_features, bn_size, growth_rate, drop_rate
        )
        num_features = self.block3.out_channels
        self.trans3 = Transition3D(num_features, num_features // 2)
        num_features //= 2

        self.block4 = DenseBlock3D(
            block_config[3], num_features, bn_size, growth_rate, drop_rate
        )
        self.norm5 = _make_group_norm(self.block4.out_channels)
        self.relu5 = nn.ReLU(inplace=True)

        self.roi_low_feature_adaptor = FeatureAdaptor3D(
            self.block1.out_channels, FEATURE_CHANNELS["roi_low_feature"]
        )
        self.roi_mid_feature_adaptor = FeatureAdaptor3D(
            self.block2.out_channels, FEATURE_CHANNELS["roi_mid_feature"]
        )
        self.roi_high_feature_adaptor = FeatureAdaptor3D(
            self.block3.out_channels, FEATURE_CHANNELS["roi_high_feature"]
        )
        self.global_deep_feature_adaptor = FeatureAdaptor3D(
            self.block4.out_channels,
            GLOBAL_FEATURE_CHANNELS["global_deep_feature"],
        )
        self.feature_channels = dict(FEATURE_CHANNELS)
        self.global_feature_channels = dict(GLOBAL_FEATURE_CHANNELS)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        roi_low_feature = self.block1(x)
        x = self.trans1(roi_low_feature)
        roi_mid_feature = self.block2(x)
        x = self.trans2(roi_mid_feature)
        roi_high_feature = self.block3(x)
        x = self.trans3(roi_high_feature)
        global_deep_feature = self.relu5(self.norm5(self.block4(x)))
        return {
            "roi_low_feature": self.roi_low_feature_adaptor(roi_low_feature),
            "roi_mid_feature": self.roi_mid_feature_adaptor(roi_mid_feature),
            "roi_high_feature": self.roi_high_feature_adaptor(roi_high_feature),
            "global_deep_feature": self.global_deep_feature_adaptor(global_deep_feature),
        }


def build_backbone(backbone_name: str, in_channels: int = 1) -> nn.Module:
    """Build the only backbone supported by this released checkpoint."""
    if backbone_name.lower() != "densenet121":
        raise ValueError(
            f"Unsupported released backbone {backbone_name!r}; expected 'densenet121'."
        )
    return DenseNet3DBackbone(in_channels=in_channels)
