"""Multi-scale atlas-guided ROI pooling used by RegionBrainAge."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleROIPool(nn.Module):
    def __init__(
        self,
        num_rois: int,
        roi_dim: int,
        feature_channels: Mapping[str, int],
        dropout: float = 0.1,
        scales: Sequence[str] = (
            "roi_low_feature",
            "roi_mid_feature",
            "roi_high_feature",
        ),
    ) -> None:
        super().__init__()
        if num_rois <= 0:
            raise ValueError("num_rois must be positive.")
        self.num_rois = int(num_rois)
        self.roi_dim = int(roi_dim)
        self.scales = tuple(scales)
        missing = [name for name in self.scales if name not in feature_channels]
        if missing:
            raise ValueError(f"feature_channels is missing required scales: {missing}")
        self.scale_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(feature_channels[name]),
                    nn.Linear(feature_channels[name], roi_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name in self.scales
            }
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(roi_dim * len(self.scales)),
            nn.Linear(roi_dim * len(self.scales), roi_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(roi_dim, roi_dim),
        )

    def _resize_labels(
        self,
        roi_labels: torch.Tensor,
        spatial_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        if roi_labels.ndim != 4:
            raise ValueError(
                f"roi_labels must have shape [B, D, H, W], got {tuple(roi_labels.shape)}."
            )
        if torch.is_floating_point(roi_labels):
            rounded = roi_labels.round()
            if not torch.allclose(roi_labels, rounded):
                raise ValueError("roi_labels must contain integer-valued ROI ids.")
            roi_labels = rounded.long()
        else:
            roi_labels = roi_labels.long()
        minimum = int(roi_labels.min().item())
        maximum = int(roi_labels.max().item())
        if minimum < 0 or maximum > self.num_rois:
            raise ValueError(
                f"roi_labels contain ids outside [0, {self.num_rois}]: "
                f"observed min={minimum}, max={maximum}."
            )
        resized = F.interpolate(
            roi_labels.unsqueeze(1).float(),
            size=spatial_size,
            mode="nearest",
        ).squeeze(1)
        rounded = resized.round()
        if not torch.allclose(resized, rounded):
            raise ValueError("Nearest-neighbor label resize produced non-integer values.")
        return rounded.long()

    def _pool_single_scale(
        self,
        feature_map: torch.Tensor,
        roi_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_map.ndim != 5:
            raise ValueError(
                f"feature_map must have shape [B, C, D, H, W], got {tuple(feature_map.shape)}."
            )
        batch_size, channels, depth, height, width = feature_map.shape
        resized_labels = self._resize_labels(roi_labels, (depth, height, width))
        flattened_features = feature_map.reshape(batch_size, channels, -1).transpose(1, 2)
        flattened_labels = resized_labels.reshape(batch_size, -1)
        batch_offsets = (
            torch.arange(batch_size, device=feature_map.device, dtype=torch.long).unsqueeze(1)
            * (self.num_rois + 1)
        )
        scatter_index = (flattened_labels + batch_offsets).reshape(-1)
        feature_values = flattened_features.reshape(-1, channels).float()
        pooled_sum = torch.zeros(
            batch_size * (self.num_rois + 1),
            channels,
            device=feature_map.device,
            dtype=torch.float32,
        )
        pooled_sum.scatter_add_(
            0,
            scatter_index.unsqueeze(1).expand(-1, channels),
            feature_values,
        )
        pooled_count = torch.zeros(
            batch_size * (self.num_rois + 1),
            1,
            device=feature_map.device,
            dtype=torch.float32,
        )
        pooled_count.scatter_add_(
            0,
            scatter_index.unsqueeze(1),
            torch.ones(
                scatter_index.numel(),
                1,
                device=feature_map.device,
                dtype=torch.float32,
            ),
        )
        pooled_sum = pooled_sum.view(batch_size, self.num_rois + 1, channels)[:, 1:, :]
        pooled_count = pooled_count.view(batch_size, self.num_rois + 1, 1)[:, 1:, :]
        roi_valid_mask = pooled_count.squeeze(-1) > 0
        pooled_mean = pooled_sum / pooled_count.clamp_min(1.0)
        pooled_mean = pooled_mean * roi_valid_mask.unsqueeze(-1).to(pooled_mean.dtype)
        return pooled_mean.to(dtype=feature_map.dtype), roi_valid_mask

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        roi_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [name for name in self.scales if name not in features]
        if missing:
            raise ValueError(f"features is missing required scales: {missing}")
        projected_tokens = []
        valid_masks = []
        for name in self.scales:
            pooled, valid_mask = self._pool_single_scale(features[name], roi_labels)
            projected = self.scale_projectors[name](pooled)
            projected = projected * valid_mask.unsqueeze(-1).to(projected.dtype)
            projected_tokens.append(projected)
            valid_masks.append(valid_mask)
        roi_valid_mask = torch.stack(valid_masks, dim=0).any(dim=0)
        fused_tokens = self.fusion(torch.cat(projected_tokens, dim=-1))
        fused_tokens = fused_tokens * roi_valid_mask.unsqueeze(-1).to(fused_tokens.dtype)
        return fused_tokens, roi_valid_mask
