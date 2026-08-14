from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
import torch.nn as nn

from .densenet121 import build_backbone
from .roi_pool import MultiScaleROIPool


class GlobalHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class RegionalHead(nn.Module):
    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(token_dim // 2, 64)
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.age_head = nn.Linear(hidden_dim, 1)

    def forward(self, roi_tokens: torch.Tensor) -> torch.Tensor:
        return self.age_head(self.feature_mlp(roi_tokens)).squeeze(-1)


class AttentionROIAggregator(nn.Module):
    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, 1),
        )
        self.global_head = GlobalHead(token_dim, hidden_dim=token_dim, dropout=dropout)

    def forward(
        self,
        roi_tokens: torch.Tensor,
        roi_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_mlp(roi_tokens).squeeze(-1)
        scores = scores.masked_fill(~roi_valid_mask, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=1)
        attention_weights = attention_weights.masked_fill(~roi_valid_mask, 0.0)
        zero_valid = roi_valid_mask.sum(dim=1, keepdim=True) == 0
        if zero_valid.any():
            attention_weights = torch.where(
                zero_valid,
                torch.zeros_like(attention_weights),
                attention_weights,
            )
        pooled = torch.sum(roi_tokens * attention_weights.unsqueeze(-1), dim=1)
        return self.global_head(pooled), roi_tokens


class RegionalBrainAgeModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        lookup_df: pd.DataFrame,
        in_channels: int = 1,
        roi_dim: int = 256,
        region_emb_dim: int = 32,
        atlas_emb_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_rois, atlas_family_ids, atlas_family_lookup = self._parse_lookup_df(
            lookup_df
        )
        self.atlas_family_lookup = atlas_family_lookup
        self.backbone = build_backbone(backbone_name=backbone_name, in_channels=in_channels)
        self.roi_pool = MultiScaleROIPool(
            num_rois=self.num_rois,
            roi_dim=roi_dim,
            feature_channels=self.backbone.feature_channels,
            dropout=dropout,
        )
        self.region_embedding = nn.Embedding(self.num_rois + 1, region_emb_dim)
        self.atlas_embedding = nn.Embedding(len(atlas_family_lookup), atlas_emb_dim)
        self.region_embedding_proj = nn.Linear(region_emb_dim, roi_dim)
        self.atlas_embedding_proj = nn.Linear(atlas_emb_dim, roi_dim)
        self.token_norm = nn.LayerNorm(roi_dim)
        self.token_dropout = nn.Dropout(dropout)
        deep_channels = self.backbone.global_feature_channels["global_deep_feature"]
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.direct_global_head = GlobalHead(
            deep_channels, hidden_dim=deep_channels, dropout=dropout
        )
        self.regional_head = RegionalHead(token_dim=roi_dim, dropout=dropout)
        self.roi_global_aggregator = AttentionROIAggregator(
            token_dim=roi_dim, dropout=dropout
        )
        self.register_buffer(
            "region_ids",
            torch.arange(1, self.num_rois + 1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "atlas_family_ids", atlas_family_ids, persistent=False
        )

    @staticmethod
    def _parse_lookup_df(
        lookup_df: pd.DataFrame,
    ) -> tuple[int, torch.Tensor, Dict[str, int]]:
        required = {"label", "source"}
        missing = required.difference(lookup_df.columns)
        if missing:
            raise ValueError(f"ROI lookup is missing columns: {sorted(missing)}")
        lookup = lookup_df.copy()
        lookup["label"] = lookup["label"].astype(int)
        lookup["source"] = lookup["source"].astype(str)
        lookup = lookup[lookup["label"] > 0].sort_values("label").reset_index(drop=True)
        labels = lookup["label"].tolist()
        if labels != list(range(1, len(labels) + 1)):
            raise ValueError("ROI labels must be consecutive positive integers starting at 1.")
        unique_sources = sorted(lookup["source"].unique().tolist())
        family_lookup = {name: index for index, name in enumerate(unique_sources)}
        family_ids = torch.tensor(
            [family_lookup[name] for name in lookup["source"].tolist()],
            dtype=torch.long,
        )
        return len(labels), family_ids, family_lookup

    def _augment_roi_tokens(self, roi_tokens: torch.Tensor) -> torch.Tensor:
        region_bias = self.region_embedding_proj(self.region_embedding(self.region_ids))
        atlas_bias = self.atlas_embedding_proj(self.atlas_embedding(self.atlas_family_ids))
        return self.token_dropout(
            self.token_norm(roi_tokens + region_bias.unsqueeze(0) + atlas_bias.unsqueeze(0))
        )

    def forward(self, x: torch.Tensor, roi_labels: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        deep_vector = self.global_pool(features["global_deep_feature"]).flatten(1)
        global_age_direct = self.direct_global_head(deep_vector)
        roi_features = {
            name: features[name]
            for name in ("roi_low_feature", "roi_mid_feature", "roi_high_feature")
        }
        roi_tokens, roi_valid_mask = self.roi_pool(roi_features, roi_labels)
        roi_tokens = self._augment_roi_tokens(roi_tokens)
        roi_tokens = roi_tokens * roi_valid_mask.unsqueeze(-1).to(roi_tokens.dtype)
        regional_age = self.regional_head(roi_tokens)
        regional_age = regional_age.masked_fill(~roi_valid_mask, 0.0)
        global_age_roi, _ = self.roi_global_aggregator(roi_tokens, roi_valid_mask)
        return {
            "global_age_direct": global_age_direct,
            "global_age_roi": global_age_roi,
            "regional_age": regional_age,
            "roi_valid_mask": roi_valid_mask,
        }
