"""Strict loading of the RegionBrainAge inference-only checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .model import RegionalBrainAgeModel


FORMAT_VERSION = "regionbrainage-inference-v1"
EXPECTED_CONFIG: dict[str, object] = {
    "backbone": "densenet121",
    "roi_dim": 256,
    "region_emb_dim": 32,
    "atlas_emb_dim": 8,
    "dropout": 0.1,
    "ho_thr": 25,
    "ho_symmetric_split": True,
    "buckner_space": "MNI",
}
EXPECTED_SOURCE_COUNTS = {
    "cortex_schaefer200": 200,
    "subcortex_harvardoxford": 15,
    "cerebellum_buckner7": 7,
}
LOOKUP_COLUMNS = ("label", "source", "source_label", "name", "full_name")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lookup(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    missing = set(LOOKUP_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Checkpoint ROI lookup is missing columns: {sorted(missing)}")
    lookup = frame.loc[:, LOOKUP_COLUMNS].copy()
    lookup["label"] = lookup["label"].astype(int)
    lookup["source_label"] = lookup["source_label"].astype(int)
    for column in ("source", "name", "full_name"):
        lookup[column] = lookup[column].astype(str)
    lookup = lookup.sort_values("label").reset_index(drop=True)
    if lookup["label"].tolist() != list(range(1, 223)):
        raise ValueError("Checkpoint ROI labels must be exactly 1 through 222.")
    if lookup["source"].value_counts().to_dict() != EXPECTED_SOURCE_COUNTS:
        raise ValueError("Checkpoint atlas-family composition is not 200/15/7 as expected.")
    if not (
        lookup["full_name"] == lookup["source"] + "::" + lookup["name"]
    ).all():
        raise ValueError("Checkpoint ROI full_name values are inconsistent.")
    for source, count in EXPECTED_SOURCE_COUNTS.items():
        observed = lookup.loc[lookup["source"] == source, "source_label"].tolist()
        if observed != list(range(1, count + 1)):
            raise ValueError(f"Unexpected source labels for {source}.")
    return lookup


def safe_load(path: str | Path) -> tuple[dict[str, Any], pd.DataFrame, str]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary.")
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint.get('format_version')!r}; "
            f"expected {FORMAT_VERSION!r}."
        )
    if checkpoint.get("checkpoint_endpoint") != "best_total":
        raise ValueError("The public predictor requires the selected best_total endpoint.")
    state = checkpoint.get("model_state_dict")
    config = checkpoint.get("config")
    records = checkpoint.get("lookup_records")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model_state_dict must be a non-empty dictionary.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Checkpoint model_state_dict must contain only named tensors.")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary.")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Checkpoint config {key!r} is {config.get(key)!r}; expected {expected!r}."
            )
    if not isinstance(records, list):
        raise ValueError("Checkpoint lookup_records must be a list.")
    lookup = validate_lookup(records)
    return checkpoint, lookup, sha256_file(checkpoint_path)


def load_model(
    path: str | Path,
    device: torch.device,
) -> tuple[RegionalBrainAgeModel, pd.DataFrame, dict[str, Any]]:
    checkpoint, lookup, checkpoint_sha256 = safe_load(path)
    config = checkpoint["config"]
    model = RegionalBrainAgeModel(
        backbone_name=str(config["backbone"]),
        lookup_df=lookup,
        in_channels=1,
        roi_dim=int(config["roi_dim"]),
        region_emb_dim=int(config["region_emb_dim"]),
        atlas_emb_dim=int(config["atlas_emb_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = {
        "format_version": checkpoint["format_version"],
        "release_version": checkpoint.get("release_version"),
        "model_name": checkpoint.get("model_name"),
        "checkpoint_endpoint": checkpoint["checkpoint_endpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "config": {key: config[key] for key in EXPECTED_CONFIG},
        "provenance": checkpoint.get("provenance", {}),
        "intended_use": checkpoint.get("intended_use"),
    }
    return model, lookup, metadata
