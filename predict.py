#!/usr/bin/env python3
"""Predict global and ROI-conditioned brain age from T1-weighted MRI.

Inputs may be either one NIfTI file or a directory of NIfTI files. The input
type must be declared explicitly:

* ``raw`` runs the release preprocessing pipeline (reorientation, SynthStrip,
  and 12-degree-of-freedom FLIRT registration to MNI152 1 mm) before inference.
* ``preprocessed`` requires a brain-extracted image already on the exact
  MNI152 1-mm grid used for training.

The bundled template-grid 222-ROI map is masked by each subject's nonzero brain
mask, reproducing the label construction used by the training data loader
without downloading or regenerating atlases at prediction time.

All predictions are raw, uncorrected ages in years. Optional validation-fitted
age-bias correction is a separate post-processing step in ``correct_age.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import torch


RELEASE_ROOT = Path(__file__).resolve().parent
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

import preprocessing  # noqa: E402
from scripts.checkpoint import load_model, sha256_file, validate_lookup  # noqa: E402


DEFAULT_MODEL = RELEASE_ROOT / "model" / "densenet121_w0p1.pt"
DEFAULT_ATLAS = RELEASE_ROOT / "resources" / "222ROI_MNI152_1mm.nii.gz"
DEFAULT_TEMPLATE = RELEASE_ROOT / "resources" / "MNI152_T1_1mm_brain.nii.gz"
PACKAGED_LOOKUP = RELEASE_ROOT / "resources" / "roi_lookup.csv"
EXPECTED_MODEL_SHA256 = "40e604c23414fbb40c695def66f7c712c76021e155f5e2d467c80be8e5ab4be2"
EXPECTED_ATLAS_SHA256 = "2f83baf0421b39cf7c8c358dc32e4f65c4473e8b42df47403541482eaac0d2e2"
EXPECTED_TEMPLATE_SHA256 = "32d5be33460f995a5d305507053c8862c823d9ca6bfb543381308df14590f212"
EXPECTED_MNI_SHAPE = (182, 218, 182)
EXPECTED_MNI_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Run RegionBrainAge on one T1 NIfTI or a folder. "
            "Declare whether inputs are raw or already preprocessed."
        )
    )
    command.add_argument(
        "--input",
        required=True,
        help=(
            "One .nii/.nii.gz file or a directory containing T1 NIfTI files. "
            "Directories are searched recursively and must contain only the intended T1 inputs."
        ),
    )
    command.add_argument(
        "--input-type",
        required=True,
        choices=("raw", "preprocessed"),
        help=(
            "raw: run SynthStrip and affine MNI registration; preprocessed: "
            "require a brain-extracted MNI152 1-mm image."
        ),
    )
    command.add_argument("--output-dir", required=True, help="A new or empty output directory.")
    command.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help=(
            "Inference device (default: auto). Auto tests CUDA before use and "
            "falls back to CPU when the installed PyTorch build cannot run on the GPU."
        ),
    )
    return command


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def discover_inputs(path_value: str) -> tuple[Path, list[Path]]:
    source = Path(path_value).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if source.is_file():
        if not is_nifti(source):
            raise ValueError(f"Input is not a NIfTI file: {source}")
        return source, [source]
    if not source.is_dir():
        raise ValueError(f"Input is neither a file nor a directory: {source}")
    files = sorted(
        item.resolve()
        for item in source.rglob("*.nii*")
        if item.is_file() and is_nifti(item)
    )
    if not files:
        raise ValueError(f"No NIfTI inputs were found recursively in {source}")
    canonical_paths = [str(item) for item in files]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError(
            "The directory selection resolves the same NIfTI more than once "
            "(for example through duplicate symbolic links)."
        )
    return source, files


def safe_subject_ids(files: list[Path], input_root: Path) -> list[str]:
    names = preprocessing.unique_output_names(files, input_path=input_root, mode="auto")
    sanitized = [re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-") for name in names]
    if any(not name for name in sanitized):
        raise ValueError("At least one input filename could not form a safe subject identifier.")
    if len(sanitized) != len(set(sanitized)):
        raise ValueError("Input filenames collide after subject-ID normalization.")
    return sanitized


def cuda_device_details(device: torch.device) -> dict[str, Any]:
    """Return JSON-safe CUDA details without assuming the GPU is executable."""
    details: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    try:
        details["gpu_name"] = torch.cuda.get_device_name(device)
    except Exception as exc:
        details["gpu_name_error"] = f"{type(exc).__name__}: {exc}"
    try:
        major, minor = torch.cuda.get_device_capability(device)
        details["gpu_compute_capability"] = f"{major}.{minor}"
    except Exception as exc:
        details["gpu_compute_capability_error"] = f"{type(exc).__name__}: {exc}"
    return details


def cuda_execution_error(device: torch.device) -> str | None:
    """Run and synchronize a real CUDA kernel, returning its error if unusable."""
    try:
        with torch.inference_mode():
            probe = torch.ones(16, device=device)
            result = (probe + 1.0).sum()
            torch.cuda.synchronize(device)
            if float(result.item()) != 32.0:
                raise RuntimeError("CUDA execution probe returned an unexpected result.")
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {' '.join(str(exc).splitlines())}"


def resolve_device(value: str) -> tuple[torch.device, dict[str, Any]]:
    """Resolve the device and verify CUDA can execute before loading the model."""
    requested = value.lower()
    selection: dict[str, Any] = {
        "requested": requested,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if requested == "cpu":
        selection.update({"resolved": "cpu", "reason": "explicit_cpu"})
        return torch.device("cpu"), selection
    if requested not in {"auto", "cuda"}:
        raise ValueError(f"Invalid torch device {value!r}.")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access a CUDA device. "
                "Use --device cpu or install a PyTorch build compatible with this system."
            )
        selection.update({"resolved": "cpu", "reason": "cuda_unavailable"})
        return torch.device("cpu"), selection

    cuda_device = torch.device("cuda")
    selection.update(cuda_device_details(cuda_device))
    probe_error = cuda_execution_error(cuda_device)
    if probe_error is None:
        selection.update({"resolved": "cuda", "reason": "cuda_probe_passed"})
        return cuda_device, selection

    selection["cuda_probe_error"] = probe_error
    gpu_description = selection.get("gpu_name", "the detected GPU")
    compatibility_message = (
        f"The installed PyTorch build cannot execute CUDA kernels on {gpu_description} "
        f"({probe_error}). Install a PyTorch build compatible with this GPU, or use CPU."
    )
    if requested == "cuda":
        raise RuntimeError(compatibility_message)

    selection.update({"resolved": "cpu", "reason": "cuda_probe_failed_cpu_fallback"})
    print(f"RegionBrainAge: {compatibility_message} Falling back to CPU.", file=sys.stderr)
    return torch.device("cpu"), selection


def singleton_three_dimensional(data: np.ndarray, name: str) -> np.ndarray:
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{name} must be 3-D or singleton-4-D; observed shape {data.shape}.")
    return data


def validate_mni_geometry(image: nib.spatialimages.SpatialImage, name: str) -> None:
    if tuple(image.shape[:3]) != EXPECTED_MNI_SHAPE:
        raise ValueError(
            f"{name} must have MNI152 1-mm shape {EXPECTED_MNI_SHAPE}; "
            f"observed {tuple(image.shape[:3])}."
        )
    if not np.allclose(image.affine, EXPECTED_MNI_AFFINE, rtol=0.0, atol=1e-3):
        raise ValueError(f"{name} affine does not match the training MNI152 1-mm grid.")


def load_atlas(path: Path) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Composite atlas not found: {path}. The model cannot predict without ROI labels."
        )
    observed_sha256 = sha256_file(path)
    if observed_sha256 != EXPECTED_ATLAS_SHA256:
        raise RuntimeError(
            "The composite atlas checksum does not match the atlas used by this release. "
            "A label map with the same numeric range but different ROI semantics is unsafe."
        )
    atlas_nifti = nib.load(str(path))
    validate_mni_geometry(atlas_nifti, "Composite atlas")
    raw = singleton_three_dimensional(np.asanyarray(atlas_nifti.dataobj), "Composite atlas")
    as_float = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(as_float).all() or not np.array_equal(as_float, np.rint(as_float)):
        raise ValueError("Composite atlas must contain finite integer labels.")
    atlas = np.rint(as_float).astype(np.int16)
    unique = np.unique(atlas)
    if not np.array_equal(unique, np.arange(223, dtype=unique.dtype)):
        raise ValueError("Composite atlas must contain every label from 0 through 222.")
    return atlas, observed_sha256


def load_and_normalize_image(
    path: Path,
    unmasked_atlas: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    image_nifti = nib.load(str(path))
    validate_mni_geometry(image_nifti, "T1 image")
    image = singleton_three_dimensional(
        np.asanyarray(image_nifti.get_fdata(dtype=np.float32)), "T1 image"
    ).astype(np.float32, copy=False)
    if not np.isfinite(image).all():
        raise ValueError("T1 image contains non-finite voxels.")
    brain_mask = image != 0
    brain_voxels = int(brain_mask.sum())
    if brain_voxels == 0:
        raise ValueError("T1 image contains no nonzero brain voxels.")
    values = image[brain_mask]
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-6:
        raise ValueError("Within-brain T1 intensities have invalid or near-zero variance.")
    normalized = np.zeros_like(image, dtype=np.float32)
    normalized[brain_mask] = (values - mean) / std
    labels = unmasked_atlas.copy()
    labels[~brain_mask] = 0
    present = np.unique(labels)
    present_positive = present[present > 0]
    metadata = {
        "brain_voxels": brain_voxels,
        "brain_fraction": float(brain_mask.mean()),
        "intensity_mean_before_normalization": mean,
        "intensity_std_before_normalization": std,
        "num_full_resolution_rois_present": int(len(present_positive)),
        "num_full_resolution_rois_missing": int(222 - len(present_positive)),
    }
    image_tensor = torch.from_numpy(normalized[None, None, ...])
    label_tensor = torch.from_numpy(labels.astype(np.int64, copy=False)[None, ...])
    return image_tensor, label_tensor, metadata


def prepare_raw_tools(synthstrip_argument: str | None) -> tuple[Path, str, str, str]:
    template = DEFAULT_TEMPLATE
    if not template.is_file():
        raise RuntimeError(f"Bundled MNI152 registration template is missing: {template}")
    if sha256_file(template) != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("The bundled MNI152 registration template checksum is incorrect.")
    validate_mni_geometry(nib.load(str(template)), "MNI152 registration template")
    reorient = preprocessing.find_fsl_cmd("fslreorient2std")
    flirt = preprocessing.find_fsl_cmd("flirt")
    synthstrip = preprocessing.find_synthstrip_cmd(synthstrip_argument)
    slicer = preprocessing.find_fsl_cmd("slicer")
    pngappend = preprocessing.find_fsl_cmd("pngappend")
    missing = [
        name
        for name, value in (
            ("fslreorient2std", reorient),
            ("flirt", flirt),
            ("mri_synthstrip", synthstrip),
            ("slicer", slicer),
            ("pngappend", pngappend),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError("Raw mode is missing required commands: " + ", ".join(missing))
    return template, str(reorient), str(synthstrip), str(flirt)


def raw_to_preprocessed(
    input_path: Path,
    subject_id: str,
    output_root: Path,
    tools: tuple[Path, str, str, str],
) -> tuple[Path | None, dict[str, Any]]:
    template, reorient, synthstrip, flirt = tools
    record = preprocessing.preprocess_one(
        in_t1=input_path,
        out_root=output_root,
        template=template,
        reorient_cmd=reorient,
        synthstrip_cmd=synthstrip,
        flirt_cmd=flirt,
        overwrite=False,
        output_name=subject_id,
    )
    if record.get("status") != "ok":
        return None, record
    subject_directory = Path(str(record["output_dir"])).resolve()
    result = subject_directory / f"{subject_directory.name}_brain_mni152_1mm.nii.gz"
    if not result.is_file():
        raise FileNotFoundError(f"Preprocessing reported success but output is missing: {result}")
    return result, record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def initialize_csv(path: Path, columns: list[str]) -> None:
    """Create a header-only CSV and make the header durable on disk."""
    with path.open("x", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def append_csv_row(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    """Append one completed result and flush it so earlier work survives interruption."""
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"CSV row for {path.name} is missing columns: {missing}")
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def verify_packaged_lookup(checkpoint_lookup: pd.DataFrame) -> None:
    if not PACKAGED_LOOKUP.is_file():
        raise FileNotFoundError(f"Packaged ROI lookup is missing: {PACKAGED_LOOKUP}")
    packaged = validate_lookup(pd.read_csv(PACKAGED_LOOKUP).to_dict("records"))
    if not packaged.equals(checkpoint_lookup.reset_index(drop=True)):
        raise ValueError("Packaged ROI lookup does not match the checkpoint lookup.")


def run(arguments: argparse.Namespace) -> int:
    input_root, files = discover_inputs(arguments.input)
    subject_ids = safe_subject_ids(files, input_root=input_root)
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model_path = DEFAULT_MODEL.resolve()
    atlas_path = DEFAULT_ATLAS.resolve()
    observed_model_hash = sha256_file(model_path)
    if observed_model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("The bundled model checksum is incorrect.")
    device, device_selection = resolve_device(arguments.device)
    print(f"Inference device: {device}", flush=True)
    model, lookup, model_metadata = load_model(model_path, device=device)
    verify_packaged_lookup(lookup)
    unmasked_atlas, atlas_sha256 = load_atlas(atlas_path)
    raw_tools = (
        prepare_raw_tools(None)
        if arguments.input_type == "raw"
        else None
    )
    preprocessing_metadata = None
    if raw_tools is not None:
        template, reorient, synthstrip, flirt = raw_tools
        preprocessing_metadata = {
            "pipeline": [
                "FSL fslreorient2std",
                "FreeSurfer SynthStrip",
                "FSL FLIRT affine registration with 12 degrees of freedom",
            ],
            "mni_template_sha256": sha256_file(template),
            "commands": {
                "fslreorient2std": Path(reorient).name,
                "synthstrip": Path(synthstrip).name,
                "flirt": Path(flirt).name,
            },
            "registration_qc": "Generated for visual review; not automatically accepted.",
        }

    input_records = []
    for subject_id, path in zip(subject_ids, files):
        input_records.append(
            {
                "subject_id": subject_id,
                "filename": path.name,
                "input_path": path,
                "input_sha256": sha256_file(path),
            }
        )
    run_identity = {
        "model_sha256": model_metadata["checkpoint_sha256"],
        "atlas_sha256": atlas_sha256,
        "input_type": arguments.input_type,
        "preprocessing": preprocessing_metadata,
        "inputs": [
            {"subject_id": row["subject_id"], "sha256": row["input_sha256"]}
            for row in input_records
        ],
    }
    run_id = "rba-" + hashlib.sha256(
        json.dumps(run_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    model_id = (
        f"RegionBrainAge-{model_metadata.get('release_version') or 'unversioned'}-"
        f"{model_metadata['checkpoint_endpoint']}"
    )

    prediction_columns = [
        "subject_id",
        "input_filename",
        "input_type",
        "global_age_direct",
        "global_age_roi",
        "global_head_gap",
        "num_valid_rois",
        "num_missing_rois",
        "registration_qc_status",
        "model_id",
        "run_id",
    ]
    roi_columns = [f"roi_{int(label):03d}" for label in lookup["label"]]
    regional_columns = [
        "subject_id",
        "input_filename",
        "input_type",
        *roi_columns,
        "model_id",
        "run_id",
    ]
    failure_columns = ["subject_id", "input_filename", "error_type", "error"]
    preprocessing_columns = ["subject_id", "status", "fail_stage", "skipped"]

    output_lookup = lookup.copy()
    output_lookup["prediction_column"] = roi_columns
    output_lookup.to_csv(output / "roi_lookup.csv", index=False)
    initialize_csv(output / "predictions.csv", prediction_columns)
    initialize_csv(output / "regional_predictions.csv", regional_columns)

    prediction_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    completed_inputs: list[dict[str, Any]] = []
    preprocessed_root = output / "preprocessed"

    for index, record in enumerate(input_records, start=1):
        subject_id = str(record["subject_id"])
        input_path = Path(record["input_path"])
        preprocessing_row = None
        print(f"[{index}/{len(input_records)}] {subject_id}", flush=True)
        try:
            if arguments.input_type == "raw":
                if raw_tools is None:
                    raise RuntimeError("Internal error: raw preprocessing tools were not initialized.")
                model_input, preprocessing_record = raw_to_preprocessed(
                    input_path,
                    subject_id,
                    preprocessed_root,
                    raw_tools,
                )
                preprocessing_row = {
                    "subject_id": subject_id,
                    "status": preprocessing_record.get("status"),
                    "fail_stage": preprocessing_record.get("fail_stage"),
                    "skipped": bool(preprocessing_record.get("skipped", False)),
                }
                if model_input is None:
                    raise RuntimeError(
                        "Preprocessing failed at stage "
                        f"{preprocessing_record.get('fail_stage') or 'unknown'}."
                    )
                registration_qc_status = "not_reviewed"
            else:
                model_input = input_path
                registration_qc_status = "not_recorded"

            image_tensor, label_tensor, image_metadata = load_and_normalize_image(
                model_input, unmasked_atlas
            )
            with torch.inference_mode():
                outputs = model(
                    image_tensor.to(device, non_blocking=device.type == "cuda"),
                    label_tensor.to(device, non_blocking=device.type == "cuda"),
                )
            direct = float(outputs["global_age_direct"][0].detach().cpu().item())
            roi_global = float(outputs["global_age_roi"][0].detach().cpu().item())
            regional = outputs["regional_age"][0].detach().cpu().numpy().astype(np.float64)
            valid = outputs["roi_valid_mask"][0].detach().cpu().numpy().astype(bool)
            if regional.shape != (222,) or valid.shape != (222,):
                raise RuntimeError(
                    "Model regional outputs must contain exactly 222 ROI values."
                )
            regional[~valid] = np.nan
            prediction_row = {
                "subject_id": subject_id,
                "input_filename": input_path.name,
                "input_type": arguments.input_type,
                "global_age_direct": direct,
                "global_age_roi": roi_global,
                "global_head_gap": abs(direct - roi_global),
                "num_valid_rois": int(valid.sum()),
                "num_missing_rois": int((~valid).sum()),
                "registration_qc_status": registration_qc_status,
                "model_id": model_id,
                "run_id": run_id,
            }
            regional_row: dict[str, Any] = {
                "subject_id": subject_id,
                "input_filename": input_path.name,
                "input_type": arguments.input_type,
                "model_id": model_id,
                "run_id": run_id,
            }
            for column, value, is_valid in zip(
                roi_columns, regional, valid, strict=True
            ):
                regional_row[column] = float(value) if is_valid else None
            completed_input = {
                "subject_id": subject_id,
                "filename": input_path.name,
                "input_sha256": record["input_sha256"],
                "model_input_sha256": sha256_file(model_input),
                "input_validation": image_metadata,
            }
            del outputs, image_tensor, label_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # preserve other inputs while recording exact failures
            if preprocessing_row is not None:
                append_csv_row(
                    output / "preprocessing_summary.csv",
                    preprocessing_row,
                    preprocessing_columns,
                )
            failure_row = {
                "subject_id": subject_id,
                "input_filename": input_path.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            append_csv_row(output / "failures.csv", failure_row, failure_columns)
            failure_rows.append(failure_row)
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        if preprocessing_row is not None:
            append_csv_row(
                output / "preprocessing_summary.csv",
                preprocessing_row,
                preprocessing_columns,
            )
        # Persist the regional row first; predictions.csv then marks a fully saved subject.
        append_csv_row(
            output / "regional_predictions.csv", regional_row, regional_columns
        )
        append_csv_row(output / "predictions.csv", prediction_row, prediction_columns)
        prediction_rows.append(prediction_row)
        completed_inputs.append(completed_input)
        print("  prediction saved", flush=True)

    status = (
        "complete"
        if prediction_rows and not failure_rows
        else "partial"
        if prediction_rows
        else "failed"
    )
    metadata = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "model_id": model_id,
        "input_type": arguments.input_type,
        "n_inputs": len(input_records),
        "n_predicted": len(prediction_rows),
        "n_failed": len(failure_rows),
        "device": str(device),
        "device_selection": device_selection,
        "primary_output": "global_age_direct",
        "secondary_outputs": ["global_age_roi", "regional_roi_ages"],
        "prediction_scale": "raw",
        "age_bias_correction_applied": False,
        "model": model_metadata,
        "atlas": {
            "filename": atlas_path.name,
            "sha256": atlas_sha256,
            "labels": [0, 222],
            "shape": list(EXPECTED_MNI_SHAPE),
            "construction": (
                "Unmasked template-grid Schaefer-200, Harvard-Oxford-15 and "
                "Buckner-7 composite; masked by each nonzero subject brain mask."
            ),
        },
        "inputs": completed_inputs,
        "failures": failure_rows,
        "interpretation": (
            "Regional outputs are ROI-conditioned chronological-age estimates, "
            "not observed regional biological ages."
        ),
    }
    write_json(output / "run_metadata.json", metadata)
    print(
        f"RegionBrainAge {status}: predicted {len(prediction_rows)}/{len(input_records)} "
        f"input(s). Outputs: {output}"
    )
    return 0 if prediction_rows and not failure_rows else 1


def main(argv: Iterable[str] | None = None) -> int:
    arguments = parser().parse_args(list(argv) if argv is not None else None)
    try:
        return run(arguments)
    except Exception as exc:
        print(f"RegionBrainAge: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
