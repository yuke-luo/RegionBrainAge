#!/usr/bin/env python3
"""Apply the packaged, validation-fitted age-bias correction to raw predictions.

This standalone command uses only the Python standard library. It does not
load the network, inspect MRI data, fit coefficients or alter raw predictions.
Chronological ages must be supplied in years at the selected MRI acquisition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NamedTuple


RELEASE_ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = RELEASE_ROOT / "resources"
ROI_COLUMNS = tuple(f"roi_{number:03d}" for number in range(1, 223))
GLOBAL_COLUMNS = ("global_age_direct", "global_age_roi")
CALIBRATION_COLUMNS = (
    "prediction_column", "roi_id", "calibration_n", "age_center_years",
    "intercept_at_age_center_years", "slope_years_per_year",
)
CORRECTED_SCALE = "age_bias_corrected"
IMPLEMENTED_FORMULA = {
    "estimated_bias": "intercept_at_age_center_years + slope_years_per_year * (chronological_age - age_center_years)",
    "corrected_age": "raw_predicted_age - estimated_bias",
    "raw_brain_PAD": "raw_predicted_age - chronological_age",
    "corrected_brain_PAD": "corrected_age - chronological_age",
}


class Coefficient(NamedTuple):
    center: float
    intercept: float
    slope: float


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_object(data: bytes, name: str) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name}: duplicate JSON key {key!r}.")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"{name}: non-finite JSON value {value!r}.")

    result = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique_object,
                        parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return result


def _csv_table(data: bytes, name: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    columns = reader.fieldnames
    if not columns or any(not column or column != column.strip() for column in columns):
        raise ValueError(f"{name}: missing, blank or whitespace-padded column name.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name}: duplicate column names.")
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{name}: row {reader.line_num} does not match its header.")
        rows.append(row)
    return list(columns), rows


def _require_columns(columns: list[str], required: Iterable[str], name: str) -> None:
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}.")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value  # Never normalize IDs or discard leading zeroes.


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label} must be a {qualifier} number.")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    result = _number(value, label)
    if not result.is_integer() or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return int(result)


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{label} must be a full SHA-256 digest.")
    return value.lower()


def _by_subject(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result = {}
    for index, row in enumerate(rows, start=2):
        subject = _nonempty_string(row.get("subject_id"), f"{name} row {index} subject_id")
        if subject in result:
            raise ValueError(f"{name}: duplicate subject_id {subject!r}.")
        result[subject] = row
    return result


def _load_calibration(resources: Path, sources: dict[Path, bytes]):
    metadata_path = resources / "age_bias_correction.json"
    coefficient_path = resources / "age_bias_correction.csv"
    metadata_data = sources.setdefault(metadata_path, metadata_path.read_bytes())
    coefficient_data = sources.setdefault(coefficient_path, coefficient_path.read_bytes())
    metadata = _json_object(metadata_data, metadata_path.name)
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("Unsupported age-bias calibration schema_version; expected 1.")
    _nonempty_string(metadata.get("calibration_id"), "calibration_id")
    model_hash = _hash(metadata.get("model_sha256"), "calibration model_sha256")
    if _hash(metadata.get("coefficients_sha256"), "coefficients_sha256") != sha256_bytes(coefficient_data):
        raise ValueError("Packaged calibration coefficient SHA-256 does not match its metadata.")
    sample = metadata.get("fit_sample")
    if not isinstance(sample, dict):
        raise ValueError("Calibration fit_sample must be an object.")
    n = _integer(sample.get("n"), "calibration sample n", minimum=1)
    lower = _number(sample.get("age_min_years"), "calibration minimum age", positive=True)
    upper = _number(sample.get("age_max_years"), "calibration maximum age", positive=True)
    if upper < lower:
        raise ValueError("Calibration age range is reversed.")
    if not isinstance(metadata.get("formula"), (str, dict)) or not metadata["formula"]:
        raise ValueError("Calibration metadata must document its formula.")
    columns, rows = _csv_table(coefficient_data, coefficient_path.name)
    _require_columns(columns, CALIBRATION_COLUMNS, coefficient_path.name)
    coefficients = {}
    expected = set(GLOBAL_COLUMNS + ROI_COLUMNS)
    for row in rows:
        column = row["prediction_column"]
        if column not in expected or column in coefficients:
            raise ValueError(f"Invalid or duplicate calibration prediction_column {column!r}.")
        if _integer(row["calibration_n"], f"{column} calibration_n", minimum=1) != n:
            raise ValueError(f"{column}: calibration_n differs from the metadata.")
        if column in GLOBAL_COLUMNS:
            if row["roi_id"].strip():
                raise ValueError(f"{column}: global calibration roi_id must be blank.")
        elif _integer(row["roi_id"], f"{column} roi_id", minimum=1) != int(column[4:]):
            raise ValueError(f"{column}: calibration ROI identifier does not match its column.")
        center = _number(row["age_center_years"], f"{column} age_center_years", positive=True)
        if not lower <= center <= upper:
            raise ValueError(f"{column}: age center lies outside the calibration age range.")
        coefficients[column] = Coefficient(
            center,
            _number(row["intercept_at_age_center_years"], f"{column} calibration intercept"),
            _number(row["slope_years_per_year"], f"{column} calibration slope"),
        )
    if set(coefficients) != expected:
        raise ValueError("Calibration must contain both global outputs and exactly roi_001 through roi_222.")
    return metadata, coefficients, model_hash, lower, upper


def _correct(raw: float, age: float, coefficient: Coefficient, label: str):
    bias = coefficient.intercept + coefficient.slope * (age - coefficient.center)
    corrected = raw - bias
    raw_pad = raw - age
    corrected_pad = corrected - age
    if not all(math.isfinite(value) for value in (bias, corrected, raw_pad, corrected_pad)):
        raise ValueError(f"{label}: correction produced a non-finite value.")
    return corrected, raw_pad, corrected_pad


def _verify_lookup(resources: Path, source: Path, calibration: dict[str, Any], sources: dict[Path, bytes]) -> None:
    """Check the packaged ROI identities, and the inference copy when supplied."""
    if "roi_lookup_sha256" not in calibration:
        return
    packaged_path = resources / "roi_lookup.csv"
    packaged = packaged_path.read_bytes()
    sources[packaged_path] = packaged
    if sha256_bytes(packaged) != _hash(calibration["roi_lookup_sha256"], "roi_lookup_sha256"):
        raise ValueError("Packaged ROI lookup SHA-256 differs from the calibration metadata.")

    def identities(data, name):
        columns, rows = _csv_table(data, name)
        _require_columns(columns, ("label", "source_label", "name", "source"), name)
        result = {}
        for row in rows:
            label = _integer(row["label"], f"{name} label", minimum=1)
            if label in result:
                raise ValueError(f"{name}: duplicate ROI label {label}.")
            result[label] = (_integer(row["source_label"], f"{name} source_label", minimum=1),
                             row["name"], row["source"])
            if "prediction_column" in row and row["prediction_column"] != f"roi_{label:03d}":
                raise ValueError(f"{name}: prediction_column does not match ROI label {label}.")
        if set(result) != set(range(1, 223)):
            raise ValueError(f"{name} must contain exactly ROI labels 1 through 222.")
        return result

    expected = identities(packaged, "packaged ROI lookup")
    raw_lookup = source / "roi_lookup.csv"
    if raw_lookup.exists():
        sources[raw_lookup] = raw_lookup.read_bytes()
        if identities(sources[raw_lookup], "prediction ROI lookup") != expected:
            raise ValueError("Prediction ROI lookup identities differ from the calibrated atlas.")


def _csv_bytes(columns: list[str], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value
                         for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def _check_output_directory(output: Path, source: Path) -> None:
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output directory must be separate from the predictions directory.")
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Output directory must be new or empty; existing files will not be overwritten.")


def _check_sources_unchanged(sources: dict[Path, bytes]) -> None:
    for path, expected in sources.items():
        if path.read_bytes() != expected:
            raise ValueError(f"Input changed while correction was being prepared: {path}")


def correct_predictions(predictions_dir: str | Path, ages_path: str | Path,
                        output_dir: str | Path, *, resources_dir: str | Path | None = None) -> dict[str, Any]:
    """Validate, calculate and atomically publish a correction of one raw run.

    ``resources_dir`` permits isolated synthetic fixtures in unit tests. The
    public CLI always uses the bundled resources and has no coefficient override.
    """
    source = Path(predictions_dir).expanduser().resolve()
    ages = Path(ages_path).expanduser().resolve()
    requested_output = Path(output_dir).expanduser()
    if requested_output.is_symlink():
        raise ValueError("Output directory may not be a symbolic link.")
    output = requested_output.resolve()
    _check_output_directory(output, source)
    if not source.is_dir():
        raise ValueError("Predictions directory does not exist or is not a directory.")
    if (source / "correction_metadata.json").exists():
        raise ValueError("This directory already contains age-bias-corrected outputs; correction cannot be repeated.")
    resources = Path(resources_dir).resolve() if resources_dir is not None else RESOURCE_DIR
    sources: dict[Path, bytes] = {}
    calibration, coefficients, model_hash, lower, upper = _load_calibration(resources, sources)
    metadata_path = source / "run_metadata.json"
    sources[metadata_path] = metadata_path.read_bytes()
    run = _json_object(sources[metadata_path], metadata_path.name)
    if "prediction_scale" in run and run["prediction_scale"] != "raw":
        raise ValueError("Input predictions are not marked as raw; age-bias correction cannot be applied again.")
    if "age_bias_correction_applied" in run and run["age_bias_correction_applied"] is not False:
        raise ValueError("Input metadata indicates age-bias correction was already applied or is not explicitly false.")
    model = run.get("model")
    if not isinstance(model, dict) or _hash(model.get("checkpoint_sha256"), "input checkpoint_sha256") != model_hash:
        raise ValueError("Input model checkpoint SHA-256 does not match the frozen calibration model.")
    run_id = _nonempty_string(run.get("run_id"), "run_metadata run_id")
    model_id = _nonempty_string(run.get("model_id"), "run_metadata model_id")
    _verify_lookup(resources, source, calibration, sources)
    tables = []
    for filename in ("predictions.csv", "regional_predictions.csv"):
        path = source / filename
        sources[path] = path.read_bytes()
        columns, rows = _csv_table(sources[path], filename)
        _require_columns(columns, ("subject_id", "model_id", "run_id"), filename)
        if not rows:
            raise ValueError(f"{filename}: no successful predictions were found.")
        for row in rows:
            if row["model_id"] != model_id or row["run_id"] != run_id:
                raise ValueError(f"{filename}: model_id or run_id differs from run_metadata.json.")
            if "prediction_scale" in row and row["prediction_scale"] != "raw":
                raise ValueError(f"{filename}: prediction_scale must be raw, not corrected.")
        if "calibration_id" in columns or any("corrected" in column.lower() for column in columns):
            raise ValueError(f"{filename}: contains correction fields; expected raw predictions.")
        tables.append((columns, rows, _by_subject(rows, filename)))
    global_columns, global_rows, global_by_id = tables[0]
    regional_columns, _, regional_by_id = tables[1]
    _require_columns(global_columns, GLOBAL_COLUMNS, "predictions.csv")
    actual_rois = {column for column in regional_columns if column.startswith("roi_")}
    if actual_rois != set(ROI_COLUMNS):
        raise ValueError("regional_predictions.csv must contain exactly roi_001 through roi_222.")
    if set(global_by_id) != set(regional_by_id):
        raise ValueError("Global and regional prediction subject IDs do not match.")
    n_subjects = len(global_rows)
    if "n_predicted" in run and _integer(run["n_predicted"], "run_metadata n_predicted") != n_subjects:
        raise ValueError("run_metadata n_predicted differs from the prediction tables.")
    if "inputs" in run:
        if not isinstance(run["inputs"], list) or any(not isinstance(row, dict) for row in run["inputs"]):
            raise ValueError("run_metadata inputs must be a list of subject records.")
        metadata_ids = _by_subject(run["inputs"], "run_metadata inputs")
        if set(metadata_ids) != set(global_by_id):
            raise ValueError("run_metadata inputs subject IDs differ from the prediction tables.")
    sources[ages] = ages.read_bytes()
    age_columns, age_rows = _csv_table(sources[ages], "ages CSV")
    _require_columns(age_columns, ("subject_id", "age"), "ages CSV")
    age_by_id = _by_subject(age_rows, "ages CSV")
    age_values = {subject: _number(row["age"], f"age for {subject!r}", positive=True)
                  for subject, row in age_by_id.items()}
    missing = set(global_by_id) - set(age_values)
    if missing:
        raise ValueError(f"Chronological age is missing for {len(missing)} predicted subject(s): {sorted(missing)[:10]}.")
    extras = sorted(set(age_values) - set(global_by_id))

    # Assemble every result in memory before creating the output directory.
    derived_global = ["chronological_age"]
    for suffix in ("direct", "roi"):
        derived_global.extend((f"global_age_{suffix}_raw", f"global_age_{suffix}_corrected",
                               f"global_pad_{suffix}_raw", f"global_pad_{suffix}_corrected"))
    reserved = set(derived_global) | {"calibration_id"}
    if reserved.intersection(global_columns) or "chronological_age" in regional_columns:
        raise ValueError("Input tables contain derived correction columns; expected raw predict.py outputs.")
    global_extra = [column for column in global_columns
                    if column not in set(GLOBAL_COLUMNS) | {"subject_id", "prediction_scale", "global_head_gap"}]
    global_output_columns = ["subject_id", *derived_global, *global_extra]
    if "global_head_gap" in global_columns:
        global_output_columns.append("global_head_gap_raw")
    global_output_columns += ["prediction_scale", "calibration_id"]
    regional_extra = [column for column in regional_columns
                      if column not in set(ROI_COLUMNS) | {"subject_id", "prediction_scale"}]
    regional_output_columns = ["subject_id", "chronological_age", *ROI_COLUMNS, *regional_extra,
                               "prediction_scale", "calibration_id"]
    if len(set(global_output_columns)) != len(global_output_columns):
        raise ValueError("Input columns conflict with corrected output field names.")
    corrected_globals, corrected_regions, corrected_pads = [], [], []
    below = above = missing_roi_total = 0
    for global_row in global_rows:
        subject = global_row["subject_id"]
        regional_row = regional_by_id[subject]
        age = age_values[subject]
        below += age < lower
        above += age > upper
        for key in ("input_filename", "input_type"):
            if key in global_row and key in regional_row and global_row[key] != regional_row[key]:
                raise ValueError(f"Prediction tables disagree on {key} for {subject!r}.")
        result = {"subject_id": subject, "chronological_age": age,
                  **{column: global_row[column] for column in global_extra},
                  "prediction_scale": CORRECTED_SCALE, "calibration_id": calibration["calibration_id"]}
        for column, suffix in zip(GLOBAL_COLUMNS, ("direct", "roi")):
            raw = _number(global_row[column], f"{subject!r} {column}")
            corrected, raw_pad, corrected_pad = _correct(raw, age, coefficients[column], f"{subject!r} {column}")
            result.update({f"global_age_{suffix}_raw": raw, f"global_age_{suffix}_corrected": corrected,
                           f"global_pad_{suffix}_raw": raw_pad, f"global_pad_{suffix}_corrected": corrected_pad})
        if "global_head_gap" in global_row:
            result["global_head_gap_raw"] = _number(global_row["global_head_gap"], f"{subject!r} global_head_gap")
        common = {"subject_id": subject, "chronological_age": age,
                  **{column: regional_row[column] for column in regional_extra},
                  "calibration_id": calibration["calibration_id"]}
        regional_result = {**common, "prediction_scale": CORRECTED_SCALE}
        pad_result = {**common, "prediction_scale": "age_bias_corrected_brain_pad"}
        missing_rois = 0
        for column in ROI_COLUMNS:
            if not regional_row[column].strip():
                regional_result[column] = pad_result[column] = ""
                missing_rois += 1
                continue
            raw = _number(regional_row[column], f"{subject!r} {column}")
            corrected, _, corrected_pad = _correct(raw, age, coefficients[column], f"{subject!r} {column}")
            regional_result[column], pad_result[column] = corrected, corrected_pad
        for key, expected in (("num_valid_rois", 222 - missing_rois), ("num_missing_rois", missing_rois)):
            if key in global_row and _integer(global_row[key], f"{subject!r} {key}") != expected:
                raise ValueError(f"{subject!r}: {key} does not match blank regional predictions.")
        missing_roi_total += missing_rois
        corrected_globals.append(result)
        corrected_regions.append(regional_result)
        corrected_pads.append(pad_result)

    outputs = {
        "corrected_predictions.csv": _csv_bytes(global_output_columns, corrected_globals),
        "corrected_regional_predictions.csv": _csv_bytes(regional_output_columns, corrected_regions),
        "corrected_regional_brain_pad.csv": _csv_bytes(regional_output_columns, corrected_pads),
    }
    metadata = {
        "schema_version": 1, "status": "complete", "prediction_scale": CORRECTED_SCALE,
        "age_bias_correction_applied": True,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "calibration_id": calibration["calibration_id"], "model_id": model_id, "source_run_id": run_id,
        "model_sha256": model_hash, "n_subjects": n_subjects, "n_rois": 222,
        "n_valid_participant_roi_pairs": n_subjects * 222 - missing_roi_total,
        "n_missing_participant_roi_pairs": missing_roi_total,
        "no_fit": True, "correction_fitted_to_input": False,
        "formula": IMPLEMENTED_FORMULA, "calibration_formula": calibration["formula"],
        "calibration_fit_sample": calibration["fit_sample"],
        "chronological_age_definition": "User-supplied chronological age in years at MRI acquisition; not inferred from filenames or demographics.",
        "identity_matching": "Exact subject_id string; output order follows predictions.csv.",
        "n_extra_age_ids": len(extras), "extra_age_ids": extras,
        "extrapolation": {"age_min_years": lower, "age_max_years": upper,
                          "n_below_minimum": below, "n_above_maximum": above,
                          "n_total": below + above, "clipped": False},
        "legacy_raw_metadata_without_prediction_scale": "prediction_scale" not in run,
        "input_files": {str(path): {"sha256": sha256_bytes(data), "bytes": len(data)}
                        for path, data in sources.items()},
        "output_files": {name: {"sha256": sha256_bytes(data), "bytes": len(data)}
                         for name, data in outputs.items()},
        "python_version": sys.version.split()[0], "raw_predictions_modified": False,
        "interpretation": "Correction removes the frozen validation-fitted linear age bias. It does not calibrate a new population or establish regional biological age.",
    }
    identity = {"source_run_id": run_id, "input_sha256": {str(path): sha256_bytes(data) for path, data in sources.items()}}
    metadata["correction_run_id"] = "rba-correction-" + sha256_bytes(json.dumps(identity, sort_keys=True).encode())[:16]
    outputs["correction_metadata.json"] = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()
    _check_sources_unchanged(sources)
    _check_output_directory(output, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.correct-age-", dir=output.parent))
    try:
        for name, data in outputs.items():
            with (staging / name).open("xb") as handle:
                handle.write(data)
        _check_sources_unchanged(sources)
        _check_output_directory(output, source)
        if output.exists():
            output.rmdir()  # Succeeds only if the destination is still empty.
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if below + above:
        print(f"Warning: {below + above} subject(s) have MRI ages outside the calibration range "
              f"[{lower:g}, {upper:g}] years ({below} below, {above} above). "
              "The fixed correction was extrapolated without clipping.", file=sys.stderr)
    return metadata


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--predictions-dir", required=True,
                         help="Raw predict.py output containing both prediction CSVs and run_metadata.json.")
    command.add_argument("--ages", required=True,
                         help="CSV with subject_id and age: chronological age in years at MRI acquisition.")
    command.add_argument("--output-dir", required=True, help="New or empty directory separate from the raw predictions.")
    return command


def main(argv: Iterable[str] | None = None) -> int:
    arguments = parser().parse_args(list(argv) if argv is not None else None)
    try:
        metadata = correct_predictions(arguments.predictions_dir, arguments.ages, arguments.output_dir)
    except (ValueError, OSError, csv.Error, UnicodeError) as exc:
        print(f"RegionBrainAge correction failed: {exc}", file=sys.stderr)
        return 2
    print(f"RegionBrainAge correction complete: {metadata['n_subjects']} subject(s); "
          f"{metadata['n_valid_participant_roi_pairs']} valid participant-ROI pairs. "
          f"Outputs: {Path(arguments.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
