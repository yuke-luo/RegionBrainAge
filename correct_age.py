#!/usr/bin/env python3
"""Apply the packaged, validation-fitted age-bias correction to raw predictions.

This standalone command uses only the Python standard library. It does not
load the network, inspect MRI data, fit coefficients or alter raw predictions.
Chronological ages must be supplied in years at the selected MRI acquisition.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import shutil
import sys
import tempfile
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


class Coefficient(NamedTuple):
    center: float
    intercept: float
    slope: float


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


def _by_subject(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result = {}
    for index, row in enumerate(rows, start=2):
        subject = _nonempty_string(row.get("subject_id"), f"{name} row {index} subject_id")
        if subject in result:
            raise ValueError(f"{name}: duplicate subject_id {subject!r}.")
        result[subject] = row
    return result


def _load_calibration(resources: Path, sources: dict[Path, bytes]) -> dict[str, Coefficient]:
    path = resources / "age_bias_correction.csv"
    sources[path] = path.read_bytes()
    columns, rows = _csv_table(sources[path], path.name)
    _require_columns(columns, CALIBRATION_COLUMNS, path.name)
    coefficients = {}
    sample_sizes = set()
    expected = set(GLOBAL_COLUMNS + ROI_COLUMNS)
    for row in rows:
        column = row["prediction_column"]
        if column not in expected or column in coefficients:
            raise ValueError(f"Invalid or duplicate calibration prediction_column {column!r}.")
        sample_sizes.add(_integer(row["calibration_n"], f"{column} calibration_n", minimum=1))
        if column in GLOBAL_COLUMNS:
            if row["roi_id"].strip():
                raise ValueError(f"{column}: global calibration roi_id must be blank.")
        elif _integer(row["roi_id"], f"{column} roi_id", minimum=1) != int(column[4:]):
            raise ValueError(f"{column}: calibration ROI identifier does not match its column.")
        coefficients[column] = Coefficient(
            _number(row["age_center_years"], f"{column} age_center_years", positive=True),
            _number(row["intercept_at_age_center_years"], f"{column} calibration intercept"),
            _number(row["slope_years_per_year"], f"{column} calibration slope"),
        )
    if set(coefficients) != expected:
        raise ValueError("Calibration must contain both global outputs and exactly roi_001 through roi_222.")
    if len(sample_sizes) != 1:
        raise ValueError("Calibration rows must use the same validation sample size.")
    return coefficients


def _correct(raw: float, age: float, coefficient: Coefficient, label: str):
    bias = coefficient.intercept + coefficient.slope * (age - coefficient.center)
    corrected = raw - bias
    raw_pad = raw - age
    corrected_pad = corrected - age
    if not all(math.isfinite(value) for value in (bias, corrected, raw_pad, corrected_pad)):
        raise ValueError(f"{label}: correction produced a non-finite value.")
    return corrected, raw_pad, corrected_pad


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
    """Apply frozen CSV coefficients and write three corrected CSV files.

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
    if any(source.glob("corrected_*.csv")):
        raise ValueError("This directory already contains age-bias-corrected outputs; correction cannot be repeated.")
    resources = Path(resources_dir).resolve() if resources_dir is not None else RESOURCE_DIR
    sources: dict[Path, bytes] = {}
    coefficients = _load_calibration(resources, sources)
    tables = []
    for filename in ("predictions.csv", "regional_predictions.csv"):
        path = source / filename
        sources[path] = path.read_bytes()
        columns, rows = _csv_table(sources[path], filename)
        _require_columns(columns, ("subject_id",), filename)
        if not rows:
            raise ValueError(f"{filename}: no successful predictions were found.")
        for row in rows:
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
    # When run identifiers are present in the CSVs, reject accidentally mixed runs.
    for key in ("model_id", "run_id"):
        if key in global_columns and key in regional_columns:
            values = {row[key] for _, rows, _ in tables for row in rows}
            if len(values) != 1 or not next(iter(values)).strip():
                raise ValueError(f"Prediction tables contain inconsistent {key} values.")
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
    global_output_columns += ["prediction_scale"]
    regional_extra = [column for column in regional_columns
                      if column not in set(ROI_COLUMNS) | {"subject_id", "prediction_scale"}]
    regional_output_columns = ["subject_id", "chronological_age", *ROI_COLUMNS, *regional_extra,
                               "prediction_scale"]
    if len(set(global_output_columns)) != len(global_output_columns):
        raise ValueError("Input columns conflict with corrected output field names.")
    corrected_globals, corrected_regions, corrected_pads = [], [], []
    missing_roi_total = 0
    for global_row in global_rows:
        subject = global_row["subject_id"]
        regional_row = regional_by_id[subject]
        age = age_values[subject]
        for key in ("input_filename", "input_type"):
            if key in global_row and key in regional_row and global_row[key] != regional_row[key]:
                raise ValueError(f"Prediction tables disagree on {key} for {subject!r}.")
        result = {"subject_id": subject, "chronological_age": age,
                  **{column: global_row[column] for column in global_extra},
                  "prediction_scale": CORRECTED_SCALE}
        for column, suffix in zip(GLOBAL_COLUMNS, ("direct", "roi")):
            raw = _number(global_row[column], f"{subject!r} {column}")
            corrected, raw_pad, corrected_pad = _correct(raw, age, coefficients[column], f"{subject!r} {column}")
            result.update({f"global_age_{suffix}_raw": raw, f"global_age_{suffix}_corrected": corrected,
                           f"global_pad_{suffix}_raw": raw_pad, f"global_pad_{suffix}_corrected": corrected_pad})
        if "global_head_gap" in global_row:
            result["global_head_gap_raw"] = _number(global_row["global_head_gap"], f"{subject!r} global_head_gap")
        common = {"subject_id": subject, "chronological_age": age,
                  **{column: regional_row[column] for column in regional_extra}}
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
    summary = {
        "n_subjects": n_subjects,
        "n_valid_participant_roi_pairs": n_subjects * 222 - missing_roi_total,
        "n_missing_participant_roi_pairs": missing_roi_total,
        "extra_age_ids": extras,
    }
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
    return summary


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--predictions-dir", required=True,
                         help="Raw predict.py output containing predictions.csv and regional_predictions.csv.")
    command.add_argument("--ages", required=True,
                         help="CSV with subject_id and age: chronological age in years at MRI acquisition.")
    command.add_argument("--output-dir", required=True, help="New or empty directory separate from the raw predictions.")
    return command


def main(argv: Iterable[str] | None = None) -> int:
    arguments = parser().parse_args(list(argv) if argv is not None else None)
    try:
        summary = correct_predictions(arguments.predictions_dir, arguments.ages, arguments.output_dir)
    except (ValueError, OSError, csv.Error, UnicodeError) as exc:
        print(f"RegionBrainAge correction failed: {exc}", file=sys.stderr)
        return 2
    print(f"RegionBrainAge correction complete: {summary['n_subjects']} subject(s); "
          f"{summary['n_valid_participant_roi_pairs']} valid participant-ROI pairs. "
          f"Outputs: {Path(arguments.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
