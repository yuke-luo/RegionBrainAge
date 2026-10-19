"""Synthetic, standard-library-only tests; no MRI or network loading required."""

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "correct_age.py"
SPEC = importlib.util.spec_from_file_location("release_correct_age", SCRIPT)
correct_age = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = correct_age
SPEC.loader.exec_module(correct_age)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames), list(reader)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.resources = self.root / "resources"
        self.resources.mkdir()
        self.output = self.root / "corrected"
        self.ages = self.root / "ages.csv"
        self.model_hash = "a" * 64
        self.global_columns = ["subject_id", "input_filename", "input_type", "global_age_direct",
                               "global_age_roi", "global_head_gap", "num_valid_rois", "num_missing_rois",
                               "model_id", "run_id"]
        self.regional_columns = ["subject_id", "input_filename", "input_type", *correct_age.ROI_COLUMNS,
                                 "model_id", "run_id"]
        self.globals = []
        self.regionals = []
        for subject in ("001", "0002"):
            identity = {"subject_id": subject, "input_filename": subject + ".nii.gz",
                        "input_type": "preprocessed", "model_id": "test-model", "run_id": "test-run"}
            self.globals.append({**identity, "global_age_direct": 70, "global_age_roi": 68,
                                 "global_head_gap": 2, "num_valid_rois": 222, "num_missing_rois": 0})
            self.regionals.append({**identity, **{name: 50 + int(name[4:]) / 10
                                                 for name in correct_age.ROI_COLUMNS}})
        self.run = {"model_id": "test-model", "run_id": "test-run", "n_predicted": 2,
                    "prediction_scale": "raw", "model": {"checkpoint_sha256": self.model_hash},
                    "inputs": [{"subject_id": subject} for subject in ("001", "0002")]}
        self.coefficients = []
        for column in (*correct_age.GLOBAL_COLUMNS, *correct_age.ROI_COLUMNS):
            if column == "global_age_direct":
                index, intercept, slope = "", 2, .5
            elif column == "global_age_roi":
                index, intercept, slope = "", -1, -.25
            else:
                index = int(column[4:])
                intercept, slope = index / 100, index / 1000
            self.coefficients.append({"prediction_column": column, "roi_id": index, "calibration_n": 10,
                                      "age_center_years": 40, "intercept_at_age_center_years": intercept,
                                      "slope_years_per_year": slope})
        self.calibration = {"schema_version": 1, "calibration_id": "synthetic-frozen", "model_sha256": self.model_hash,
                            "fit_sample": {"n": 10, "age_min_years": 20, "age_max_years": 80},
                            "formula": "bias = a + b * (age - center); corrected_age = raw_age - bias"}
        self.save_calibration()
        self.save_inputs()
        # Shuffled IDs and an additional failed/non-predicted input are intentional.
        self.age_rows = [{"subject_id": "0002", "age": 60}, {"subject_id": "failed-03", "age": 55},
                         {"subject_id": "001", "age": 50}]
        self.save_ages()

    def save_inputs(self):
        write_csv(self.raw / "predictions.csv", self.global_columns, self.globals)
        write_csv(self.raw / "regional_predictions.csv", self.regional_columns, self.regionals)
        (self.raw / "run_metadata.json").write_text(json.dumps(self.run))

    def save_ages(self):
        write_csv(self.ages, ["subject_id", "age"], self.age_rows)

    def save_calibration(self):
        path = self.resources / "age_bias_correction.csv"
        write_csv(path, correct_age.CALIBRATION_COLUMNS, self.coefficients)
        self.calibration["coefficients_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.resources / "age_bias_correction.json").write_text(json.dumps(self.calibration))

    def run_correction(self, **changes):
        arguments = dict(predictions_dir=self.raw, ages_path=self.ages, output_dir=self.output,
                         resources_dir=self.resources)
        arguments.update(changes)
        return correct_age.correct_predictions(**arguments)

    def assert_rejected(self, match):
        before = {p: p.read_bytes() for p in self.raw.iterdir() if p.is_file()}
        with self.assertRaisesRegex(ValueError, match):
            self.run_correction()
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_math_centering_sign_and_identity_preservation(self):
        originals = {p: p.read_bytes() for p in (*self.raw.iterdir(), self.ages)}
        metadata = self.run_correction()
        _, rows = read_csv(self.output / "corrected_predictions.csv")
        self.assertEqual([row["subject_id"] for row in rows], ["001", "0002"])
        first = rows[0]
        self.assertEqual(float(first["chronological_age"]), 50)
        self.assertEqual(float(first["global_age_direct_raw"]), 70)
        self.assertEqual(float(first["global_age_direct_corrected"]), 63)
        self.assertEqual(float(first["global_pad_direct_raw"]), 20)
        self.assertEqual(float(first["global_pad_direct_corrected"]), 13)
        self.assertEqual(float(first["global_age_roi_corrected"]), 71.5)
        self.assertEqual(float(first["global_pad_roi_corrected"]), 21.5)
        self.assertEqual(float(rows[1]["global_age_direct_corrected"]), 58)
        self.assertEqual(float(rows[1]["global_pad_direct_corrected"]), -2)
        self.assertTrue(metadata["no_fit"])
        self.assertFalse(metadata["correction_fitted_to_input"])
        self.assertEqual(metadata["extra_age_ids"], ["failed-03"])
        self.assertEqual(metadata["n_valid_participant_roi_pairs"], 444)
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})
        for name, record in metadata["output_files"].items():
            self.assertEqual(record["sha256"], hashlib.sha256((self.output / name).read_bytes()).hexdigest())

    def test_per_roi_coefficients_are_matched_by_name_not_position(self):
        self.coefficients.reverse()
        self.regionals.reverse()
        self.save_calibration()
        self.save_inputs()
        self.run_correction()
        _, ages = read_csv(self.output / "corrected_regional_predictions.csv")
        _, pads = read_csv(self.output / "corrected_regional_brain_pad.csv")
        self.assertEqual([row["subject_id"] for row in ages], ["001", "0002"])
        for index in (1, 37, 201, 222):
            column = f"roi_{index:03d}"
            self.assertAlmostEqual(float(ages[0][column]), 50 + .08 * index)
            self.assertAlmostEqual(float(pads[0][column]), .08 * index)
            self.assertAlmostEqual(float(ages[1][column]), 50 + .07 * index)
        self.assertEqual(pads[0]["prediction_scale"], "age_bias_corrected_brain_pad")

    def test_blank_rois_stay_blank_and_are_not_imputed(self):
        self.regionals[0]["roi_003"] = ""
        self.globals[0]["num_valid_rois"] = 221
        self.globals[0]["num_missing_rois"] = 1
        self.save_inputs()
        metadata = self.run_correction()
        for filename in ("corrected_regional_predictions.csv", "corrected_regional_brain_pad.csv"):
            _, rows = read_csv(self.output / filename)
            self.assertEqual(rows[0]["roi_003"], "")
            self.assertNotEqual(rows[0]["roi_002"], "")
        self.assertEqual(metadata["n_missing_participant_roi_pairs"], 1)

    def test_extrapolation_warns_without_clipping_and_empty_output_is_allowed(self):
        self.age_rows[0]["age"] = 95
        self.age_rows[-1]["age"] = 18
        self.save_ages()
        self.output.mkdir()
        warning = io.StringIO()
        with contextlib.redirect_stderr(warning):
            metadata = self.run_correction()
        self.assertIn("2 subject(s)", warning.getvalue())
        self.assertIn("without clipping", warning.getvalue())
        self.assertEqual(metadata["extrapolation"]["n_below_minimum"], 1)
        self.assertEqual(metadata["extrapolation"]["n_above_maximum"], 1)
        _, rows = read_csv(self.output / "corrected_predictions.csv")
        self.assertEqual(float(rows[0]["global_age_direct_corrected"]), 79)
        self.assertEqual(float(rows[1]["global_age_direct_corrected"]), 40.5)

    def test_legacy_metadata_requires_exact_model_but_not_scale_field(self):
        del self.run["prediction_scale"]
        self.save_inputs()
        result = self.run_correction()
        self.assertTrue(result["legacy_raw_metadata_without_prediction_scale"])

    def test_unknown_model_rejected_before_output_creation(self):
        self.run["model"]["checkpoint_sha256"] = "b" * 64
        self.save_inputs()
        self.assert_rejected("checkpoint SHA-256")

    def test_duplicate_ids_rejected_in_either_prediction_table(self):
        for table in ("globals", "regionals"):
            with self.subTest(table=table):
                rows = getattr(self, table)
                rows.append(dict(rows[0]))
                self.save_inputs()
                self.assert_rejected("duplicate subject_id")
                rows.pop()

    def test_duplicate_age_ids_rejected_even_if_extra(self):
        self.age_rows.append(dict(self.age_rows[1]))
        self.save_ages()
        self.assert_rejected("duplicate subject_id")

    def test_missing_and_mismatched_subject_ids_rejected(self):
        self.age_rows.pop()
        self.save_ages()
        self.assert_rejected("age is missing")
        self.age_rows.append({"subject_id": "001", "age": 50})
        self.save_ages()
        self.regionals[0]["subject_id"] = "different"
        self.save_inputs()
        self.assert_rejected("subject IDs do not match")

    def test_blank_subject_id_rejected(self):
        self.globals[0]["subject_id"] = " "
        self.save_inputs()
        self.assert_rejected("subject_id must be a nonempty string")

    def test_nonfinite_and_blank_global_values_rejected(self):
        for value in ("nan", "inf", "-Infinity", "", "1e309"):
            with self.subTest(value=value):
                self.globals[0]["global_age_direct"] = value
                self.save_inputs()
                self.assert_rejected("finite number")

    def test_nonfinite_regional_values_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.regionals[0]["roi_222"] = value
                self.save_inputs()
                self.assert_rejected("finite number")

    def test_invalid_ages_rejected_including_unused_extra_ids(self):
        for value in ("nan", "inf", "", 0, -1):
            with self.subTest(value=value):
                self.age_rows[1]["age"] = value
                self.save_ages()
                self.assert_rejected("finite number")

    def test_missing_or_extra_roi_columns_rejected(self):
        self.regional_columns.remove("roi_222")
        for row in self.regionals:
            row.pop("roi_222")
        self.save_inputs()
        self.assert_rejected("exactly roi_001 through roi_222")
        self.regional_columns.append("roi_223")
        for row in self.regionals:
            row["roi_223"] = 50
        self.save_inputs()
        self.assert_rejected("exactly roi_001 through roi_222")

    def test_metadata_and_table_identity_consistency(self):
        for key in ("run_id", "model_id"):
            with self.subTest(key=key):
                old = self.regionals[0][key]
                self.regionals[0][key] = "wrong"
                self.save_inputs()
                self.assert_rejected("differs from run_metadata")
                self.regionals[0][key] = old
        self.run["inputs"][0]["subject_id"] = "wrong"
        self.save_inputs()
        self.assert_rejected("run_metadata inputs subject IDs")

    def test_missing_roi_count_inconsistency_rejected(self):
        self.regionals[0]["roi_003"] = ""
        self.save_inputs()
        self.assert_rejected("does not match blank regional")

    def test_calibration_hash_and_roi_mapping_rejected(self):
        self.calibration["coefficients_sha256"] = "0" * 64
        (self.resources / "age_bias_correction.json").write_text(json.dumps(self.calibration))
        self.assert_rejected("coefficient SHA-256")
        self.coefficients[2]["roi_id"] = 2
        self.save_calibration()
        self.assert_rejected("ROI identifier does not match")

    def test_duplicate_missing_and_nonfinite_calibration_rejected(self):
        self.coefficients.append(dict(self.coefficients[0]))
        self.save_calibration()
        self.assert_rejected("duplicate calibration")
        self.coefficients.pop()
        removed = self.coefficients.pop()
        self.save_calibration()
        self.assert_rejected("exactly roi_001 through roi_222")
        self.coefficients.append(removed)
        self.coefficients[0]["slope_years_per_year"] = "nan"
        self.save_calibration()
        self.assert_rejected("finite number")

    def test_nonempty_output_and_input_directory_overwrite_rejected(self):
        self.output.mkdir()
        keep = self.output / "keep.txt"
        keep.write_text("keep unchanged")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.run_correction()
        self.assertEqual(keep.read_text(), "keep unchanged")
        for destination in (self.raw, self.raw / "nested"):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "separate"):
                    self.run_correction(output_dir=destination)

    def test_corrected_directory_cannot_be_corrected_again(self):
        self.run_correction()
        with self.assertRaisesRegex(ValueError, "already contains age-bias-corrected"):
            self.run_correction(predictions_dir=self.output, output_dir=self.root / "again")
        self.assertFalse((self.root / "again").exists())

    def test_corrected_scale_marker_rejected_even_with_known_model(self):
        self.run["prediction_scale"] = "age_bias_corrected"
        self.save_inputs()
        self.assert_rejected("not marked as raw")
        self.run["prediction_scale"] = "raw"
        self.global_columns.append("prediction_scale")
        for row in self.globals:
            row["prediction_scale"] = "age_bias_corrected"
        self.save_inputs()
        self.assert_rejected("prediction_scale must be raw")

    def test_correction_applied_flag_blocks_legacy_metadata(self):
        del self.run["prediction_scale"]
        self.run["age_bias_correction_applied"] = True
        self.save_inputs()
        self.assert_rejected("already applied")

    def test_packaged_and_prediction_roi_lookup_identity_checks(self):
        columns = ["label", "source_label", "name", "source"]
        rows = [{"label": index, "source_label": index, "name": f"Region {index}", "source": "synthetic"}
                for index in range(1, 223)]
        lookup = self.resources / "roi_lookup.csv"
        write_csv(lookup, columns, rows)
        self.calibration["roi_lookup_sha256"] = hashlib.sha256(lookup.read_bytes()).hexdigest()
        self.save_calibration()
        rows[0]["name"] = "Wrong region"
        write_csv(self.raw / "roi_lookup.csv", columns, rows)
        self.assert_rejected("identities differ")
        rows[0]["name"] = "Region 1"
        write_csv(self.raw / "roi_lookup.csv", columns, rows)
        self.run_correction()
        self.assertTrue((self.output / "correction_metadata.json").is_file())

    def test_duplicate_headers_and_malformed_csv_fail(self):
        path = self.raw / "predictions.csv"
        path.write_text("subject_id,subject_id\n001,001\n")
        self.assert_rejected("duplicate column")
        self.save_inputs()
        path.write_text(path.read_text() + "001,too,few\n")
        self.assert_rejected("does not match its header")

    def test_nonfinite_computed_values_fail_before_writing(self):
        self.coefficients[0]["slope_years_per_year"] = 1e308
        self.save_calibration()
        self.assert_rejected("correction produced a non-finite")

    def test_atomic_failure_leaves_no_partial_output(self):
        with mock.patch.object(correct_age.os, "rename", side_effect=OSError("synthetic publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.run_correction()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".corrected.correct-age-*")))

    def test_cli_help_needs_only_standard_library(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--predictions-dir", result.stdout)
        self.assertIn("--ages", result.stdout)
        self.assertIn("MRI", result.stdout)

    def test_cli_application_with_packaged_fixture_and_no_site_packages(self):
        # Copy only the standalone script and synthetic packaged resources.
        standalone = self.root / "standalone"
        standalone.mkdir()
        (standalone / "correct_age.py").write_bytes(SCRIPT.read_bytes())
        (standalone / "resources").mkdir()
        for resource in self.resources.iterdir():
            (standalone / "resources" / resource.name).write_bytes(resource.read_bytes())
        result = subprocess.run([sys.executable, "-S", str(standalone / "correct_age.py"),
                                 "--predictions-dir", str(self.raw), "--ages", str(self.ages),
                                 "--output-dir", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output / "correction_metadata.json").is_file())
        self.assertIn("2 subject(s)", result.stdout)


if __name__ == "__main__":
    unittest.main()
