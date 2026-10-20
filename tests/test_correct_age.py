"""CSV-only correction tests; no MRI, model, third-party packages or network."""

import contextlib
import csv
import importlib.util
import io
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
OUTPUT_FILES = {
    "corrected_predictions.csv",
    "corrected_regional_predictions.csv",
    "corrected_regional_brain_pad.csv",
}


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
        self.save_calibration()
        self.save_inputs()
        # Shuffled IDs and a non-predicted input test ID matching rather than row order.
        self.age_rows = [{"subject_id": "0002", "age": 60}, {"subject_id": "failed-03", "age": 55},
                         {"subject_id": "001", "age": 50}]
        self.save_ages()

    def save_inputs(self):
        write_csv(self.raw / "predictions.csv", self.global_columns, self.globals)
        write_csv(self.raw / "regional_predictions.csv", self.regional_columns, self.regionals)

    def save_ages(self):
        write_csv(self.ages, ["subject_id", "age"], self.age_rows)

    def save_calibration(self):
        write_csv(self.resources / "age_bias_correction.csv", correct_age.CALIBRATION_COLUMNS,
                  self.coefficients)

    def source_bytes(self):
        return {path: path.read_bytes() for path in
                [*self.raw.iterdir(), *self.resources.iterdir(), self.ages] if path.is_file()}

    def run_correction(self, **changes):
        arguments = dict(predictions_dir=self.raw, ages_path=self.ages, output_dir=self.output,
                         resources_dir=self.resources)
        arguments.update(changes)
        return correct_age.correct_predictions(**arguments)

    def assert_rejected(self, match):
        before = self.source_bytes()
        with self.assertRaisesRegex(ValueError, match):
            self.run_correction()
        self.assertFalse(self.output.exists())
        self.assertEqual(before, self.source_bytes())

    def test_math_centering_sign_and_identity_preservation(self):
        originals = self.source_bytes()
        summary = self.run_correction()
        _, rows = read_csv(self.output / "corrected_predictions.csv")
        self.assertEqual([row["subject_id"] for row in rows], ["001", "0002"])
        first = rows[0]
        expected = {"chronological_age": 50, "global_age_direct_raw": 70,
                    "global_age_direct_corrected": 63, "global_pad_direct_raw": 20,
                    "global_pad_direct_corrected": 13, "global_age_roi_raw": 68,
                    "global_age_roi_corrected": 71.5, "global_pad_roi_raw": 18,
                    "global_pad_roi_corrected": 21.5, "global_head_gap_raw": 2}
        for column, value in expected.items():
            with self.subTest(column=column):
                self.assertEqual(float(first[column]), value)
        self.assertEqual(float(rows[1]["global_age_direct_corrected"]), 58)
        self.assertEqual(float(rows[1]["global_pad_direct_corrected"]), -2)
        self.assertEqual(first["prediction_scale"], "age_bias_corrected")
        self.assertEqual(summary, {"n_subjects": 2, "n_valid_participant_roi_pairs": 444,
                                   "n_missing_participant_roi_pairs": 0, "extra_age_ids": ["failed-03"]})
        self.assertEqual(originals, self.source_bytes())
        self.assertEqual({path.name for path in self.output.iterdir()}, OUTPUT_FILES)
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_minimal_csv_inputs_need_no_model_or_metadata(self):
        self.global_columns = ["subject_id", "global_age_direct", "global_age_roi"]
        self.regional_columns = ["subject_id", *correct_age.ROI_COLUMNS]
        self.globals = [{key: row[key] for key in self.global_columns} for row in self.globals]
        self.regionals = [{key: row[key] for key in self.regional_columns} for row in self.regionals]
        self.save_inputs()
        self.assertEqual(self.run_correction()["n_subjects"], 2)
        self.assertEqual({path.name for path in self.resources.iterdir()}, {"age_bias_correction.csv"})
        self.assertEqual({path.name for path in self.raw.iterdir()},
                         {"predictions.csv", "regional_predictions.csv"})
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_extra_csv_columns_preserved_verbatim(self):
        self.global_columns.append("review_note")
        self.regional_columns.append("site")
        for index, (global_row, regional_row) in enumerate(zip(self.globals, self.regionals)):
            global_row["review_note"] = f"  scan {index}, reviewed\nby analyst  "
            regional_row["site"] = f"000{index}"
        self.save_inputs()
        self.run_correction()
        _, globals_out = read_csv(self.output / "corrected_predictions.csv")
        for raw_row, result in zip(self.globals, globals_out):
            self.assertEqual(result["review_note"], raw_row["review_note"])
            self.assertEqual(result["input_filename"], raw_row["input_filename"])
        for filename in ("corrected_regional_predictions.csv", "corrected_regional_brain_pad.csv"):
            _, regional_out = read_csv(self.output / filename)
            self.assertEqual([row["site"] for row in regional_out], ["0000", "0001"])

    def test_per_roi_coefficients_and_subjects_matched_by_name_not_position(self):
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
            self.assertAlmostEqual(float(pads[1][column]), -10 + .07 * index)
        self.assertEqual(pads[0]["prediction_scale"], "age_bias_corrected_brain_pad")

    def test_blank_rois_stay_blank_and_are_not_imputed(self):
        self.regionals[0]["roi_003"] = ""
        self.regionals[0]["roi_004"] = "  "
        self.globals[0]["num_valid_rois"] = 220
        self.globals[0]["num_missing_rois"] = 2
        self.save_inputs()
        summary = self.run_correction()
        for filename in ("corrected_regional_predictions.csv", "corrected_regional_brain_pad.csv"):
            _, rows = read_csv(self.output / filename)
            self.assertEqual(rows[0]["roi_003"], "")
            self.assertEqual(rows[0]["roi_004"], "")
            self.assertNotEqual(rows[0]["roi_002"], "")
        self.assertEqual(summary["n_missing_participant_roi_pairs"], 2)
        self.assertEqual(summary["n_valid_participant_roi_pairs"], 442)

    def test_positive_ages_outside_typical_range_are_not_clipped_or_warned(self):
        self.age_rows[0]["age"] = 150
        self.age_rows[-1]["age"] = .5
        self.save_ages()
        self.output.mkdir()
        warning = io.StringIO()
        with contextlib.redirect_stderr(warning):
            self.run_correction()
        self.assertEqual(warning.getvalue(), "")
        _, rows = read_csv(self.output / "corrected_predictions.csv")
        self.assertEqual(float(rows[0]["chronological_age"]), .5)
        self.assertEqual(float(rows[1]["chronological_age"]), 150)
        self.assertEqual(float(rows[0]["global_age_direct_corrected"]), 87.75)
        self.assertEqual(float(rows[1]["global_age_direct_corrected"]), 13)

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

    def test_ids_are_not_numerically_coerced_or_whitespace_normalized(self):
        for value in ("1", "001 "):
            with self.subTest(value=value):
                self.age_rows[-1]["subject_id"] = value
                self.save_ages()
                self.assert_rejected("age is missing")

    def test_blank_subject_id_rejected(self):
        self.globals[0]["subject_id"] = " "
        self.save_inputs()
        self.assert_rejected("subject_id must be a nonempty string")

    def test_nonfinite_and_blank_global_values_rejected(self):
        for value in ("nan", "inf", "-Infinity", "", "1e309", "not-a-number"):
            with self.subTest(value=value):
                self.globals[0]["global_age_direct"] = value
                self.save_inputs()
                self.assert_rejected("finite number")

    def test_nonfinite_regional_values_rejected(self):
        for value in ("nan", "inf", "-inf", "not-a-number"):
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
        self.regional_columns.extend(("roi_222", "roi_223"))
        for row in self.regionals:
            row.update(roi_222=50, roi_223=50)
        self.save_inputs()
        self.assert_rejected("exactly roi_001 through roi_222")

    def test_missing_required_global_and_age_columns_rejected(self):
        self.global_columns.remove("global_age_roi")
        for row in self.globals:
            row.pop("global_age_roi")
        self.save_inputs()
        self.assert_rejected("missing required columns")
        self.global_columns.append("global_age_roi")
        for row in self.globals:
            row["global_age_roi"] = 68
        self.save_inputs()
        write_csv(self.ages, ["subject_id"], [{"subject_id": "001"}])
        self.assert_rejected("missing required columns")

    def test_optional_identifiers_and_input_descriptors_must_agree(self):
        for key in ("run_id", "model_id", "input_filename", "input_type"):
            with self.subTest(key=key):
                old = self.regionals[0][key]
                self.regionals[0][key] = "wrong"
                self.save_inputs()
                self.assert_rejected(key)
                self.regionals[0][key] = old

    def test_missing_roi_count_inconsistency_rejected(self):
        self.regionals[0]["roi_003"] = ""
        self.save_inputs()
        self.assert_rejected("does not match blank regional")

    def test_calibration_roi_mapping_rejected(self):
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
        originals = self.source_bytes()
        self.output.mkdir()
        keep = self.output / "keep.txt"
        keep.write_text("keep unchanged")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.run_correction()
        self.assertEqual(keep.read_text(), "keep unchanged")
        for destination in (self.raw, self.raw / "nested", self.root):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "separate"):
                    self.run_correction(output_dir=destination)
        self.assertEqual(originals, self.source_bytes())

    def test_symbolic_link_output_rejected(self):
        destination = self.root / "destination"
        destination.mkdir()
        self.output.symlink_to(destination, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.run_correction()
        self.assertEqual(list(destination.iterdir()), [])
        self.assertTrue(self.output.is_symlink())

    def test_corrected_directory_cannot_be_corrected_again(self):
        self.run_correction()
        with self.assertRaisesRegex(ValueError, "already contains age-bias-corrected"):
            self.run_correction(predictions_dir=self.output, output_dir=self.root / "again")
        self.assertFalse((self.root / "again").exists())

    def test_nonraw_scale_rejected_in_either_table(self):
        for columns, rows in ((self.global_columns, self.globals),
                              (self.regional_columns, self.regionals)):
            columns.append("prediction_scale")
            for row in rows:
                row["prediction_scale"] = "age_bias_corrected"
            self.save_inputs()
            self.assert_rejected("prediction_scale must be raw")
            columns.remove("prediction_scale")
            for row in rows:
                row.pop("prediction_scale")

    def test_derived_correction_fields_rejected_even_without_scale_marker(self):
        for column in ("global_age_direct_corrected", "global_age_direct_raw", "chronological_age"):
            with self.subTest(column=column):
                self.global_columns.append(column)
                for row in self.globals:
                    row[column] = 50
                self.save_inputs()
                self.assert_rejected("correction (fields|columns)")
                self.global_columns.remove(column)
                for row in self.globals:
                    row.pop(column)

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

    def test_atomic_failure_preserves_sources_and_leaves_no_partial_output(self):
        originals = self.source_bytes()
        with mock.patch.object(correct_age.os, "rename", side_effect=OSError("synthetic publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.run_correction()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".corrected.correct-age-*")))
        self.assertEqual(originals, self.source_bytes())

    def test_cli_help_needs_only_standard_library(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--predictions-dir", result.stdout)
        self.assertIn("--ages", result.stdout)
        self.assertIn("MRI", result.stdout)

    def test_cli_application_with_only_csv_fixtures_and_no_site_packages(self):
        standalone = self.root / "standalone"
        standalone.mkdir()
        (standalone / "correct_age.py").write_bytes(SCRIPT.read_bytes())
        (standalone / "resources").mkdir()
        coefficients = self.resources / "age_bias_correction.csv"
        (standalone / "resources" / coefficients.name).write_bytes(coefficients.read_bytes())
        originals = self.source_bytes()
        result = subprocess.run([sys.executable, "-S", str(standalone / "correct_age.py"),
                                 "--predictions-dir", str(self.raw), "--ages", str(self.ages),
                                 "--output-dir", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("2 subject(s)", result.stdout)
        self.assertEqual({path.name for path in self.output.iterdir()}, OUTPUT_FILES)
        self.assertFalse(list(self.root.rglob("*.json")))
        self.assertEqual(originals, self.source_bytes())
        _, rows = read_csv(self.output / "corrected_predictions.csv")
        self.assertEqual(float(rows[0]["global_age_direct_corrected"]), 63)


if __name__ == "__main__":
    unittest.main()
