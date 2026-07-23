from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "verify" / "validate_predictions.py"
SPEC = importlib.util.spec_from_file_location("validate_predictions", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


class PredictionValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.trip_dir = self.root / "T00-Sample"
        self.trip_dir.mkdir()
        document = {
            "frames": [
                {"frame_id": 0, "timestamp": 0.0},
                {"frame_id": 1, "timestamp": 0.05},
                {"frame_id": 2, "timestamp": 0.10},
            ]
        }
        with gzip.open(
            self.trip_dir / "T00-Sample.json.gz", "wt", encoding="utf-8"
        ) as handle:
            json.dump(document, handle)
        self.csv_path = self.root / "T00-Sample.csv"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write(self, rows: list[dict], fieldnames=None) -> None:
        fields = fieldnames or ["frame_id", "timestamp", "predicted_ttc"]
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _valid_rows(self) -> list[dict]:
        return [
            {"frame_id": 0, "timestamp": 0.0, "predicted_ttc": "inf"},
            {"frame_id": 1, "timestamp": 0.05, "predicted_ttc": "2.5"},
            {"frame_id": 2, "timestamp": 0.10, "predicted_ttc": "1.0"},
        ]

    def test_valid_file_passes(self) -> None:
        self._write(self._valid_rows())
        result = VALIDATOR.validate_prediction_file(self.csv_path, self.trip_dir)
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.finite_predictions, 2)

    def test_missing_frame_fails(self) -> None:
        self._write(self._valid_rows()[:2])
        with self.assertRaisesRegex(ValueError, "expected 3 rows"):
            VALIDATOR.validate_prediction_file(self.csv_path, self.trip_dir)

    def test_duplicate_or_out_of_order_frame_fails(self) -> None:
        rows = self._valid_rows()
        rows[2]["frame_id"] = 1
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            VALIDATOR.validate_prediction_file(self.csv_path, self.trip_dir)

    def test_timestamp_mismatch_fails(self) -> None:
        rows = self._valid_rows()
        rows[1]["timestamp"] = 0.06
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            VALIDATOR.validate_prediction_file(self.csv_path, self.trip_dir)

    def test_nan_negative_and_malformed_ttc_fail(self) -> None:
        for invalid in ("nan", "-1", "unknown", ""):
            rows = self._valid_rows()
            rows[1]["predicted_ttc"] = invalid
            self._write(rows)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                VALIDATOR.validate_prediction_file(self.csv_path, self.trip_dir)

    def test_extra_ground_truth_column_fails_closed(self) -> None:
        rows = self._valid_rows()
        for row in rows:
            row["ground_truth_ttc"] = "inf"
        self._write(
            rows,
            ["frame_id", "timestamp", "predicted_ttc", "ground_truth_ttc"],
        )
        with self.assertRaisesRegex(ValueError, "unexpected columns"):
            VALIDATOR.validate_prediction_file(self.csv_path, self.trip_dir)

    def test_utf8_bom_fails_before_organizer_evaluator(self) -> None:
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["frame_id", "timestamp", "predicted_ttc"]
            )
            writer.writeheader()
            writer.writerows(self._valid_rows())
        with self.assertRaisesRegex(ValueError, "BOM"):
            VALIDATOR.validate_prediction_file(self.csv_path, self.trip_dir)


if __name__ == "__main__":
    unittest.main()
