from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
MODULE_PATH = SRC / "experiment_classical_vertical_slice.py"
SPEC = importlib.util.spec_from_file_location(
    "experiment_classical_vertical_slice",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


class CliContractTests(unittest.TestCase):
    def test_full_frame_is_default_and_roi_candidate_is_explicit(self) -> None:
        parser = PIPELINE.build_parser()

        reference = parser.parse_args([])
        candidate = parser.parse_args(["--stereo-roi-top", "96"])

        self.assertEqual(reference.stereo_roi_top, 0)
        self.assertEqual(candidate.stereo_roi_top, 96)


class RuntimeAggregationTests(unittest.TestCase):
    def test_io_and_compute_timings_are_reported_separately(self) -> None:
        records = [
            PIPELINE.RuntimeRecord(
                trip_id="T01-Sample",
                frame_id=0,
                io_ms=4.0,
                stereo_pair_ms=6.0,
                left_match_ms=3.0,
                right_match_ms=3.0,
                lr_consistency_ms=1.0,
                ground_ms=1.0,
                components_ms=1.0,
                tracking_ms=1.0,
                total_compute_ms=10.0,
                end_to_end_with_io_ms=14.0,
            ),
            PIPELINE.RuntimeRecord(
                trip_id="T01-Sample",
                frame_id=1,
                io_ms=6.0,
                stereo_pair_ms=11.0,
                left_match_ms=5.0,
                right_match_ms=6.0,
                lr_consistency_ms=2.0,
                ground_ms=1.0,
                components_ms=1.0,
                tracking_ms=1.0,
                total_compute_ms=16.0,
                end_to_end_with_io_ms=22.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            runtime_root = output_root / "runtime"
            runtime_root.mkdir()
            PIPELINE.pd.DataFrame(
                [asdict(record) for record in records]
            ).to_csv(runtime_root / "T01-Sample.csv", index=False)

            report = PIPELINE.aggregate_runtime(output_root)

            self.assertEqual(report["io_ms"]["p50"], 5.0)
            self.assertEqual(report["total_compute_ms"]["p50"], 13.0)
            self.assertEqual(
                report["end_to_end_with_io_ms"]["p50"],
                18.0,
            )
            self.assertIn("lr_consistency_ms", report)
            self.assertIn("stereo_pair_ms", report)
            self.assertTrue((output_root / "runtime_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
