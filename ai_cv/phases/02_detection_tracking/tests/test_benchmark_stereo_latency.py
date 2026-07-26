from __future__ import annotations

import math
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import benchmark_stereo_latency as BENCHMARK  # noqa: E402


class RuntimeSummaryTests(unittest.TestCase):
    def test_nvidia_smi_parser_selects_current_process_peak(self) -> None:
        output = "101, 350\n202, 900 MiB\n101, 420 MiB\nmalformed"

        value = BENCHMARK._parse_nvidia_smi_process_memory(output, 101)

        self.assertEqual(value, 420.0)
        self.assertIsNone(
            BENCHMARK._parse_nvidia_smi_process_memory(output, 999)
        )
        self.assertEqual(
            BENCHMARK._parse_nvidia_smi_device_memory("1536 MiB\n"), 1536.0
        )

    def test_peak_process_memory_is_available_on_supported_host(self) -> None:
        peak = BENCHMARK._peak_rss_mb()

        self.assertIsNotNone(peak)
        assert peak is not None
        self.assertGreater(peak, 0)

    def test_percentiles_are_reported_in_milliseconds(self) -> None:
        summary = BENCHMARK.percentile_summary([10.0, 20.0, 30.0, 40.0])

        self.assertEqual(summary["p50"], 25.0)
        self.assertEqual(summary["mean"], 25.0)
        self.assertAlmostEqual(summary["p95"], 38.5)

    def test_aggregate_keeps_io_and_pipeline_compute_separate(self) -> None:
        rows = [
            {
                "image_load_ms": 100.0,
                "stereo_ms": 20.0,
                "pipeline_compute_ms": 40.0,
                "end_to_end_ms": 140.0,
                "not_a_timing": 1,
            },
            {
                "image_load_ms": 200.0,
                "stereo_ms": 25.0,
                "pipeline_compute_ms": 45.0,
                "end_to_end_ms": 245.0,
                "not_a_timing": 2,
            },
        ]

        summary = BENCHMARK.aggregate_runtime(rows)

        self.assertEqual(summary["pipeline_compute_ms"]["p95"], 44.75)
        self.assertEqual(summary["image_load_ms"]["p50"], 150.0)
        self.assertNotIn("not_a_timing", summary)


class EvaluationGateTests(unittest.TestCase):
    def test_parity_accumulator_passes_uniform_conversion_limits(self) -> None:
        reference = fake_stereo_result(np.ones((10, 10), dtype=np.float32))
        candidate = fake_stereo_result(
            np.full((10, 10), 1.1, dtype=np.float32)
        )
        accumulator = BENCHMARK.ParityAccumulator()
        accumulator.add(
            trip_id="T01-Sample",
            frame_id=0,
            reference=reference,
            candidate=candidate,
        )

        report = accumulator.finalize(expected_frames=1)

        self.assertTrue(report["passed"])
        self.assertLessEqual(
            report["aggregate"]["mean_absolute_error_px"], 0.25
        )

    def test_parity_accumulator_rejects_error_outliers_and_missing_pixels(
        self,
    ) -> None:
        reference_disparity = np.ones((10, 100), dtype=np.float32)
        candidate_disparity = np.full((10, 100), 1.3, dtype=np.float32)
        candidate_disparity.flat[:6] = 5.0
        candidate_disparity.flat[6:12] = -1.0
        candidate_valid = candidate_disparity > 0
        accumulator = BENCHMARK.ParityAccumulator()
        accumulator.add(
            trip_id="T01-Sample",
            frame_id=0,
            reference=fake_stereo_result(reference_disparity),
            candidate=fake_stereo_result(
                candidate_disparity, valid=candidate_valid
            ),
        )

        report = accumulator.finalize(expected_frames=1)

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["mean_absolute_error_px"])
        self.assertFalse(report["gates"]["bad_3px_fraction"])
        self.assertFalse(
            report["gates"]["missing_reference_valid_fraction"]
        )

    def test_danger_confusion_counts_exact_threshold_outcomes(self) -> None:
        confusion = BENCHMARK.danger_confusion(
            [
                (1.0, 1.5),
                (1.0, math.inf),
                (math.inf, 1.5),
                (math.inf, math.inf),
            ]
        )

        self.assertEqual(confusion, {"tp": 1, "fp": 1, "fn": 1, "tn": 1})

    def test_all_hard_and_quality_gates_can_pass(self) -> None:
        timing = {
            "pipeline_compute_ms": {
                "p50": 40.0,
                "p95": 49.0,
                "p99": 52.0,
                "mean": 41.0,
            }
        }
        report = BENCHMARK.acceptance_report(
            timing=timing,
            evaluation={
                "overall_composite_score": 28.3,
                "overall_f1": 0.395,
            },
            peak_gpu_memory_mb=4096.0,
            frames_measured=3600,
            expected_frames=3600,
            nondeterministic_predictions=0,
            lane_reference=(28.5, 0.400),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failure_reasons"], [])

    def test_gate_uses_pipeline_p95_not_disk_inclusive_latency(self) -> None:
        timing = {
            "pipeline_compute_ms": {
                "p50": 40.0,
                "p95": 49.0,
                "p99": 52.0,
                "mean": 41.0,
            },
            "end_to_end_ms": {
                "p50": 200.0,
                "p95": 250.0,
                "p99": 275.0,
                "mean": 210.0,
            },
        }
        report = BENCHMARK.acceptance_report(
            timing=timing,
            evaluation={
                "overall_composite_score": 28.7,
                "overall_f1": 0.402,
            },
            peak_gpu_memory_mb=0.0,
            frames_measured=3600,
            expected_frames=3600,
            nondeterministic_predictions=0,
            lane_reference=None,
        )

        self.assertTrue(
            report["gates"]["pipeline_p95_strictly_below_target"]
        )
        self.assertTrue(report["passed"])

    def test_latency_target_is_strict_and_recorded(self) -> None:
        timing = {
            "pipeline_compute_ms": {
                "p50": 60.0,
                "p95": 75.0,
                "p99": 80.0,
                "mean": 62.0,
            }
        }

        report = BENCHMARK.acceptance_report(
            timing=timing,
            evaluation={
                "overall_composite_score": 28.7,
                "overall_f1": 0.402,
            },
            peak_gpu_memory_mb=0.0,
            frames_measured=3600,
            expected_frames=3600,
            nondeterministic_predictions=0,
            lane_reference=None,
            latency_target_ms=75.0,
        )

        self.assertFalse(
            report["gates"]["pipeline_p95_strictly_below_target"]
        )
        self.assertEqual(report["latency_target_ms"], 75.0)
        self.assertEqual(report["latency_comparison"], "strict_less_than")

    def test_official_protocol_gates_repeats_warmup_and_runtime_rows(
        self,
    ) -> None:
        timing = {
            "pipeline_compute_ms": {
                "p50": 60.0,
                "p95": 70.0,
                "p99": 74.0,
                "mean": 62.0,
            }
        }
        report = BENCHMARK.acceptance_report(
            timing=timing,
            evaluation={
                "overall_composite_score": 28.7,
                "overall_f1": 0.402,
            },
            peak_gpu_memory_mb=0.0,
            frames_measured=3599,
            expected_frames=3600,
            nondeterministic_predictions=0,
            lane_reference=None,
            repeats=4,
            warmup_frames=99,
            complete_dataset_protocol=False,
            runtime_rows_measured=14396,
            expected_runtime_rows=18000,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["exactly_five_repeats"])
        self.assertFalse(report["gates"]["warmup_at_least_100_frames"])
        self.assertFalse(report["gates"]["complete_six_trip_dataset"])
        self.assertFalse(report["gates"]["exactly_3600_frames_per_repeat"])
        self.assertFalse(report["gates"]["exactly_18000_runtime_rows"])

    def test_unknown_gpu_memory_cannot_pass_a_hard_resource_gate(self) -> None:
        timing = {
            "pipeline_compute_ms": {
                "p50": 40.0,
                "p95": 49.0,
                "p99": 52.0,
                "mean": 41.0,
            }
        }
        report = BENCHMARK.acceptance_report(
            timing=timing,
            evaluation={
                "overall_composite_score": 28.7,
                "overall_f1": 0.402,
            },
            peak_gpu_memory_mb=None,
            frames_measured=3600,
            expected_frames=3600,
            nondeterministic_predictions=0,
            lane_reference=None,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["gpu_vram_le_5gb"])
        self.assertTrue(
            any("VRAM peak unavailable" in reason for reason in report["failure_reasons"])
        )

    def test_partial_smoke_run_is_not_reported_as_quality_rejection(self) -> None:
        timing = {
            "pipeline_compute_ms": {
                "p50": 40.0,
                "p95": 49.0,
                "p99": 52.0,
                "mean": 41.0,
            }
        }

        report = BENCHMARK.acceptance_report(
            timing=timing,
            evaluation=None,
            peak_gpu_memory_mb=0.0,
            frames_measured=2,
            expected_frames=2,
            nondeterministic_predictions=0,
            lane_reference=None,
        )

        self.assertEqual(report["status"], "not_evaluated")
        self.assertIsNone(report["passed"])
        self.assertIsNone(report["gates"]["stage2a_quality_budget"])
        self.assertTrue(report["pending_reasons"])


class DatasetProtocolTests(unittest.TestCase):
    def test_freezes_exactly_600_ordered_unique_frames_per_trip(self) -> None:
        class Dataset:
            def __init__(self, path: Path) -> None:
                self.trip_id = path.name

            def iter_frames(self):
                return [
                    types.SimpleNamespace(
                        frame_id=index,
                        timestamp=index / 10.0,
                        min_ttc=math.inf,
                    )
                    for index in range(600)
                ]

            def load_calibration(self):
                return {"baseline_m": 0.54, "trip": self.trip_id}

        protocol, frozen = BENCHMARK.freeze_dataset_protocol(
            Dataset, Path("Practice_Dataset")
        )

        self.assertTrue(protocol["complete"])
        self.assertEqual(protocol["expected_frames_per_repeat"], 3600)
        self.assertEqual(len(frozen), 6)
        self.assertTrue(
            all(len(frame_ids) == 600 for frame_ids in frozen.values())
        )

    def test_duplicate_or_misordered_or_missing_frame_rejects_protocol(
        self,
    ) -> None:
        class Dataset:
            def __init__(self, path: Path) -> None:
                self.trip_id = path.name

            def iter_frames(self):
                ids = list(range(600))
                if self.trip_id == "T01-Sample":
                    ids = ids[:-1]
                elif self.trip_id == "T02-Sample":
                    ids[-1] = ids[-2]
                elif self.trip_id == "T03-Sample":
                    ids[10], ids[11] = ids[11], ids[10]
                return [
                    types.SimpleNamespace(
                        frame_id=index,
                        timestamp=position / 10.0,
                        min_ttc=math.inf,
                    )
                    for position, index in enumerate(ids)
                ]

            def load_calibration(self):
                return {"baseline_m": 0.54}

        protocol, _ = BENCHMARK.freeze_dataset_protocol(
            Dataset, Path("Practice_Dataset")
        )

        self.assertFalse(protocol["complete"])
        self.assertTrue(any("599 frames" in item for item in protocol["issues"]))
        self.assertTrue(any("duplicate" in item for item in protocol["issues"]))
        self.assertTrue(any("ordered" in item for item in protocol["issues"]))


class ProvenanceValidationTests(unittest.TestCase):
    def test_onnx_sidecar_binds_artifact_checkpoint_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"static-onnx")
            manifest = {
                "schema": BENCHMARK.ARTIFACT_MANIFEST_SCHEMA,
                "artifact": {
                    "name": artifact.name,
                    "kind": "lightstereo-s-onnx-opset17-static",
                    "bytes": artifact.stat().st_size,
                    "sha256": BENCHMARK.sha256_file(artifact),
                },
                "generation_command_argv": ["python", "export.py"],
                "metadata": {
                    "openstereo_revision": BENCHMARK.OPENSTEREO_REVISION,
                    "openstereo_tracked_tree_clean": True,
                    "backend": "lightstereo-onnx",
                    "precision": "fp32",
                    "checkpoint": {
                        "name": "LightStereo-S-KITTI.ckpt",
                        "sha256": (
                            BENCHMARK.OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
                        ),
                    },
                    "config": {
                        "relative_path": BENCHMARK.LIGHTSTEREO_CONFIG_RELATIVE,
                        "sha256": "c" * 64,
                    },
                    "opset": 17,
                    "input_names": BENCHMARK.LIGHTSTEREO_INPUT_NAMES,
                    "output_name": BENCHMARK.LIGHTSTEREO_OUTPUT_NAME,
                    "input_shape_nchw": BENCHMARK.LIGHTSTEREO_INPUT_SHAPE,
                    "dynamic_axes": False,
                },
            }
            sidecar = artifact.with_name(f"{artifact.name}.manifest.json")
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            checkout = {
                "openstereo_revision": BENCHMARK.OPENSTEREO_REVISION,
                "config_path": BENCHMARK.LIGHTSTEREO_CONFIG_RELATIVE,
                "config_sha256": "c" * 64,
            }
            with mock.patch.object(
                BENCHMARK,
                "_validate_openstereo_checkout",
                return_value=checkout,
            ):
                provenance = BENCHMARK.validate_model_provenance(
                    backend="lightstereo-onnx",
                    precision="fp32",
                    model_path=artifact,
                    openstereo_root=root,
                    config_path=None,
                )

            self.assertEqual(
                provenance["source_checkpoint_sha256"],
                BENCHMARK.OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256,
            )
            self.assertEqual(
                provenance["artifact_sha256"],
                BENCHMARK.sha256_file(artifact),
            )

            artifact.write_bytes(b"tampered")
            with (
                mock.patch.object(
                    BENCHMARK,
                    "_validate_openstereo_checkout",
                    return_value=checkout,
                ),
                self.assertRaisesRegex(
                    BENCHMARK.BackendConfigurationError, "does not match"
                ),
            ):
                BENCHMARK.validate_model_provenance(
                    backend="lightstereo-onnx",
                    precision="fp32",
                    model_path=artifact,
                    openstereo_root=root,
                    config_path=None,
                )

    def test_tensorrt_declared_precision_must_match_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.engine"
            artifact.write_bytes(b"engine")
            sidecar = artifact.with_name(f"{artifact.name}.manifest.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "schema": BENCHMARK.ARTIFACT_MANIFEST_SCHEMA,
                        "artifact": {
                            "name": artifact.name,
                            "kind": "lightstereo-s-tensorrt10-fp16",
                            "bytes": artifact.stat().st_size,
                            "sha256": BENCHMARK.sha256_file(artifact),
                        },
                        "generation_command_argv": ["python", "build-engine"],
                        "metadata": {
                            "openstereo_revision": (
                                BENCHMARK.OPENSTEREO_REVISION
                            ),
                            "backend": "lightstereo-tensorrt",
                            "checkpoint": {
                                "sha256": (
                                    BENCHMARK
                                    .OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
                                )
                            },
                            "config": {
                                "relative_path": (
                                    BENCHMARK.LIGHTSTEREO_CONFIG_RELATIVE
                                ),
                                "sha256": "c" * 64,
                            },
                            "source_onnx": {
                                "name": "source.onnx",
                                "sha256": "a" * 64,
                            },
                            "input_shape_nchw": (
                                BENCHMARK.LIGHTSTEREO_INPUT_SHAPE
                            ),
                            "input_names": BENCHMARK.LIGHTSTEREO_INPUT_NAMES,
                            "output_name": BENCHMARK.LIGHTSTEREO_OUTPUT_NAME,
                            "precision": "fp16",
                            "calibration": None,
                            "dependencies": {"tensorrt": "10.8.0"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    BENCHMARK,
                    "_validate_openstereo_checkout",
                    return_value={
                        "openstereo_revision": BENCHMARK.OPENSTEREO_REVISION,
                        "config_path": BENCHMARK.LIGHTSTEREO_CONFIG_RELATIVE,
                        "config_sha256": "c" * 64,
                    },
                ),
                self.assertRaisesRegex(
                    BENCHMARK.BackendConfigurationError,
                    "declared backend/precision",
                ),
            ):
                BENCHMARK.validate_model_provenance(
                    backend="lightstereo-tensorrt",
                    precision="int8",
                    model_path=artifact,
                    openstereo_root=root,
                    config_path=None,
                )

    def test_parity_report_must_bind_exact_candidate_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parity.json"
            report = {
                "kind": BENCHMARK.PARITY_REPORT_KIND,
                "status": "passed",
                "passed": True,
                "manifest": {"entry_count": 72, "sha256": "m" * 64},
                "reference": {
                    "backend": "lightstereo-pytorch",
                    "precision": "fp32",
                    "model_sha256": (
                        BENCHMARK.OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
                    ),
                },
                "candidate": {
                    "backend": "lightstereo-tensorrt",
                    "precision": "fp16",
                    "model_sha256": "a" * 64,
                },
                "aggregate": {"frame_count": 72},
                "gates": {
                    "mean_absolute_error_px": True,
                    "bad_3px_fraction": True,
                    "missing_reference_valid_fraction": True,
                    "all_72_frames": True,
                },
            }
            path.write_text(json.dumps(report), encoding="utf-8")
            provenance = {
                "source_checkpoint_sha256": (
                    BENCHMARK.OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
                )
            }

            validated = BENCHMARK.validate_parity_report(
                path,
                backend="lightstereo-tensorrt",
                precision="fp16",
                model_sha256="a" * 64,
                model_provenance=provenance,
            )
            self.assertTrue(validated["passed"])
            with self.assertRaisesRegex(
                BENCHMARK.BackendConfigurationError,
                "does not match",
            ):
                BENCHMARK.validate_parity_report(
                    path,
                    backend="lightstereo-tensorrt",
                    precision="fp16",
                    model_sha256="b" * 64,
                    model_provenance=provenance,
                )

    def test_lane_reference_requires_matching_protocol_and_model_lineage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lane.json"
            fingerprint = "d" * 64
            provenance = {
                "source_checkpoint_sha256": (
                    BENCHMARK.OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
                ),
                "config_sha256": "c" * 64,
                "openstereo_revision": BENCHMARK.OPENSTEREO_REVISION,
            }
            path.write_text(
                json.dumps(
                    {
                        "backend": "lightstereo-pytorch",
                        "precision": "fp32",
                        "model_sha256": (
                            BENCHMARK.OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
                        ),
                        "model_provenance": provenance,
                        "configuration": {
                            "protocol_schema": (
                                BENCHMARK.BENCHMARK_PROTOCOL_SCHEMA
                            ),
                            "trips": list(BENCHMARK.TRIPS),
                            "repeats": 5,
                            "warmup_frames": 100,
                            "max_frames_per_trip": None,
                            "latency_target_ms": 75.0,
                            "latency_comparison": "strict_less_than",
                        },
                        "dataset": {
                            "frames_per_repeat": 3600,
                            "runtime_rows": 18000,
                            "protocol": {
                                "complete": True,
                                "dataset_fingerprint_sha256": fingerprint,
                            },
                        },
                        "evaluation": {
                            "overall_composite_score": 28.7,
                            "overall_f1": 0.402,
                        },
                        "acceptance": {
                            "gates": {
                                "exactly_five_repeats": True,
                                "warmup_at_least_100_frames": True,
                                "complete_six_trip_dataset": True,
                                "exactly_3600_frames_per_repeat": True,
                                "exactly_18000_runtime_rows": True,
                                "repeat_determinism": True,
                                "artifact_provenance_valid": True,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            quality, record = BENCHMARK.load_lane_reference_summary(
                path,
                dataset_protocol={
                    "complete": True,
                    "dataset_fingerprint_sha256": fingerprint,
                },
                model_provenance=provenance,
                latency_target_ms=75.0,
            )

            self.assertEqual(quality, (28.7, 0.402))
            self.assertEqual(record["dataset_fingerprint_sha256"], fingerprint)


class CliContractTests(unittest.TestCase):
    def test_parity_subcommand_requires_frozen_inputs_and_candidate(self) -> None:
        parser = BENCHMARK.build_parity_parser()
        args = parser.parse_args(
            [
                "--manifest",
                "parity.json",
                "--data-root",
                "Practice_Dataset",
                "--reference-model-path",
                "reference.pth",
                "--candidate-backend",
                "lightstereo-tensorrt",
                "--candidate-precision",
                "fp16",
                "--candidate-model-path",
                "candidate.engine",
            ]
        )

        self.assertEqual(args.warmup_frames, 5)
        self.assertEqual(args.candidate_precision, "fp16")
        self.assertFalse(hasattr(args, "stereo_roi_top"))

    def test_parity_main_does_not_require_benchmark_roi_arguments(self) -> None:
        reference = mock.Mock()
        candidate = mock.Mock()
        report = {
            "status": "passed",
            "passed": True,
            "aggregate": {
                "mean_absolute_error_px": 0.1,
                "p95_absolute_error_px": 0.2,
                "bad_3px_fraction": 0.0,
                "missing_reference_valid_fraction": 0.0,
            },
        }
        arguments = [
            "--manifest",
            "parity.json",
            "--data-root",
            "Practice_Dataset",
            "--reference-model-path",
            "reference.pth",
            "--candidate-backend",
            "lightstereo-onnx",
            "--candidate-precision",
            "fp32",
            "--candidate-model-path",
            "candidate.onnx",
        ]
        with (
            mock.patch.object(
                BENCHMARK,
                "create_backend",
                side_effect=[reference, candidate],
            ),
            mock.patch.object(
                BENCHMARK, "run_parity_gate", return_value=report
            ),
        ):
            status = BENCHMARK.parity_main(arguments)

        self.assertEqual(status, 0)
        reference.close.assert_called_once_with()
        candidate.close.assert_called_once_with()

    def test_backend_precision_and_repeats_are_explicit(self) -> None:
        parser = BENCHMARK.build_parser()
        args = parser.parse_args(
            ["--backend", "sgbm", "--precision", "fp32", "--repeats", "5"]
        )

        self.assertEqual(args.backend, "sgbm")
        self.assertEqual(args.precision, "fp32")
        self.assertEqual(args.repeats, 5)
        self.assertEqual(args.warmup_frames, 100)
        self.assertEqual(args.latency_target_ms, 75.0)
        self.assertEqual(args.opencv_threads, 6)
        self.assertEqual(args.stereo_workers, 1)
        self.assertEqual(args.stereo_roi_top, 0)

    def test_sgbm_roi_candidate_is_explicit_in_cli(self) -> None:
        parser = BENCHMARK.build_parser()
        args = parser.parse_args(
            [
                "--backend",
                "sgbm",
                "--precision",
                "fp32",
                "--repeats",
                "5",
                "--stereo-roi-top",
                "96",
            ]
        )

        self.assertEqual(args.stereo_roi_top, 96)

    def test_sgbm_roi_rejects_unfrozen_crop(self) -> None:
        parser = BENCHMARK.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--backend",
                    "sgbm",
                    "--precision",
                    "fp32",
                    "--repeats",
                    "5",
                    "--stereo-roi-top",
                    "95",
                ]
            )

    def test_converted_official_run_requires_parity_report(self) -> None:
        with self.assertRaises(SystemExit):
            BENCHMARK.main(
                [
                    "--backend",
                    "lightstereo-onnx",
                    "--precision",
                    "fp32",
                    "--repeats",
                    "5",
                    "--model-path",
                    "candidate.onnx",
                    "--lane-reference-summary",
                    "lane.json",
                ]
            )

    def test_official_cli_requires_five_repeats_and_100_warmups(self) -> None:
        for arguments in (
            [
                "--backend",
                "sgbm",
                "--precision",
                "fp32",
                "--repeats",
                "4",
            ],
            [
                "--backend",
                "sgbm",
                "--precision",
                "fp32",
                "--repeats",
                "5",
                "--warmup-frames",
                "99",
            ],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    BENCHMARK.main(arguments)

    def test_regular_main_rejects_invalid_or_learned_backend_roi(self) -> None:
        with self.assertRaises(SystemExit):
            BENCHMARK.main(
                [
                    "--backend",
                    "sgbm",
                    "--precision",
                    "fp32",
                    "--repeats",
                    "1",
                    "--stereo-roi-top",
                    "360",
                ]
            )
        with self.assertRaises(SystemExit):
            BENCHMARK.main(
                [
                    "--backend",
                    "lightstereo-onnx",
                    "--precision",
                    "fp32",
                    "--repeats",
                    "1",
                    "--skip-evaluation",
                    "--stereo-roi-top",
                    "96",
                ]
            )

    def test_comparison_table_contains_required_decision_fields(self) -> None:
        summary = {
            "backend": "sgbm",
            "precision": "fp32",
            "model_sha256": None,
            "timing_ms": {
                "pipeline_compute_ms": {
                    "p50": 40.0,
                    "p95": 49.0,
                    "p99": 55.0,
                }
            },
            "throughput": {"fps_from_pipeline_mean": 23.0},
            "evaluation": {
                "overall_composite_score": 28.7,
                "overall_f1": 0.402,
                "worst_trip_composite": 4.6,
            },
            "danger_confusion": {"tp": 135, "fp": 340, "fn": 69, "tn": 3056},
            "resources": {
                "peak_process_ram_mb": 512.0,
                "peak_gpu_memory_mb": 0.0,
            },
            "environment": {"processor": "test-cpu"},
            "acceptance": {
                "status": "accepted",
                "passed": True,
                "failure_reasons": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.csv"
            BENCHMARK.write_comparison(output, summary)
            contents = output.read_text(encoding="utf-8")

        self.assertIn("pipeline_p95_ms", contents)
        self.assertIn("latency_target_ms_strict_lt", contents)
        self.assertIn("overall_composite", contents)
        self.assertIn("danger_f1", contents)
        self.assertIn("peak_gpu_memory_mb", contents)
        self.assertIn("135,340,69", contents)

    def test_aggregator_applies_latency_tie_then_higher_f1_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [
                fake_summary("fast", 40.0, 0.400, passed=True),
                fake_summary("tie-higher-f1", 41.5, 0.410, passed=True),
                fake_summary("outside-tie", 43.0, 0.900, passed=True),
                fake_summary("rejected", 20.0, 0.990, passed=False),
            ]
            paths = []
            for index, candidate in enumerate(candidates):
                path = root / f"candidate-{index}" / "benchmark_summary.json"
                path.parent.mkdir()
                path.write_text(json.dumps(candidate), encoding="utf-8")
                paths.append(path)
            comparison = root / "comparison.csv"
            selection_path = root / "selection.json"

            selection = BENCHMARK.aggregate_benchmark_summaries(
                paths,
                comparison_output=comparison,
                selection_output=selection_path,
            )

            self.assertEqual(
                selection["selected"]["backend"], "tie-higher-f1"
            )
            self.assertEqual(selection["eligible_count"], 3)
            self.assertEqual(len(selection["tie_candidate_paths"]), 2)
            self.assertEqual(
                len(comparison.read_text(encoding="utf-8").splitlines()),
                5,
            )
            self.assertTrue(selection_path.is_file())

    def test_aggregate_main_is_nonzero_without_accepted_candidate(self) -> None:
        selection = {
            "candidate_count": 1,
            "selection_status": "no_accepted_candidate",
            "selected": None,
        }
        with mock.patch.object(
            BENCHMARK,
            "aggregate_benchmark_summaries",
            return_value=selection,
        ):
            status = BENCHMARK.aggregate_main(
                ["--summaries", "summaries"]
            )

        self.assertEqual(status, 1)

    def test_official_rejected_run_returns_nonzero(self) -> None:
        backend = mock.Mock()
        summary = {
            "backend": "sgbm",
            "precision": "fp32",
            "timing_ms": {
                "pipeline_compute_ms": {
                    "p50": 70.0,
                    "p95": 75.0,
                    "p99": 80.0,
                }
            },
            "acceptance": {"status": "rejected", "passed": False},
        }
        with (
            mock.patch.object(
                BENCHMARK,
                "validate_model_provenance",
                return_value={"kind": "classical"},
            ),
            mock.patch.object(
                BENCHMARK, "create_backend", return_value=backend
            ),
            mock.patch.object(BENCHMARK, "benchmark", return_value=summary),
        ):
            status = BENCHMARK.main(
                [
                    "--backend",
                    "sgbm",
                    "--precision",
                    "fp32",
                    "--repeats",
                    "5",
                ]
            )

        self.assertEqual(status, 1)
        backend.close.assert_called_once_with()


def fake_summary(
    backend: str, p95: float, f1: float, *, passed: bool
) -> dict:
    return {
        "backend": backend,
        "precision": "fp32",
        "model_sha256": None,
        "configuration": {
            "protocol_schema": BENCHMARK.BENCHMARK_PROTOCOL_SCHEMA,
            "trips": list(BENCHMARK.TRIPS),
            "repeats": 5,
            "warmup_frames": 100,
            "max_frames_per_trip": None,
            "latency_target_ms": 75.0,
            "latency_comparison": "strict_less_than",
        },
        "dataset": {
            "frames_per_repeat": 3600,
            "runtime_rows": 18000,
            "protocol": {"complete": True},
        },
        "timing_ms": {
            "pipeline_compute_ms": {
                "p50": p95 - 2,
                "p95": p95,
                "p99": p95 + 2,
            }
        },
        "throughput": {"fps_from_pipeline_mean": 25.0},
        "evaluation": {
            "overall_composite_score": 28.7,
            "overall_f1": f1,
            "worst_trip_composite": 4.6,
        },
        "danger_confusion": {"tp": 1, "fp": 2, "fn": 3, "tn": 4},
        "resources": {
            "peak_process_ram_mb": 100.0,
            "peak_gpu_memory_mb": 1000.0,
        },
        "environment": {"processor": "test-cpu"},
        "acceptance": {
            "status": "accepted" if passed else "rejected",
            "passed": passed,
            "latency_target_ms": 75.0,
            "latency_comparison": "strict_less_than",
            "gates": {
                "pipeline_p95_strictly_below_target": passed,
                "exactly_five_repeats": True,
                "warmup_at_least_100_frames": True,
                "complete_six_trip_dataset": True,
                "exactly_3600_frames_per_repeat": True,
                "exactly_18000_runtime_rows": True,
                "repeat_determinism": True,
                "gpu_vram_le_5gb": True,
                "artifact_provenance_valid": True,
                "converted_backend_parity_passed": True,
                "stage2a_quality_budget": True,
                "lane_fp32_quality_budget": True,
            },
            "failure_reasons": [] if passed else ["latency"],
        },
    }


def fake_stereo_result(
    disparity: np.ndarray, *, valid: np.ndarray | None = None
) -> BENCHMARK.StereoResult:
    if valid is None:
        valid = np.isfinite(disparity) & (disparity > 0)
    return BENCHMARK.StereoResult(
        disparity_px=disparity,
        valid_mask=valid,
        confidence=None,
        backend="fake",
        precision="fp32",
        input_shape=(1, 3, disparity.shape[0], disparity.shape[1]),
        model_sha256=None,
        timings_ms={"stereo_total": 1.0},
    )


if __name__ == "__main__":
    unittest.main()
