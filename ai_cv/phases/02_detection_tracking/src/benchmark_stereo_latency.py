"""Reproducible decoded-pair-to-TTC latency benchmark for Phase 2B.

Examples:

  python ai_cv/phases/02_detection_tracking/src/benchmark_stereo_latency.py \
    --backend sgbm --precision fp32 --repeats 5

  python ai_cv/phases/02_detection_tracking/src/benchmark_stereo_latency.py \
    --backend lightstereo-onnx --precision fp32 --repeats 5 \
    --model-path ~/benchmarks/OpenStereo/artifacts/lightstereo_s_384x640.onnx

  python ai_cv/phases/02_detection_tracking/src/benchmark_stereo_latency.py \
    aggregate --summaries ai_cv/outputs/benchmarks/phase02b_latency

  python ai_cv/phases/02_detection_tracking/src/benchmark_stereo_latency.py \
    parity --manifest artifacts/lightstereo_parity_72.json \
    --data-root ~/guardian-data/phase02b/Practice_Dataset \
    --reference-model-path artifacts/lightstereo_s.ckpt \
    --candidate-backend lightstereo-onnx --candidate-precision fp32 \
    --candidate-model-path artifacts/lightstereo_s_384x640.onnx

Disk loading is measured and reported separately. The deployment gate uses
``pipeline_compute_ms``: stereo inference plus the unchanged Guardian ground,
component, tracking, and TTC stages. Its strict P95 target defaults to 75 ms
and can be changed explicitly with ``--latency-target-ms``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from classical_geometry import (
    collision_corridor_mask,
    estimate_ground_model,
    extract_obstacle_components,
    ground_and_obstacle_masks,
)
from classical_tracking import ComponentTracker, select_minimum_ttc
from stereo_backends import (
    BackendConfigurationError,
    StereoBackend,
    StereoResult,
    create_backend,
    disparity_parity,
    sha256_file,
)


TRIPS = tuple(f"T0{index}-Sample" for index in range(1, 7))
DEFAULT_PIPELINE_LATENCY_TARGET_MS = 75.0
OFFICIAL_REPEATS = 5
MINIMUM_WARMUP_FRAMES = 100
EXPECTED_FRAMES_PER_TRIP = 600
EXPECTED_FRAMES_PER_REPEAT = EXPECTED_FRAMES_PER_TRIP * len(TRIPS)
EXPECTED_RUNTIME_ROWS = EXPECTED_FRAMES_PER_REPEAT * OFFICIAL_REPEATS
BENCHMARK_PROTOCOL_SCHEMA = "guardian.phase02b.latency-protocol.v1"
ARTIFACT_MANIFEST_SCHEMA = "guardian.phase02b.model-artifact.v1"
PARITY_REPORT_KIND = "guardian.phase02b.lightstereo-parity.v1"
OPENSTEREO_REVISION = "23d71c92e33ad1f80dfc42bf29f5c6a914d38769"
OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256 = (
    "3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a"
)
LIGHTSTEREO_CONFIG_RELATIVE = "cfgs/lightstereo/lightstereo_s_kitti.yaml"
LIGHTSTEREO_INPUT_SHAPE = [1, 3, 384, 640]
LIGHTSTEREO_INPUT_NAMES = ["left_img", "right_img"]
LIGHTSTEREO_OUTPUT_NAME = "disp_pred"
GPU_MEMORY_LIMIT_MB = 5 * 1024
STAGE2A_COMPOSITE = 28.7
STAGE2A_DANGER_F1 = 0.402
MAX_COMPOSITE_LOSS = 0.5
MAX_F1_LOSS = 0.01
PARITY_MAX_MAE_PX = 0.25
PARITY_MAX_BAD_3PX_FRACTION = 0.005
PARITY_MAX_MISSING_VALID_FRACTION = 0.005


@dataclass(frozen=True)
class PipelineFrame:
    predicted_ttc: float
    stereo_result: StereoResult
    ground_confidence: float
    component_count: int
    relevant_component_count: int
    stereo_ms: float
    ground_ms: float
    components_ms: float
    tracking_ms: float
    pipeline_compute_ms: float


class GuardianTtcPipeline:
    """Guardian track-p35 post-processing behind a stereo adapter."""

    def __init__(
        self,
        backend: StereoBackend,
        image_shape: tuple[int, int],
        focal_length_px: float,
        baseline_m: float,
        ttc_policy: str = "baseline",
    ) -> None:
        if ttc_policy not in {
            "baseline",
            "guarded",
            "object-depth",
            "filtered-motion",
            "object-centric",
        }:
            raise ValueError(f"unknown TTC policy: {ttc_policy}")
        self.backend = backend
        self.image_shape = image_shape
        self.focal_length_px = focal_length_px
        self.baseline_m = baseline_m
        self.ttc_policy = ttc_policy
        guarded_policy = ttc_policy != "baseline"
        uses_object_depth = ttc_policy in {"object-depth", "object-centric"}
        uses_filtered_motion = ttc_policy in {
            "filtered-motion",
            "object-centric",
        }
        corridor_top = 0.10 if guarded_policy else 0.16
        corridor_bottom = 0.50 if guarded_policy else 0.55
        self.risk_corridor = collision_corridor_mask(
            image_shape,
            top_width_fraction=corridor_top,
            bottom_width_fraction=corridor_bottom,
        )
        self.tracker = ComponentTracker(
            image_shape,
            depth_attribute=(
                "object_depth_m"
                if uses_object_depth
                else "depth_p35_m"
            ),
            risk_top_width_fraction=corridor_top,
            risk_bottom_width_fraction=corridor_bottom,
            minimum_bottom_fraction=0.50 if guarded_policy else 0.0,
            minimum_height_fraction=0.05 if guarded_policy else 0.0,
            use_uncertainty_filter=uses_filtered_motion,
            include_predicted_tracks=uses_filtered_motion,
        )

    def process(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        timestamp: float,
    ) -> PipelineFrame:
        pipeline_started = time.perf_counter()
        stereo_started = time.perf_counter()
        stereo = self.backend.infer(left_bgr, right_bgr)
        stereo_ms = (time.perf_counter() - stereo_started) * 1000.0
        if stereo.disparity_px.shape != self.image_shape:
            raise BackendConfigurationError(
                f"{stereo.backend} returned {stereo.disparity_px.shape}; "
                f"Guardian requires native disparity {self.image_shape}"
            )

        started = time.perf_counter()
        ground_model, _ = estimate_ground_model(stereo.disparity_px)
        ground_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        components = []
        if ground_model is not None:
            _, obstacle_evidence, _ = ground_and_obstacle_masks(
                stereo.disparity_px, ground_model
            )
            support_mask = stereo.valid_mask
            if stereo.confidence is not None:
                support_mask = (
                    stereo.valid_mask
                    & np.isfinite(stereo.confidence)
                    & (stereo.confidence >= 0.5)
                )
            components, _, _ = extract_obstacle_components(
                stereo.disparity_px,
                obstacle_evidence,
                support_mask,
                self.focal_length_px,
                self.baseline_m,
                compute_object_depth=self.ttc_policy
                in {"object-depth", "object-centric"},
            )
        components_ms = (time.perf_counter() - started) * 1000.0
        ground_confidence = (
            float(ground_model.confidence) if ground_model is not None else 0.0
        )

        started = time.perf_counter()
        relevant_components = [
            component
            for component in components
            if component_in_risk_corridor(component, self.risk_corridor)
        ]
        current_tracks = self.tracker.update(components, timestamp)
        risk_tracks = self.tracker.risk_tracks(current_tracks)
        if self.ttc_policy in {"guarded", "object-depth"}:
            selection_options = {
                "minimum_track_confidence": 0.75,
                "maximum_closing_speed_mps": 20.0,
                "maximum_depth_m": 20.0,
                "maximum_motion_residual_m": 0.8,
            }
        elif self.ttc_policy in {"filtered-motion", "object-centric"}:
            selection_options = {
                "minimum_track_confidence": 0.68,
                "maximum_closing_speed_mps": 20.0,
                "maximum_depth_m": 20.0,
                "maximum_motion_residual_m": 1.2,
                "use_filtered_motion": True,
            }
        else:
            selection_options = {}
        predicted_ttc, _, _, _ = select_minimum_ttc(
            risk_tracks,
            ground_confidence,
            **selection_options,
        )
        tracking_ms = (time.perf_counter() - started) * 1000.0
        return PipelineFrame(
            predicted_ttc=float(predicted_ttc),
            stereo_result=stereo,
            ground_confidence=ground_confidence,
            component_count=len(components),
            relevant_component_count=len(relevant_components),
            stereo_ms=stereo_ms,
            ground_ms=ground_ms,
            components_ms=components_ms,
            tracking_ms=tracking_ms,
            pipeline_compute_ms=(time.perf_counter() - pipeline_started) * 1000.0,
        )


def component_in_risk_corridor(component: Any, corridor: np.ndarray) -> bool:
    height, width = corridor.shape
    center_x = int(np.clip(component.center_x, 0, width - 1))
    bottom_y = int(np.clip(component.bottom_y - 1, 0, height - 1))
    return bool(corridor[bottom_y, center_x])


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("cannot summarize an empty timing series")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("timing series must be finite and non-negative")
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def aggregate_runtime(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("no runtime rows were produced")
    timing_columns = [
        key
        for key in rows[0]
        if key.endswith("_ms")
        and all(key in row and isinstance(row[key], (int, float)) for row in rows)
    ]
    return {
        column: percentile_summary([float(row[column]) for row in rows])
        for column in timing_columns
    }


def danger_confusion(
    predictions_and_truth: Iterable[tuple[float, float]],
) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for prediction, truth in predictions_and_truth:
        predicted_danger = prediction < 2.0
        actual_danger = truth < 2.0
        if predicted_danger and actual_danger:
            tp += 1
        elif predicted_danger:
            fp += 1
        elif actual_danger:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


class ParityAccumulator:
    """Exact pixel-weighted aggregation for the frozen conversion pre-gate."""

    def __init__(self) -> None:
        self._error_chunks: list[np.ndarray] = []
        self._reference_valid_pixels = 0
        self._missing_reference_valid_pixels = 0
        self._additional_valid_pixels = 0
        self._total_pixels = 0
        self.per_frame: list[dict[str, Any]] = []

    def add(
        self,
        *,
        trip_id: str,
        frame_id: int,
        reference: StereoResult,
        candidate: StereoResult,
    ) -> None:
        frame_metrics = disparity_parity(reference, candidate)
        shared = reference.valid_mask & candidate.valid_mask
        errors = np.abs(
            reference.disparity_px[shared] - candidate.disparity_px[shared]
        ).astype(np.float32, copy=False)
        self._error_chunks.append(errors)
        reference_valid = int(np.count_nonzero(reference.valid_mask))
        missing = int(
            np.count_nonzero(reference.valid_mask & ~candidate.valid_mask)
        )
        additional = int(
            np.count_nonzero(candidate.valid_mask & ~reference.valid_mask)
        )
        self._reference_valid_pixels += reference_valid
        self._missing_reference_valid_pixels += missing
        self._additional_valid_pixels += additional
        self._total_pixels += int(reference.disparity_px.size)
        self.per_frame.append(
            {
                "trip_id": trip_id,
                "frame_id": int(frame_id),
                **frame_metrics,
                "reference_stereo_ms": reference.timings_ms.get(
                    "stereo_total"
                ),
                "candidate_stereo_ms": candidate.timings_ms.get(
                    "stereo_total"
                ),
            }
        )

    def finalize(self, *, expected_frames: int) -> dict[str, Any]:
        if len(self.per_frame) != expected_frames:
            raise BackendConfigurationError(
                f"parity gate requires {expected_frames} frames; "
                f"processed {len(self.per_frame)}"
            )
        if not self._error_chunks:
            raise BackendConfigurationError("parity gate produced no errors")
        errors = np.concatenate(self._error_chunks)
        aggregate = {
            "frame_count": len(self.per_frame),
            "compared_pixels": int(errors.size),
            "mean_absolute_error_px": float(np.mean(errors)),
            "p95_absolute_error_px": float(np.percentile(errors, 95)),
            "maximum_absolute_error_px": float(np.max(errors)),
            "bad_1px_fraction": float(np.mean(errors > 1.0)),
            "bad_3px_fraction": float(np.mean(errors > 3.0)),
            "missing_reference_valid_fraction": float(
                self._missing_reference_valid_pixels
                / max(1, self._reference_valid_pixels)
            ),
            "additional_valid_fraction": float(
                self._additional_valid_pixels / max(1, self._total_pixels)
            ),
        }
        gates = {
            "mean_absolute_error_px": (
                aggregate["mean_absolute_error_px"] <= PARITY_MAX_MAE_PX
            ),
            "bad_3px_fraction": (
                aggregate["bad_3px_fraction"]
                <= PARITY_MAX_BAD_3PX_FRACTION
            ),
            "missing_reference_valid_fraction": (
                aggregate["missing_reference_valid_fraction"]
                <= PARITY_MAX_MISSING_VALID_FRACTION
            ),
            "all_72_frames": len(self.per_frame) == expected_frames,
        }
        passed = all(gates.values())
        return {
            "aggregate": aggregate,
            "thresholds": {
                "maximum_mean_absolute_error_px": PARITY_MAX_MAE_PX,
                "maximum_bad_3px_fraction": PARITY_MAX_BAD_3PX_FRACTION,
                "maximum_missing_reference_valid_fraction": (
                    PARITY_MAX_MISSING_VALID_FRACTION
                ),
            },
            "gates": gates,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "failure_reasons": [
                name for name, gate_passed in gates.items() if not gate_passed
            ],
            "per_frame": self.per_frame,
        }


def run_parity_gate(
    *,
    reference_backend: StereoBackend,
    candidate_backend: StereoBackend,
    manifest_path: Path,
    data_root: Path,
    output_path: Path,
    warmup_frames: int,
    progress_every: int,
) -> dict[str, Any]:
    """Run the frozen 72-pair learned-engine conversion pre-gate."""
    from lightstereo_deployment import (
        PARITY_KIND,
        PARITY_TOTAL,
        resolve_manifest_pairs,
    )

    if (
        reference_backend.name != "lightstereo-pytorch"
        or reference_backend.precision != "fp32"
    ):
        raise BackendConfigurationError(
            "parity reference must be lightstereo-pytorch/fp32"
        )
    manifest, resolved_pairs = resolve_manifest_pairs(
        manifest_path,
        data_root,
        expected_kind=PARITY_KIND,
        expected_count=PARITY_TOTAL,
    )
    entries = manifest["entries"]

    def load_images(index: int) -> tuple[np.ndarray, np.ndarray]:
        left_path, right_path = resolved_pairs[index]
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise BackendConfigurationError(
                f"cannot decode parity pair {left_path}, {right_path}"
            )
        return left, right

    if warmup_frames:
        left, right = load_images(0)
        for _ in range(warmup_frames):
            reference_backend.infer(left, right)
            candidate_backend.infer(left, right)

    accumulator = ParityAccumulator()
    for index, entry in enumerate(entries):
        left, right = load_images(index)
        try:
            reference = reference_backend.infer(left, right)
            candidate = candidate_backend.infer(left, right)
        except Exception as error:
            raise BackendConfigurationError(
                "parity inference failed at "
                f"{entry['trip_id']} frame {entry['frame_id']}: {error}"
            ) from error
        accumulator.add(
            trip_id=str(entry["trip_id"]),
            frame_id=int(entry["frame_id"]),
            reference=reference,
            candidate=candidate,
        )
        if progress_every and (
            index % progress_every == 0 or index + 1 == len(entries)
        ):
            print(
                f"parity {index + 1}/{len(entries)} "
                f"{entry['trip_id']} #{entry['frame_id']}",
                flush=True,
            )

    gate = accumulator.finalize(expected_frames=PARITY_TOTAL)
    repository_root = Path(__file__).resolve().parents[4]
    report = {
        "schema_version": 1,
        "kind": "guardian.phase02b.lightstereo-parity.v1",
        "manifest": {
            "path": str(manifest_path.expanduser().resolve()),
            "sha256": sha256_file(manifest_path.expanduser().resolve()),
            "entries_sha256": manifest["entries_sha256"],
            "entry_count": manifest["entry_count"],
        },
        "reference": {
            "backend": reference_backend.name,
            "precision": reference_backend.precision,
            "model_sha256": reference_backend.model_sha256,
        },
        "candidate": {
            "backend": candidate_backend.name,
            "precision": candidate_backend.precision,
            "model_sha256": candidate_backend.model_sha256,
        },
        **gate,
        "environment": environment_metadata(repository_root),
    }
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def _prediction_equal(left: float, right: float) -> bool:
    if math.isinf(left) and math.isinf(right):
        return True
    return left == right


def _format_ttc(value: float) -> str | float:
    return "inf" if not math.isfinite(value) else round(value, 6)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _peak_rss_mb() -> float | None:
    """Read peak resident memory without adding a psutil dependency."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            succeeded = psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            if succeeded:
                return float(counters.PeakWorkingSetSize / (1024 * 1024))
        except (AttributeError, OSError):
            return None
        return None
    try:
        import resource

        maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
        return maximum / divisor
    except (ImportError, OSError):
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
        return completed.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _parse_nvidia_smi_process_memory(
    output: str, process_id: int
) -> float | None:
    matches = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            pid = int(fields[0])
            memory_mb = float(fields[1].split()[0])
        except (ValueError, IndexError):
            continue
        if pid == process_id:
            matches.append(memory_mb)
    return max(matches) if matches else None


def _parse_nvidia_smi_device_memory(output: str) -> float | None:
    for line in output.splitlines():
        try:
            return float(line.strip().split()[0])
        except (ValueError, IndexError):
            continue
    return None


class ProcessGpuMemorySampler:
    """Poll total GPU memory owned by this process, including engine workspaces."""

    def __init__(self, device_id: int, poll_interval_seconds: float = 0.10) -> None:
        self.device_id = int(device_id)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.process_id = os.getpid()
        self.source: str | None = None
        self.error: str | None = None
        self._nvml = None
        self._nvml_handle = None
        self._maximum_mb: float | None = None
        self._used_device_wide_fallback = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        try:
            nvml = importlib.import_module("pynvml")
            nvml.nvmlInit()
            self._nvml_handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self._nvml = nvml
            self.source = "nvml_process_used_gpu_memory"
        except Exception as error:
            self.error = str(error)
            if _command_output(["nvidia-smi", "-L"]) is not None:
                self.source = "nvidia_smi_process_observed_peak"

    @property
    def available(self) -> bool:
        return self.source is not None

    def _sample_nvml(self) -> float:
        assert self._nvml is not None
        functions = [
            getattr(
                self._nvml,
                "nvmlDeviceGetComputeRunningProcesses_v3",
                None,
            ),
            getattr(
                self._nvml,
                "nvmlDeviceGetComputeRunningProcesses_v2",
                None,
            ),
            getattr(
                self._nvml,
                "nvmlDeviceGetComputeRunningProcesses",
                None,
            ),
        ]
        function = next((item for item in functions if item is not None), None)
        values = []
        if function is None:
            self.error = "pynvml has no compute-process query"
        else:
            try:
                processes = function(self._nvml_handle)
                values = [
                    float(process.usedGpuMemory / (1024 * 1024))
                    for process in processes
                    if int(process.pid) == self.process_id
                    and getattr(process, "usedGpuMemory", None) is not None
                    and int(process.usedGpuMemory) >= 0
                ]
            except Exception as error:
                self.error = f"NVML process query unavailable: {error}"
        if values:
            if not self._used_device_wide_fallback:
                self.source = "nvml_process_used_gpu_memory"
            return max(values)
        memory = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        self._used_device_wide_fallback = True
        self.source = "nvml_device_memory_used_conservative"
        return float(memory.used / (1024 * 1024))

    def _sample_nvidia_smi(self) -> float | None:
        output = _command_output(
            [
                "nvidia-smi",
                f"--id={self.device_id}",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        process_memory = (
            _parse_nvidia_smi_process_memory(output, self.process_id)
            if output is not None
            else None
        )
        if process_memory is not None:
            if not self._used_device_wide_fallback:
                self.source = "nvidia_smi_process_observed_peak"
            return process_memory
        device_output = _command_output(
            [
                "nvidia-smi",
                f"--id={self.device_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        device_memory = (
            _parse_nvidia_smi_device_memory(device_output)
            if device_output is not None
            else None
        )
        if device_memory is not None:
            self._used_device_wide_fallback = True
            self.source = "nvidia_smi_device_memory_used_conservative"
        return device_memory

    def sample_now(self) -> float | None:
        if self.source is None:
            return None
        try:
            value = (
                self._sample_nvml()
                if self._nvml is not None
                else self._sample_nvidia_smi()
            )
        except Exception as error:
            self.error = str(error)
            return None
        if value is not None:
            with self._lock:
                self._maximum_mb = max(self._maximum_mb or 0.0, value)
        return value

    def _poll(self) -> None:
        while not self._stop_event.wait(self.poll_interval_seconds):
            self.sample_now()

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self.sample_now()
        self._thread = threading.Thread(
            target=self._poll,
            name="gpu-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> float | None:
        self.sample_now()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_interval_seconds * 2))
            self._thread = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
        return self.maximum_mb

    @property
    def maximum_mb(self) -> float | None:
        with self._lock:
            return self._maximum_mb


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata(repository_root: Path) -> dict[str, Any]:
    git_revision = _command_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"]
    )
    git_status = _command_output(
        ["git", "-C", str(repository_root), "status", "--short"]
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "dependencies": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": _package_version("torch"),
            "onnx": _package_version("onnx"),
            "onnxruntime-gpu": _package_version("onnxruntime-gpu"),
            "tensorrt": _package_version("tensorrt"),
            "nvidia-ml-py": _package_version("nvidia-ml-py"),
        },
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "git_revision": git_revision,
        "git_dirty": bool(git_status),
    }


def _load_dataset_class(starter_root: Path) -> Any:
    resolved = starter_root.expanduser().resolve()
    if not resolved.is_dir():
        raise BackendConfigurationError(
            f"starter-kit root not found: {resolved}"
        )
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    try:
        from team_kit.dataset_loader import TripDataset
    except ImportError as error:
        raise BackendConfigurationError(
            f"cannot import team_kit.dataset_loader from {resolved}: {error}"
        ) from error
    return TripDataset


def _trip_context(
    dataset: Any,
    backend: StereoBackend,
    ttc_policy: str = "baseline",
) -> GuardianTtcPipeline:
    calibration = dataset.load_calibration()
    image_shape = (
        int(calibration["image_height"]),
        int(calibration["image_width"]),
    )
    return GuardianTtcPipeline(
        backend,
        image_shape,
        float(calibration["K_left"][0][0]),
        float(calibration["baseline_m"]),
        ttc_policy,
    )


def warm_up_backend(
    backend: StereoBackend,
    trip_dataset_class: Any,
    practice_root: Path,
    trip_id: str,
    frame_count: int,
    ttc_policy: str = "baseline",
) -> None:
    if frame_count == 0:
        return
    dataset = trip_dataset_class(practice_root / trip_id)
    frames = list(dataset.iter_frames())
    if not frames:
        raise BackendConfigurationError(f"trip {trip_id} has no frames")
    pipeline = _trip_context(dataset, backend, ttc_policy)
    for index in range(frame_count):
        frame = frames[index % len(frames)]
        left = dataset.load_left(frame.frame_id)
        right = dataset.load_right(frame.frame_id)
        pipeline.process(left, right, frame.timestamp)


def _evaluate_predictions(
    predictions_root: Path,
    practice_root: Path,
    starter_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    if str(starter_root.resolve()) not in sys.path:
        sys.path.insert(0, str(starter_root.resolve()))
    from team_kit.evaluation import evaluate

    report = evaluate(predictions_root, practice_root, report_path)
    per_trip = [asdict(metric) for metric in report.per_trip]
    return {
        "overall_mae_critical": report.overall_mae_critical,
        "overall_inv_ttc_mae": report.overall_inv_ttc_mae,
        "overall_f1": report.overall_f1,
        "overall_composite_score": report.overall_composite_score,
        "worst_trip_composite": min(
            metric.composite_score for metric in report.per_trip
        ),
        "per_trip": per_trip,
    }


def _read_json_document(path: Path, description: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackendConfigurationError(
            f"cannot read {description} {resolved}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise BackendConfigurationError(
            f"{description} {resolved} must contain a JSON object"
        )
    return document


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_dataset_protocol(
    trip_dataset_class: Any,
    practice_root: Path,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    """Freeze the six-trip frame order instead of trusting observed counts."""
    trip_records: dict[str, Any] = {}
    frame_ids_by_trip: dict[str, tuple[int, ...]] = {}
    issues: list[str] = []
    fingerprint_input: list[dict[str, Any]] = []
    for trip_id in TRIPS:
        dataset = trip_dataset_class(practice_root / trip_id)
        frames = list(dataset.iter_frames())
        frame_ids = tuple(int(frame.frame_id) for frame in frames)
        frame_ids_by_trip[trip_id] = frame_ids
        unique = len(set(frame_ids)) == len(frame_ids)
        ordered = list(frame_ids) == sorted(frame_ids)
        exact_count = len(frame_ids) == EXPECTED_FRAMES_PER_TRIP
        if not exact_count:
            issues.append(
                f"{trip_id} has {len(frame_ids)} frames, expected "
                f"{EXPECTED_FRAMES_PER_TRIP}"
            )
        if not unique:
            issues.append(f"{trip_id} contains duplicate frame IDs")
        if not ordered:
            issues.append(f"{trip_id} frame IDs are not monotonically ordered")
        descriptors = [
            {
                "frame_id": int(frame.frame_id),
                "timestamp": float(frame.timestamp),
                "min_ttc": float(frame.min_ttc),
            }
            for frame in frames
        ]
        trip_records[trip_id] = {
            "frame_count": len(frame_ids),
            "unique_frame_ids": unique,
            "ordered_frame_ids": ordered,
            "frame_ids_sha256": _canonical_sha256(frame_ids),
        }
        fingerprint_input.append(
            {
                "trip_id": trip_id,
                "frames": descriptors,
                "calibration": _json_safe(dataset.load_calibration()),
            }
        )
    protocol = {
        "schema": BENCHMARK_PROTOCOL_SCHEMA,
        "trip_ids": list(TRIPS),
        "expected_frames_per_trip": EXPECTED_FRAMES_PER_TRIP,
        "expected_frames_per_repeat": EXPECTED_FRAMES_PER_REPEAT,
        "trips": trip_records,
        "complete": not issues,
        "issues": issues,
        "dataset_fingerprint_sha256": _canonical_sha256(fingerprint_input),
    }
    return protocol, frame_ids_by_trip


def _validate_openstereo_checkout(
    root: Path,
    config_path: Path | None,
) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    expected_config = (resolved / LIGHTSTEREO_CONFIG_RELATIVE).resolve()
    selected_config = (
        config_path.expanduser().resolve()
        if config_path is not None
        else expected_config
    )
    if selected_config != expected_config:
        raise BackendConfigurationError(
            "--config-path must select the pinned config inside OpenStereo: "
            f"{expected_config}"
        )
    if not selected_config.is_file():
        raise BackendConfigurationError(
            f"pinned LightStereo-S config is missing: {selected_config}"
        )
    revision = _command_output(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"]
    )
    if revision != OPENSTEREO_REVISION:
        raise BackendConfigurationError(
            "OpenStereo must be checked out at "
            f"{OPENSTEREO_REVISION}; found {revision!r}"
        )
    status = _command_output(
        ["git", "-C", str(resolved), "status", "--porcelain"]
    )
    if status is None:
        raise BackendConfigurationError(
            f"cannot verify the OpenStereo worktree at {resolved}"
        )
    if status:
        raise BackendConfigurationError(
            "OpenStereo worktree must be clean before a learned benchmark"
        )
    return {
        "openstereo_revision": revision,
        "config_path": LIGHTSTEREO_CONFIG_RELATIVE,
        "config_sha256": sha256_file(selected_config),
    }


def validate_model_provenance(
    *,
    backend: str,
    precision: str,
    model_path: Path | None,
    openstereo_root: Path,
    config_path: Path | None,
    artifact_manifest_path: Path | None = None,
    calibration_manifest_path: Path | None = None,
    calibration_cache_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the existing deployment sidecar and normalize its lineage."""
    if backend == "sgbm":
        return {"kind": "classical", "backend": backend, "precision": precision}
    if model_path is None:
        raise BackendConfigurationError(
            f"--model-path is required for backend {backend}"
        )
    artifact = model_path.expanduser().resolve()
    if not artifact.is_file():
        raise BackendConfigurationError(f"model artifact not found: {artifact}")
    checkout = _validate_openstereo_checkout(openstereo_root, config_path)
    artifact_sha = sha256_file(artifact)
    if backend == "lightstereo-pytorch":
        if precision != "fp32":
            raise BackendConfigurationError(
                "the frozen learned reference must use "
                "lightstereo-pytorch/fp32"
            )
        if artifact_sha != OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256:
            raise BackendConfigurationError(
                "LightStereo-S checkpoint SHA-256 mismatch; expected "
                f"{OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256}, got {artifact_sha}"
            )
        return {
            "kind": "official-lightstereo-s-checkpoint",
            "backend": backend,
            "precision": precision,
            "artifact_sha256": artifact_sha,
            "source_checkpoint_sha256": artifact_sha,
            **checkout,
        }

    manifest_path = (
        artifact_manifest_path.expanduser().resolve()
        if artifact_manifest_path is not None
        else artifact.with_name(f"{artifact.name}.manifest.json")
    )
    manifest = _read_json_document(manifest_path, "artifact manifest")
    if manifest.get("schema") != ARTIFACT_MANIFEST_SCHEMA:
        raise BackendConfigurationError(
            f"{manifest_path}: unsupported artifact-manifest schema"
        )
    artifact_record = manifest.get("artifact")
    metadata = manifest.get("metadata")
    command = manifest.get("generation_command_argv")
    if not isinstance(artifact_record, dict) or not isinstance(metadata, dict):
        raise BackendConfigurationError(
            f"{manifest_path}: artifact and metadata objects are required"
        )
    if not isinstance(command, list) or not command:
        raise BackendConfigurationError(
            f"{manifest_path}: exact generation command is missing"
        )
    if (
        artifact_record.get("name") != artifact.name
        or artifact_record.get("bytes") != artifact.stat().st_size
        or artifact_record.get("sha256") != artifact_sha
    ):
        raise BackendConfigurationError(
            f"{manifest_path}: artifact name/size/SHA-256 does not match "
            f"{artifact}"
        )
    if metadata.get("openstereo_revision") != OPENSTEREO_REVISION:
        raise BackendConfigurationError(
            f"{manifest_path}: wrong OpenStereo revision"
        )
    if (
        metadata.get("input_shape_nchw") != LIGHTSTEREO_INPUT_SHAPE
        or metadata.get("input_names") != LIGHTSTEREO_INPUT_NAMES
        or metadata.get("output_name") != LIGHTSTEREO_OUTPUT_NAME
    ):
        raise BackendConfigurationError(
            f"{manifest_path}: static input/output contract mismatch"
        )
    checkpoint = metadata.get("checkpoint")
    config = metadata.get("config")
    if (
        metadata.get("backend") != backend
        or metadata.get("precision") != precision
    ):
        raise BackendConfigurationError(
            f"{manifest_path}: declared backend/precision does not match "
            f"{backend}/{precision}"
        )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("sha256")
        != OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
        or not isinstance(config, dict)
        or config.get("relative_path") != LIGHTSTEREO_CONFIG_RELATIVE
        or config.get("sha256") != checkout["config_sha256"]
    ):
        raise BackendConfigurationError(
            f"{manifest_path}: checkpoint/config provenance does not match "
            "the pinned clean OpenStereo checkout"
        )

    common = {
        "backend": backend,
        "precision": precision,
        "artifact_sha256": artifact_sha,
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "generation_command_argv": command,
        **checkout,
    }
    if backend == "lightstereo-onnx":
        if precision != "fp32":
            raise BackendConfigurationError(
                "the generated ONNX lane supports only declared fp32"
            )
        if (
            artifact_record.get("kind")
            != "lightstereo-s-onnx-opset17-static"
            or metadata.get("opset") != 17
            or metadata.get("dynamic_axes") is not False
            or metadata.get("openstereo_tracked_tree_clean") is not True
        ):
            raise BackendConfigurationError(
                f"{manifest_path}: ONNX provenance contract mismatch"
            )
        return {
            "kind": "generated-lightstereo-onnx",
            "source_checkpoint_sha256": checkpoint["sha256"],
            **common,
        }

    if backend != "lightstereo-tensorrt":
        raise BackendConfigurationError(f"unknown learned backend {backend!r}")
    if precision not in {"fp16", "int8"}:
        raise BackendConfigurationError(
            "generated TensorRT engines must be declared fp16 or int8"
        )
    if (
        artifact_record.get("kind")
        != f"lightstereo-s-tensorrt10-{precision}"
        or metadata.get("precision") != precision
    ):
        raise BackendConfigurationError(
            f"{manifest_path}: TensorRT declared precision does not match "
            f"{precision}"
        )
    dependencies = metadata.get("dependencies", {})
    if not str(dependencies.get("tensorrt", "")).startswith("10."):
        raise BackendConfigurationError(
            f"{manifest_path}: TensorRT 10.x build provenance is required"
        )
    source = metadata.get("source_onnx")
    if not isinstance(source, dict) or not isinstance(source.get("name"), str):
        raise BackendConfigurationError(
            f"{manifest_path}: source ONNX provenance is missing"
        )
    source_onnx = artifact.parent / source["name"]
    source_sidecar = source_onnx.with_name(f"{source_onnx.name}.manifest.json")
    if (
        not source_onnx.is_file()
        or sha256_file(source_onnx) != source.get("sha256")
        or source.get("manifest_name") != source_sidecar.name
        or not source_sidecar.is_file()
        or sha256_file(source_sidecar) != source.get("manifest_sha256")
    ):
        raise BackendConfigurationError(
            f"{manifest_path}: source ONNX file/hash is unavailable; place "
            f"{source['name']} beside the engine"
        )
    source_provenance = validate_model_provenance(
        backend="lightstereo-onnx",
        precision="fp32",
        model_path=source_onnx,
        openstereo_root=openstereo_root,
        config_path=config_path,
    )
    calibration = metadata.get("calibration")
    if precision == "fp16" and calibration is not None:
        raise BackendConfigurationError(
            f"{manifest_path}: FP16 engine must not claim INT8 calibration"
        )
    calibration_provenance = None
    if precision == "int8":
        if not isinstance(calibration, dict):
            raise BackendConfigurationError(
                f"{manifest_path}: INT8 calibration provenance is required"
            )
        manifest_record = calibration.get("manifest", {})
        cache_record = calibration.get("cache", {})
        selected_manifest = (
            calibration_manifest_path.expanduser().resolve()
            if calibration_manifest_path is not None
            else artifact.parent / str(manifest_record.get("name", ""))
        )
        selected_cache = (
            calibration_cache_path.expanduser().resolve()
            if calibration_cache_path is not None
            else artifact.parent / str(cache_record.get("name", ""))
        )
        if (
            not selected_manifest.is_file()
            or sha256_file(selected_manifest) != manifest_record.get("sha256")
        ):
            raise BackendConfigurationError(
                "INT8 calibration manifest is missing or its SHA-256 changed; "
                "pass --calibration-manifest"
            )
        pair_manifest = _read_json_document(
            selected_manifest, "INT8 calibration manifest"
        )
        if (
            pair_manifest.get("kind")
            != "lightstereo-tensorrt-int8-calibration"
            or pair_manifest.get("entry_count") != 300
            or pair_manifest.get("entries_sha256")
            != manifest_record.get("entries_sha256")
            or pair_manifest.get("content_sha256")
            != manifest_record.get("content_sha256")
        ):
            raise BackendConfigurationError(
                "INT8 calibration manifest selection/content provenance "
                "does not match the engine sidecar"
            )
        if (
            not selected_cache.is_file()
            or sha256_file(selected_cache) != cache_record.get("sha256")
        ):
            raise BackendConfigurationError(
                "INT8 calibration cache is missing or its SHA-256 changed; "
                "pass --calibration-cache"
            )
        cache_sidecar = selected_cache.with_name(
            f"{selected_cache.name}.manifest.json"
        )
        if _read_json_document(
            cache_sidecar, "INT8 calibration-cache manifest"
        ) != cache_record.get("expected_metadata"):
            raise BackendConfigurationError(
                "INT8 calibration-cache metadata no longer matches the engine"
            )
        expected_metadata = cache_record.get("expected_metadata", {})
        if (
            expected_metadata.get("onnx_sha256") != source.get("sha256")
            or expected_metadata.get("batch_count") != 300
            or expected_metadata.get("input_shape_nchw")
            != LIGHTSTEREO_INPUT_SHAPE
            or expected_metadata.get("input_names") != LIGHTSTEREO_INPUT_NAMES
        ):
            raise BackendConfigurationError(
                "INT8 cache was produced for a different model/input contract"
            )
        calibration_provenance = {
            "manifest_sha256": manifest_record["sha256"],
            "entries_sha256": manifest_record["entries_sha256"],
            "content_sha256": manifest_record["content_sha256"],
            "cache_sha256": cache_record["sha256"],
        }
    return {
        "kind": f"generated-lightstereo-tensorrt-{precision}",
        "source_onnx_sha256": source["sha256"],
        "source_checkpoint_sha256": source_provenance[
            "source_checkpoint_sha256"
        ],
        "calibration": calibration_provenance,
        **common,
    }


def validate_parity_report(
    path: Path,
    *,
    backend: str,
    precision: str,
    model_sha256: str,
    model_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    report = _read_json_document(resolved, "parity report")
    candidate = report.get("candidate", {})
    reference = report.get("reference", {})
    aggregate = report.get("aggregate", {})
    gates = report.get("gates", {})
    if (
        report.get("kind") != PARITY_REPORT_KIND
        or report.get("passed") is not True
        or report.get("status") != "passed"
        or aggregate.get("frame_count") != 72
        or report.get("manifest", {}).get("entry_count") != 72
        or gates.get("all_72_frames") is not True
        or any(value is not True for value in gates.values())
    ):
        raise BackendConfigurationError(
            f"{resolved}: frozen 72-pair parity gate has not passed"
        )
    if (
        candidate.get("backend") != backend
        or candidate.get("precision") != precision
        or candidate.get("model_sha256") != model_sha256
    ):
        raise BackendConfigurationError(
            f"{resolved}: parity candidate does not match the current "
            f"{backend}/{precision} artifact SHA-256"
        )
    expected_checkpoint = model_provenance.get("source_checkpoint_sha256")
    if (
        reference.get("backend") != "lightstereo-pytorch"
        or reference.get("precision") != "fp32"
        or reference.get("model_sha256") != expected_checkpoint
        or expected_checkpoint != OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
    ):
        raise BackendConfigurationError(
            f"{resolved}: parity reference is not the matching official "
            "LightStereo-S PyTorch FP32 checkpoint"
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "candidate_model_sha256": model_sha256,
        "manifest_sha256": report["manifest"].get("sha256"),
        "passed": True,
    }


def load_lane_reference_summary(
    path: Path,
    *,
    dataset_protocol: Mapping[str, Any],
    model_provenance: Mapping[str, Any],
    latency_target_ms: float,
) -> tuple[tuple[float, float], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    document = _read_json_document(resolved, "lane FP32 summary")
    configuration = document.get("configuration", {})
    dataset = document.get("dataset", {})
    provenance = document.get("model_provenance", {})
    acceptance_gates = document.get("acceptance", {}).get("gates", {})
    required_protocol_gates = (
        "exactly_five_repeats",
        "warmup_at_least_100_frames",
        "complete_six_trip_dataset",
        "exactly_3600_frames_per_repeat",
        "exactly_18000_runtime_rows",
        "repeat_determinism",
        "artifact_provenance_valid",
    )
    if (
        document.get("backend") != "lightstereo-pytorch"
        or document.get("precision") != "fp32"
    ):
        raise BackendConfigurationError(
            f"{resolved}: lane reference must be lightstereo-pytorch/fp32"
        )
    if (
        configuration.get("trips") != list(TRIPS)
        or configuration.get("protocol_schema") != BENCHMARK_PROTOCOL_SCHEMA
        or configuration.get("repeats") != OFFICIAL_REPEATS
        or int(configuration.get("warmup_frames", -1))
        < MINIMUM_WARMUP_FRAMES
        or configuration.get("max_frames_per_trip") is not None
        or configuration.get("latency_target_ms") != latency_target_ms
        or configuration.get("latency_comparison") != "strict_less_than"
        or dataset.get("frames_per_repeat") != EXPECTED_FRAMES_PER_REPEAT
        or dataset.get("runtime_rows") != EXPECTED_RUNTIME_ROWS
        or dataset.get("protocol", {}).get("complete") is not True
        or dataset.get("protocol", {}).get("dataset_fingerprint_sha256")
        != dataset_protocol.get("dataset_fingerprint_sha256")
        or any(
            acceptance_gates.get(name) is not True
            for name in required_protocol_gates
        )
    ):
        raise BackendConfigurationError(
            f"{resolved}: lane reference does not use the complete frozen "
            "six-trip, five-repeat protocol or matching dataset"
        )
    if (
        document.get("model_sha256")
        != OFFICIAL_LIGHTSTEREO_CHECKPOINT_SHA256
        or
        provenance.get("source_checkpoint_sha256")
        != model_provenance.get("source_checkpoint_sha256")
        or provenance.get("config_sha256")
        != model_provenance.get("config_sha256")
        or provenance.get("openstereo_revision") != OPENSTEREO_REVISION
    ):
        raise BackendConfigurationError(
            f"{resolved}: lane reference checkpoint/config provenance does "
            "not match the converted artifact"
        )
    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        raise BackendConfigurationError(
            f"{resolved}: lane reference lacks a full official evaluation"
        )
    try:
        quality = (
            float(evaluation["overall_composite_score"]),
            float(evaluation["overall_f1"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BackendConfigurationError(
            f"{resolved}: invalid lane reference quality metrics"
        ) from error
    return quality, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "model_sha256": document.get("model_sha256"),
        "dataset_fingerprint_sha256": dataset_protocol.get(
            "dataset_fingerprint_sha256"
        ),
    }


def acceptance_report(
    *,
    timing: dict[str, dict[str, float]],
    evaluation: dict[str, Any] | None,
    peak_gpu_memory_mb: float | None,
    frames_measured: int,
    expected_frames: int,
    nondeterministic_predictions: int,
    lane_reference: tuple[float, float] | None,
    latency_target_ms: float = DEFAULT_PIPELINE_LATENCY_TARGET_MS,
    repeats: int = OFFICIAL_REPEATS,
    warmup_frames: int = MINIMUM_WARMUP_FRAMES,
    complete_dataset_protocol: bool = True,
    runtime_rows_measured: int = EXPECTED_RUNTIME_ROWS,
    expected_runtime_rows: int = EXPECTED_RUNTIME_ROWS,
    parity_passed: bool = True,
    provenance_valid: bool = True,
) -> dict[str, Any]:
    gates: dict[str, bool | None] = {
        "pipeline_p95_strictly_below_target": (
            timing["pipeline_compute_ms"]["p95"] < latency_target_ms
        ),
        "exactly_five_repeats": repeats == OFFICIAL_REPEATS,
        "warmup_at_least_100_frames": (
            warmup_frames >= MINIMUM_WARMUP_FRAMES
        ),
        "complete_six_trip_dataset": complete_dataset_protocol,
        "exactly_3600_frames_per_repeat": (
            frames_measured == expected_frames == EXPECTED_FRAMES_PER_REPEAT
        ),
        "exactly_18000_runtime_rows": (
            runtime_rows_measured
            == expected_runtime_rows
            == EXPECTED_RUNTIME_ROWS
        ),
        "repeat_determinism": nondeterministic_predictions == 0,
        "gpu_vram_le_5gb": (
            peak_gpu_memory_mb is not None
            and peak_gpu_memory_mb <= GPU_MEMORY_LIMIT_MB
        ),
        "artifact_provenance_valid": provenance_valid,
        "converted_backend_parity_passed": parity_passed,
        "stage2a_quality_budget": None,
        "lane_fp32_quality_budget": None if lane_reference is not None else True,
    }
    if evaluation is not None:
        candidate_composite = float(evaluation["overall_composite_score"])
        candidate_f1 = float(evaluation["overall_f1"])
        gates["stage2a_quality_budget"] = (
            STAGE2A_COMPOSITE - candidate_composite <= MAX_COMPOSITE_LOSS
            and STAGE2A_DANGER_F1 - candidate_f1 <= MAX_F1_LOSS
        )
        if lane_reference is not None:
            reference_composite, reference_f1 = lane_reference
            gates["lane_fp32_quality_budget"] = (
                reference_composite - candidate_composite <= MAX_COMPOSITE_LOSS
                and reference_f1 - candidate_f1 <= MAX_F1_LOSS
            )
    reasons = [
        name for name, passed in gates.items() if passed is False
    ]
    if peak_gpu_memory_mb is None:
        reasons.append(
            "GPU VRAM peak unavailable; use PyTorch/TensorRT allocation telemetry "
            "or add an NVML sampler before accepting this candidate"
        )
    if evaluation is None:
        return {
            "status": "not_evaluated",
            "passed": None,
            "latency_target_ms": latency_target_ms,
            "latency_comparison": "strict_less_than",
            "gates": gates,
            "failure_reasons": reasons,
            "pending_reasons": [
                "partial/smoke runs are never deployment-eligible; quality "
                "gates require the complete official protocol"
            ],
        }
    passed = all(value is True for value in gates.values())
    return {
        "status": "accepted" if passed else "rejected",
        "passed": passed,
        "latency_target_ms": latency_target_ms,
        "latency_comparison": "strict_less_than",
        "gates": gates,
        "failure_reasons": reasons,
        "pending_reasons": [],
    }


def benchmark(
    *,
    backend: StereoBackend,
    practice_root: Path,
    starter_root: Path,
    output_root: Path,
    trips: Sequence[str],
    repeats: int,
    warmup_frames: int,
    max_frames_per_trip: int | None,
    skip_evaluation: bool,
    lane_reference_path: Path | None,
    model_provenance: Mapping[str, Any],
    parity_provenance: Mapping[str, Any] | None,
    latency_target_ms: float,
    progress_every: int,
    gpu_device_id: int,
    ttc_policy: str = "baseline",
) -> dict[str, Any]:
    trip_dataset_class = _load_dataset_class(starter_root)
    missing_trips = [
        trip_id for trip_id in trips if not (practice_root / trip_id).is_dir()
    ]
    if missing_trips:
        raise BackendConfigurationError(
            f"missing practice trip directories: {', '.join(missing_trips)}"
        )
    dataset_protocol, frozen_frame_ids = freeze_dataset_protocol(
        trip_dataset_class, practice_root
    )
    lane_reference = None
    lane_reference_provenance = None
    if lane_reference_path is not None:
        (
            lane_reference,
            lane_reference_provenance,
        ) = load_lane_reference_summary(
            lane_reference_path,
            dataset_protocol=dataset_protocol,
            model_provenance=model_provenance,
            latency_target_ms=latency_target_ms,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_root = output_root / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    gpu_sampler = (
        ProcessGpuMemorySampler(gpu_device_id)
        if backend.name != "sgbm"
        else None
    )
    print(
        f"Warm-up: {warmup_frames} frames with {backend.name}/{backend.precision}",
        flush=True,
    )
    warm_up_backend(
        backend,
        trip_dataset_class,
        practice_root,
        trips[0],
        warmup_frames,
        ttc_policy,
    )
    if gpu_sampler is not None:
        gpu_sampler.start()

    runtime_rows: list[dict[str, Any]] = []
    reference_predictions: dict[tuple[str, int], float] = {}
    prediction_truth: list[tuple[float, float]] = []
    observed_input_shape: tuple[int, int, int, int] | None = None
    observed_backend_metadata: dict[str, Any] | None = None
    nondeterministic_predictions = 0
    peak_rss = _peak_rss_mb()

    for repeat_index in range(repeats):
        for trip_id in trips:
            dataset = trip_dataset_class(practice_root / trip_id)
            frames = list(dataset.iter_frames())
            enumerated_ids = tuple(int(frame.frame_id) for frame in frames)
            if enumerated_ids != frozen_frame_ids[trip_id]:
                raise BackendConfigurationError(
                    f"{trip_id} frame enumeration changed after protocol freeze"
                )
            if max_frames_per_trip is not None:
                frames = frames[:max_frames_per_trip]
            pipeline = _trip_context(dataset, backend, ttc_policy)
            prediction_rows = []

            for frame_index, frame in enumerate(frames):
                end_to_end_started = time.perf_counter()
                load_started = time.perf_counter()
                left = dataset.load_left(frame.frame_id)
                right = dataset.load_right(frame.frame_id)
                image_load_ms = (time.perf_counter() - load_started) * 1000.0
                outcome = pipeline.process(left, right, frame.timestamp)
                if observed_input_shape is None:
                    observed_input_shape = outcome.stereo_result.input_shape
                    observed_backend_metadata = dict(
                        outcome.stereo_result.metadata
                    )
                elif observed_input_shape != outcome.stereo_result.input_shape:
                    raise BackendConfigurationError(
                        "backend input shape changed during a fixed-shape run: "
                        f"{observed_input_shape} -> "
                        f"{outcome.stereo_result.input_shape}"
                    )
                elif observed_backend_metadata != dict(
                    outcome.stereo_result.metadata
                ):
                    raise BackendConfigurationError(
                        "backend metadata changed during a reproducible run"
                    )
                end_to_end_ms = (time.perf_counter() - end_to_end_started) * 1000.0
                key = (trip_id, int(frame.frame_id))
                if repeat_index == 0:
                    reference_predictions[key] = outcome.predicted_ttc
                    prediction_truth.append(
                        (outcome.predicted_ttc, float(frame.min_ttc))
                    )
                    prediction_rows.append(
                        {
                            "frame_id": frame.frame_id,
                            "timestamp": round(float(frame.timestamp), 6),
                            "predicted_ttc": _format_ttc(outcome.predicted_ttc),
                        }
                    )
                elif not _prediction_equal(
                    reference_predictions[key], outcome.predicted_ttc
                ):
                    nondeterministic_predictions += 1

                row: dict[str, Any] = {
                    "repeat": repeat_index + 1,
                    "trip_id": trip_id,
                    "frame_id": int(frame.frame_id),
                    "image_load_ms": image_load_ms,
                    "stereo_ms": outcome.stereo_ms,
                    "ground_ms": outcome.ground_ms,
                    "components_ms": outcome.components_ms,
                    "tracking_ms": outcome.tracking_ms,
                    "pipeline_compute_ms": outcome.pipeline_compute_ms,
                    "end_to_end_ms": end_to_end_ms,
                    "valid_fraction": float(
                        np.mean(outcome.stereo_result.valid_mask)
                    ),
                    "ground_confidence": outcome.ground_confidence,
                    "component_count": outcome.component_count,
                    "relevant_component_count": outcome.relevant_component_count,
                }
                for timing_name, value in outcome.stereo_result.timings_ms.items():
                    row[f"backend_{timing_name}_ms"] = float(value)
                runtime_rows.append(row)
                current_peak = _peak_rss_mb()
                if current_peak is not None:
                    peak_rss = max(peak_rss or 0.0, current_peak)
                if progress_every and (
                    frame_index % progress_every == 0
                    or frame_index + 1 == len(frames)
                ):
                    print(
                        f"repeat {repeat_index + 1}/{repeats} {trip_id}: "
                        f"{frame_index + 1}/{len(frames)} "
                        f"compute={outcome.pipeline_compute_ms:.2f}ms",
                        flush=True,
                    )
            if repeat_index == 0:
                _write_csv(predictions_root / f"{trip_id}.csv", prediction_rows)

    sampled_gpu_memory = (
        gpu_sampler.stop() if gpu_sampler is not None else None
    )
    _write_csv(output_root / "runtime_frames.csv", runtime_rows)
    timing = aggregate_runtime(runtime_rows)
    evaluation = None
    if not skip_evaluation:
        evaluation = _evaluate_predictions(
            predictions_root,
            practice_root,
            starter_root,
            output_root / "evaluation.json",
        )
    confusion = danger_confusion(prediction_truth)
    allocator_gpu_memory = backend.peak_gpu_memory_mb()
    if sampled_gpu_memory is not None and sampled_gpu_memory > 0:
        peak_gpu_memory = sampled_gpu_memory
        gpu_memory_source = gpu_sampler.source if gpu_sampler is not None else None
    elif backend.name == "lightstereo-pytorch":
        peak_gpu_memory = allocator_gpu_memory
        gpu_memory_source = "torch_cuda_max_memory_reserved"
    elif backend.name == "sgbm":
        peak_gpu_memory = 0.0
        gpu_memory_source = "cpu_backend"
    else:
        peak_gpu_memory = None
        gpu_memory_source = gpu_sampler.source if gpu_sampler is not None else None
    complete_protocol = (
        dataset_protocol["complete"] is True
        and tuple(trips) == TRIPS
        and max_frames_per_trip is None
    )
    expected_runtime_rows = EXPECTED_FRAMES_PER_REPEAT * OFFICIAL_REPEATS
    acceptance = acceptance_report(
        timing=timing,
        evaluation=evaluation,
        peak_gpu_memory_mb=peak_gpu_memory,
        frames_measured=len(reference_predictions),
        expected_frames=EXPECTED_FRAMES_PER_REPEAT,
        nondeterministic_predictions=nondeterministic_predictions,
        lane_reference=lane_reference,
        latency_target_ms=latency_target_ms,
        repeats=repeats,
        warmup_frames=warmup_frames,
        complete_dataset_protocol=complete_protocol,
        runtime_rows_measured=len(runtime_rows),
        expected_runtime_rows=expected_runtime_rows,
        parity_passed=(
            parity_provenance is not None
            if backend.name
            in {"lightstereo-onnx", "lightstereo-tensorrt"}
            else True
        ),
        provenance_valid=bool(model_provenance),
    )
    repository_root = Path(__file__).resolve().parents[4]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend.name,
        "precision": backend.precision,
        "model_sha256": backend.model_sha256,
        "model_provenance": dict(model_provenance),
        "input_shape": list(observed_input_shape or ()),
        "backend_metadata": observed_backend_metadata or {},
        "configuration": {
            "ttc_policy": ttc_policy,
            "trips": list(trips),
            "repeats": repeats,
            "warmup_frames": warmup_frames,
            "max_frames_per_trip": max_frames_per_trip,
            "protocol_schema": BENCHMARK_PROTOCOL_SCHEMA,
            "latency_target_ms": latency_target_ms,
            "latency_comparison": "strict_less_than",
            "gpu_memory_limit_mb": GPU_MEMORY_LIMIT_MB,
            "stage2a_reference": {
                "overall_composite_score": STAGE2A_COMPOSITE,
                "overall_f1": STAGE2A_DANGER_F1,
            },
            "quality_budget": {
                "maximum_composite_loss": MAX_COMPOSITE_LOSS,
                "maximum_f1_loss": MAX_F1_LOSS,
            },
            "lane_reference_summary": (
                str(lane_reference_path) if lane_reference_path else None
            ),
            "lane_reference_provenance": lane_reference_provenance,
            "parity_report": (
                dict(parity_provenance)
                if parity_provenance is not None
                else None
            ),
        },
        "dataset": {
            "practice_root": str(practice_root.resolve()),
            "frames_per_repeat": len(reference_predictions),
            "runtime_rows": len(runtime_rows),
            "expected_frames_per_repeat": EXPECTED_FRAMES_PER_REPEAT,
            "expected_runtime_rows": expected_runtime_rows,
            "protocol": dataset_protocol,
        },
        "timing_ms": timing,
        "throughput": {
            "fps_from_pipeline_mean": (
                1000.0 / timing["pipeline_compute_ms"]["mean"]
            ),
            "fps_at_pipeline_p95": (
                1000.0 / timing["pipeline_compute_ms"]["p95"]
            ),
        },
        "resources": {
            "peak_process_ram_mb": peak_rss,
            "peak_gpu_memory_mb": peak_gpu_memory,
            "gpu_memory_measurement": {
                "source": gpu_memory_source,
                "sampler_error": (
                    gpu_sampler.error if gpu_sampler is not None else None
                ),
                "backend_allocator_peak_mb": allocator_gpu_memory,
            },
        },
        "evaluation": evaluation,
        "danger_confusion": confusion,
        "nondeterministic_predictions": nondeterministic_predictions,
        "acceptance": acceptance,
        "environment": environment_metadata(repository_root),
    }
    (output_root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_comparison(output_root / "comparison.csv", summary)
    return summary


def comparison_row(
    summary: dict[str, Any], *, summary_path: Path | None = None
) -> dict[str, Any]:
    timing = summary["timing_ms"]["pipeline_compute_ms"]
    evaluation = summary["evaluation"] or {}
    confusion = summary["danger_confusion"]
    resources = summary["resources"]
    environment = summary["environment"]
    latency_target = summary.get("configuration", {}).get(
        "latency_target_ms",
        summary.get("acceptance", {}).get("latency_target_ms", ""),
    )
    return {
        "summary_path": str(summary_path) if summary_path is not None else "",
        "backend": summary["backend"],
        "precision": summary["precision"],
        "model_sha256": summary["model_sha256"] or "",
        "pipeline_p50_ms": timing["p50"],
        "pipeline_p95_ms": timing["p95"],
        "pipeline_p99_ms": timing["p99"],
        "latency_target_ms_strict_lt": latency_target,
        "fps_from_mean": summary["throughput"]["fps_from_pipeline_mean"],
        "overall_composite": evaluation.get("overall_composite_score", ""),
        "danger_f1": evaluation.get("overall_f1", ""),
        "tp": confusion["tp"],
        "fp": confusion["fp"],
        "fn": confusion["fn"],
        "worst_trip_composite": evaluation.get("worst_trip_composite", ""),
        "cpu": environment["processor"],
        "peak_process_ram_mb": resources["peak_process_ram_mb"],
        "peak_gpu_memory_mb": (
            resources["peak_gpu_memory_mb"]
            if resources["peak_gpu_memory_mb"] is not None
            else ""
        ),
        "status": summary["acceptance"]["status"],
        "passed": (
            summary["acceptance"]["passed"]
            if summary["acceptance"]["passed"] is not None
            else ""
        ),
        "failure_reasons": "; ".join(
            summary["acceptance"]["failure_reasons"]
        ),
    }


def write_comparison(path: Path, summary: dict[str, Any]) -> None:
    _write_csv(path, [comparison_row(summary)])


def discover_benchmark_summaries(paths: Sequence[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            discovered.add(resolved)
        elif resolved.is_dir():
            discovered.update(resolved.rglob("benchmark_summary.json"))
        else:
            raise BackendConfigurationError(
                f"summary path does not exist: {resolved}"
            )
    if not discovered:
        raise BackendConfigurationError("no benchmark_summary.json files found")
    return sorted(discovered)


def summary_is_deployment_eligible(summary: Mapping[str, Any]) -> bool:
    acceptance = summary.get("acceptance", {})
    gates = acceptance.get("gates", {})
    configuration = summary.get("configuration", {})
    dataset = summary.get("dataset", {})
    required_gates = (
        "pipeline_p95_strictly_below_target",
        "exactly_five_repeats",
        "warmup_at_least_100_frames",
        "complete_six_trip_dataset",
        "exactly_3600_frames_per_repeat",
        "exactly_18000_runtime_rows",
        "repeat_determinism",
        "gpu_vram_le_5gb",
        "artifact_provenance_valid",
        "converted_backend_parity_passed",
        "stage2a_quality_budget",
        "lane_fp32_quality_budget",
    )
    target = configuration.get("latency_target_ms")
    try:
        p95 = float(summary["timing_ms"]["pipeline_compute_ms"]["p95"])
        target_value = float(target)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        acceptance.get("passed") is True
        and acceptance.get("status") == "accepted"
        and acceptance.get("latency_comparison") == "strict_less_than"
        and acceptance.get("latency_target_ms") == target
        and configuration.get("protocol_schema") == BENCHMARK_PROTOCOL_SCHEMA
        and configuration.get("trips") == list(TRIPS)
        and configuration.get("repeats") == OFFICIAL_REPEATS
        and isinstance(configuration.get("warmup_frames"), int)
        and configuration["warmup_frames"] >= MINIMUM_WARMUP_FRAMES
        and configuration.get("max_frames_per_trip") is None
        and dataset.get("frames_per_repeat") == EXPECTED_FRAMES_PER_REPEAT
        and dataset.get("runtime_rows") == EXPECTED_RUNTIME_ROWS
        and dataset.get("protocol", {}).get("complete") is True
        and all(gates.get(name) is True for name in required_gates)
        and p95 < target_value
    )


def aggregate_benchmark_summaries(
    summary_paths: Sequence[Path],
    *,
    comparison_output: Path,
    selection_output: Path,
) -> dict[str, Any]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in discover_benchmark_summaries(summary_paths):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            comparison_row(summary, summary_path=path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise BackendConfigurationError(
                f"invalid benchmark summary {path}: {error}"
            ) from error
        records.append((path, summary))
    _write_csv(
        comparison_output,
        [
            comparison_row(summary, summary_path=path)
            for path, summary in records
        ],
    )

    eligible = [
        (path, summary)
        for path, summary in records
        if summary_is_deployment_eligible(summary)
    ]
    selected: tuple[Path, dict[str, Any]] | None = None
    tied: list[tuple[Path, dict[str, Any]]] = []
    if eligible:
        fastest_p95 = min(
            float(summary["timing_ms"]["pipeline_compute_ms"]["p95"])
            for _, summary in eligible
        )
        tied = [
            (path, summary)
            for path, summary in eligible
            if (
                float(summary["timing_ms"]["pipeline_compute_ms"]["p95"])
                - fastest_p95
            )
            / fastest_p95
            < 0.05
        ]
        selected = max(
            tied,
            key=lambda item: (
                float(item[1]["evaluation"]["overall_f1"]),
                float(item[1]["evaluation"]["overall_composite_score"]),
                -float(item[1]["timing_ms"]["pipeline_compute_ms"]["p95"]),
                item[1]["backend"],
                item[1]["precision"],
            ),
        )

    selection = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(records),
        "eligible_count": len(eligible),
        "tie_rule": (
            "Select the lowest-P95 accepted candidate; candidates less than "
            "5% slower than the fastest are tied, then higher danger-F1 wins."
        ),
        "tie_candidate_paths": [str(path) for path, _ in tied],
        "selected": (
            {
                "summary_path": str(selected[0]),
                "backend": selected[1]["backend"],
                "precision": selected[1]["precision"],
                "pipeline_p95_ms": selected[1]["timing_ms"][
                    "pipeline_compute_ms"
                ]["p95"],
                "overall_f1": selected[1]["evaluation"]["overall_f1"],
                "overall_composite_score": selected[1]["evaluation"][
                    "overall_composite_score"
                ],
            }
            if selected is not None
            else None
        ),
        "selection_status": (
            "selected" if selected is not None else "no_accepted_candidate"
        ),
    }
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_output.write_text(
        json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8"
    )
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=(
            "sgbm",
            "lightstereo-pytorch",
            "lightstereo-onnx",
            "lightstereo-tensorrt",
        ),
    )
    parser.add_argument(
        "--precision", required=True, choices=("fp32", "fp16", "int8")
    )
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument(
        "--practice-root", type=Path, default=Path("Practice_Dataset")
    )
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--trips", nargs="+", default=list(TRIPS))
    parser.add_argument("--max-frames-per-trip", type=int)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--ttc-policy",
        choices=(
            "baseline",
            "guarded",
            "object-depth",
            "filtered-motion",
            "object-centric",
        ),
        default="guarded",
        help=(
            "Post-processing policy. 'guarded' suppresses implausible "
            "road-surface tracks and is the frozen Phase 3 candidate; "
            "'object-depth' and 'filtered-motion' isolate the Phase 05A "
            "measurement/filter ablations; "
            "'object-centric' uses inner-ROI modal depth plus an "
            "uncertainty-aware distance/velocity filter."
        ),
    )
    parser.add_argument(
        "--latency-target-ms",
        type=float,
        default=DEFAULT_PIPELINE_LATENCY_TARGET_MS,
        help=(
            "Strict pipeline-compute P95 deployment threshold; candidates "
            "must be below this value (default: 75.0 ms)."
        ),
    )
    parser.add_argument(
        "--lane-reference-summary",
        type=Path,
        help="FP32 benchmark_summary.json used for conversion quality gating.",
    )
    parser.add_argument(
        "--parity-report",
        type=Path,
        help=(
            "Passed frozen 72-pair parity JSON required for an official "
            "ONNX/TensorRT run."
        ),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        help=(
            "Generated model sidecar; defaults to "
            "<model-path>.manifest.json."
        ),
    )
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument(
        "--openstereo-root",
        type=Path,
        default=Path("~/benchmarks/OpenStereo"),
    )
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument(
        "--stereo-workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="1=sequential matchers, 2=persistent concurrent matchers.",
    )
    parser.add_argument(
        "--stereo-roi-top",
        type=int,
        choices=(0, 96),
        default=0,
        help=(
            "SGBM-only top crop in native-image rows; 0 preserves the frozen "
            "full-frame reference, and 96 is the single Phase 2B ROI candidate."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def build_aggregate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Phase 2B benchmark summaries and select a deployment "
            "candidate using the frozen latency/F1 rule."
        )
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        type=Path,
        help="Summary JSON files or directories searched recursively.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path(
            "ai_cv/outputs/benchmarks/phase02b_latency/comparison.csv"
        ),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path(
            "ai_cv/outputs/benchmarks/phase02b_latency/selection.json"
        ),
    )
    return parser


def build_parity_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a converted LightStereo backend with the PyTorch FP32 "
            "reference on the frozen 72-pair manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-model-path", type=Path, required=True)
    parser.add_argument(
        "--candidate-backend",
        required=True,
        choices=(
            "lightstereo-pytorch",
            "lightstereo-onnx",
            "lightstereo-tensorrt",
        ),
    )
    parser.add_argument(
        "--candidate-precision",
        required=True,
        choices=("fp32", "fp16", "int8"),
    )
    parser.add_argument("--candidate-model-path", type=Path, required=True)
    parser.add_argument(
        "--openstereo-root",
        type=Path,
        default=Path("~/benchmarks/OpenStereo"),
    )
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ai_cv/outputs/benchmarks/phase02b_latency/parity_report.json"
        ),
    )
    return parser


def parity_main(argv: Sequence[str]) -> int:
    parser = build_parity_parser()
    args = parser.parse_args(argv)
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    reference_backend: StereoBackend | None = None
    candidate_backend: StereoBackend | None = None
    try:
        reference_backend = create_backend(
            "lightstereo-pytorch",
            precision="fp32",
            model_path=args.reference_model_path,
            openstereo_root=args.openstereo_root,
            config_path=args.config_path,
            device_id=args.device_id,
        )
        candidate_backend = create_backend(
            args.candidate_backend,
            precision=args.candidate_precision,
            model_path=args.candidate_model_path,
            openstereo_root=args.openstereo_root,
            config_path=args.config_path,
            device_id=args.device_id,
        )
        report = run_parity_gate(
            reference_backend=reference_backend,
            candidate_backend=candidate_backend,
            manifest_path=args.manifest,
            data_root=args.data_root,
            output_path=args.output,
            warmup_frames=args.warmup_frames,
            progress_every=args.progress_every,
        )
    except BackendConfigurationError as error:
        parser.error(str(error))
    finally:
        if candidate_backend is not None:
            candidate_backend.close()
        if reference_backend is not None:
            reference_backend.close()
    aggregate = report["aggregate"]
    print(
        f"PARITY status={report['status']} "
        f"MAE={aggregate['mean_absolute_error_px']:.4f}px "
        f"P95={aggregate['p95_absolute_error_px']:.4f}px "
        f"bad3={aggregate['bad_3px_fraction']:.4%} "
        f"missing={aggregate['missing_reference_valid_fraction']:.4%}; "
        f"report={args.output}"
    )
    return 0 if report["passed"] else 1


def aggregate_main(argv: Sequence[str]) -> int:
    parser = build_aggregate_parser()
    args = parser.parse_args(argv)
    try:
        selection = aggregate_benchmark_summaries(
            args.summaries,
            comparison_output=args.comparison_output,
            selection_output=args.selection_output,
        )
    except BackendConfigurationError as error:
        parser.error(str(error))
    print(
        f"Aggregated {selection['candidate_count']} candidates; "
        f"status={selection['selection_status']}; "
        f"comparison={args.comparison_output}; "
        f"selection={args.selection_output}"
    )
    return 0 if selection["selected"] is not None else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "aggregate":
        return aggregate_main(arguments[1:])
    if arguments and arguments[0] == "parity":
        return parity_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.max_frames_per_trip is not None and args.max_frames_per_trip <= 0:
        parser.error("--max-frames-per-trip must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if (
        not math.isfinite(args.latency_target_ms)
        or args.latency_target_ms <= 0
    ):
        parser.error("--latency-target-ms must be finite and positive")
    if args.backend != "sgbm" and args.stereo_roi_top != 0:
        parser.error("--stereo-roi-top is only supported by --backend sgbm")
    invalid_trips = sorted(set(args.trips) - set(TRIPS))
    if invalid_trips:
        parser.error(f"unknown practice trip(s): {', '.join(invalid_trips)}")
    full_evaluation = (
        tuple(args.trips) == TRIPS and args.max_frames_per_trip is None
    )
    if not args.skip_evaluation and not full_evaluation:
        parser.error(
            "official evaluation requires all six complete trips; add "
            "--skip-evaluation for a partial smoke run"
        )
    if not args.skip_evaluation and args.repeats != OFFICIAL_REPEATS:
        parser.error(
            f"official evaluation requires exactly {OFFICIAL_REPEATS} repeats"
        )
    if (
        not args.skip_evaluation
        and args.warmup_frames < MINIMUM_WARMUP_FRAMES
    ):
        parser.error(
            "official evaluation requires at least "
            f"{MINIMUM_WARMUP_FRAMES} warm-up frames"
        )
    requires_lane_reference = (
        args.backend in {"lightstereo-onnx", "lightstereo-tensorrt"}
        or (
            args.backend == "lightstereo-pytorch"
            and args.precision != "fp32"
        )
    )
    if requires_lane_reference and args.lane_reference_summary is None:
        parser.error(
            "converted learned backends require --lane-reference-summary "
            "pointing to the LightStereo PyTorch FP32 benchmark_summary.json"
        )
    converted_backend = args.backend in {
        "lightstereo-onnx",
        "lightstereo-tensorrt",
    }
    if (
        converted_backend
        and not args.skip_evaluation
        and args.parity_report is None
    ):
        parser.error(
            "official ONNX/TensorRT evaluation requires --parity-report "
            "from the passed frozen 72-pair conversion gate"
        )
    output_root = args.output_root or (
        Path("ai_cv/outputs/benchmarks/phase02b_latency")
        / args.backend
        / args.precision
    )

    backend: StereoBackend | None = None
    try:
        model_provenance = validate_model_provenance(
            backend=args.backend,
            precision=args.precision,
            model_path=args.model_path,
            openstereo_root=args.openstereo_root,
            config_path=args.config_path,
            artifact_manifest_path=args.artifact_manifest,
            calibration_manifest_path=args.calibration_manifest,
            calibration_cache_path=args.calibration_cache,
        )
        parity_provenance = None
        if args.parity_report is not None:
            if not converted_backend:
                raise BackendConfigurationError(
                    "--parity-report applies only to ONNX/TensorRT backends"
                )
            parity_provenance = validate_parity_report(
                args.parity_report,
                backend=args.backend,
                precision=args.precision,
                model_sha256=model_provenance["artifact_sha256"],
                model_provenance=model_provenance,
            )
        backend = create_backend(
            args.backend,
            precision=args.precision,
            model_path=args.model_path,
            openstereo_root=args.openstereo_root,
            config_path=args.config_path,
            device_id=args.device_id,
            opencv_threads=args.opencv_threads,
            stereo_workers=args.stereo_workers,
            stereo_roi_top=args.stereo_roi_top,
        )
        summary = benchmark(
            backend=backend,
            practice_root=args.practice_root,
            starter_root=args.starter_root,
            output_root=output_root,
            trips=args.trips,
            repeats=args.repeats,
            warmup_frames=args.warmup_frames,
            max_frames_per_trip=args.max_frames_per_trip,
            skip_evaluation=args.skip_evaluation,
            lane_reference_path=args.lane_reference_summary,
            model_provenance=model_provenance,
            parity_provenance=parity_provenance,
            latency_target_ms=args.latency_target_ms,
            progress_every=args.progress_every,
            gpu_device_id=args.device_id,
            ttc_policy=args.ttc_policy,
        )
    except BackendConfigurationError as error:
        parser.error(str(error))
    finally:
        if backend is not None:
            backend.close()

    pipeline = summary["timing_ms"]["pipeline_compute_ms"]
    print(
        f"RESULT {summary['backend']}/{summary['precision']}: "
        f"pipeline P50={pipeline['p50']:.2f}ms "
        f"P95={pipeline['p95']:.2f}ms P99={pipeline['p99']:.2f}ms; "
        f"status={summary['acceptance']['status']}; "
        f"summary={output_root / 'benchmark_summary.json'}"
    )
    if not args.skip_evaluation and summary["acceptance"]["passed"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
