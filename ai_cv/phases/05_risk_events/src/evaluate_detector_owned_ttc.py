"""Targeted detector-owned stereo TTC ablation for Phase 05A.

YOLO detections are the object proposals. SGBM supplies metric depth inside
each proposal, and the existing causal Guardian tracker estimates TTC. The
script intentionally defaults to the weak T03/T05 trips and emits two fixed
policies from the same run:

* ``detector_owned``: the frozen guarded motion limits.
* ``detector_owned_ego_cap``: additionally rejects a closing speed greater
  than ego speed + 3 m/s (a forward-lane physical plausibility check).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE02_SRC = (
    REPOSITORY_ROOT / "ai_cv" / "phases" / "02_detection_tracking" / "src"
)
if str(PHASE02_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE02_SRC))

from classical_geometry import (  # noqa: E402
    ObstacleComponent,
    collision_corridor_mask,
    estimate_ground_model,
    estimate_object_depth,
)
from classical_tracking import ComponentTracker, select_minimum_ttc  # noqa: E402
from cross_validate_guarded_ttc import score  # noqa: E402
from cross_validate_yolo26_fusion import load_detections_csv  # noqa: E402
from detector_interfaces import Detection  # noqa: E402
from stereo_backends import create_backend  # noqa: E402


ROAD_USER_CLASSES = frozenset(
    {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
)


def _clip_bbox(
    detection: Detection, image_shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    height, width = image_shape
    x0, y0, x1, y1 = detection.bbox_xyxy
    left = int(np.clip(math.floor(x0), 0, width - 1))
    top = int(np.clip(math.floor(y0), 0, height - 1))
    right = int(np.clip(math.ceil(x1), left + 1, width))
    bottom = int(np.clip(math.ceil(y1), top + 1, height))
    if right - left < 3 or bottom - top < 3:
        return None
    return left, top, right, bottom


def detection_component(
    detection: Detection,
    disparity_px: np.ndarray,
    valid_mask: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    corridor_mask: np.ndarray,
    *,
    component_id: int,
) -> ObstacleComponent | None:
    """Convert one road-user detection into a metric stereo component."""
    if detection.class_name.lower() not in ROAD_USER_CLASSES:
        return None
    bbox = _clip_bbox(detection, disparity_px.shape)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    roi_disparity = disparity_px[y0:y1, x0:x1]
    roi_valid = (
        valid_mask[y0:y1, x0:x1]
        & np.isfinite(roi_disparity)
        & (roi_disparity > 0.5)
    )
    estimate = estimate_object_depth(
        roi_disparity,
        roi_valid,
        roi_valid,
        focal_length_px,
        baseline_m,
        inner_width_fraction=0.70,
        inner_height_fraction=0.72,
    )
    if estimate is None:
        return None

    valid_count = int(np.count_nonzero(roi_valid))
    box_area = (x1 - x0) * (y1 - y0)
    lr_support = float(valid_count / max(1, box_area))
    corridor_overlap = float(
        np.mean(corridor_mask[y0:y1, x0:x1])
    )
    # Detection confidence establishes identity; stereo support and modal
    # confidence establish whether its metric depth is usable.
    quality = float(
        np.clip(
            0.35 * detection.confidence
            + 0.25 * min(1.0, lr_support / 0.35)
            + 0.40 * estimate.confidence,
            0.0,
            1.0,
        )
    )
    return ObstacleComponent(
        component_id=component_id,
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
        area=box_area,
        center_x=0.5 * (x0 + x1),
        center_y=0.5 * (y0 + y1),
        bottom_y=y1,
        depth_m=estimate.depth_m,
        depth_p20_m=estimate.depth_m,
        depth_p35_m=estimate.depth_m,
        depth_mad_m=estimate.depth_mad_m,
        lr_support=lr_support,
        corridor_overlap=corridor_overlap,
        quality=quality,
        object_depth_m=estimate.depth_m,
        object_depth_mad_m=estimate.depth_mad_m,
        object_depth_confidence=estimate.confidence,
        object_depth_mode_count=estimate.mode_count,
    )


def _write_predictions(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_id", "timestamp", "predicted_ttc"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_runtime_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ttc_text(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.6f}"


def _prediction_equal(first: float, second: float) -> bool:
    if math.isinf(first) and math.isinf(second):
        return True
    return first == second


def _latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(np.mean(array)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    starter_root = args.starter_root.resolve()
    if str(starter_root) not in sys.path:
        sys.path.insert(0, str(starter_root))
    from team_kit.dataset_loader import TripDataset

    backend = create_backend(
        "sgbm",
        precision="fp32",
        opencv_threads=args.opencv_threads,
        stereo_workers=args.stereo_workers,
    )
    detector = None
    gpu_sampler = None
    detector_workload: dict[str, float | int] | None = None
    if args.detector_backend != "cached":
        from benchmark_stereo_latency import ProcessGpuMemorySampler
        from yolo26_backends import get_detector_backend

        detector = get_detector_backend(
            args.detector_backend,
            str(args.model_path),
            args.detector_confidence,
        )
        if hasattr(detector, "model") and hasattr(detector.model, "model"):
            from ultralytics.utils.torch_utils import get_flops, get_num_params

            detector_workload = {
                "parameters": int(get_num_params(detector.model.model)),
                "gflops_per_640x640_frame": float(
                    get_flops(detector.model.model, imgsz=640)
                ),
            }
        gpu_sampler = ProcessGpuMemorySampler(0)
        gpu_sampler.start()
    executor = (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="guardian-live")
        if detector is not None and args.parallel_inference
        else None
    )
    report: dict[str, object] = {
        "experiment": "phase05a_detector_owned_ttc",
        "detector": {
            "backend": args.detector_backend,
            "model_path": str(args.model_path) if detector is not None else None,
            "model_sha256": (
                getattr(detector, "model_sha256", None)
                if detector is not None
                else None
            ),
            "confidence_threshold": args.detector_confidence,
            "parallel_with_stereo": bool(executor is not None),
        },
        "protocol": {
            "warmup_frames": args.warmup_frames,
            "repeats": args.repeats,
            "start_frame_index": args.start_frame_index,
            "max_frames_per_trip": args.max_frames_per_trip,
            "latency_target_ms": args.latency_target_ms,
            "disk_loading_excluded_from_gate": True,
        },
        "hardware_independent_workload": {
            "detector": detector_workload,
            "stereo": {
                "native_width": 640,
                "native_height": 360,
                "disparities_per_matcher": 96,
                "matchers": 2,
                "disparity_hypotheses_per_pair": 640 * 360 * 96 * 2,
            },
            "interpretation": (
                "Detector GFLOPs and SGBM disparity hypotheses are separate "
                "workload proxies and must not be summed into one FLOP count."
            ),
        },
        "policies": {
            "detector_owned": {
                "maximum_closing_speed_mps": 20.0,
            },
            "detector_owned_ego_cap": {
                "maximum_closing_speed_mps": "min(20, ego_speed_mps + 3)",
            },
        },
        "trips": {},
    }
    runtime_rows: list[dict[str, object]] = []
    nondeterministic_predictions = 0
    prediction_comparisons = 0
    gpu_memory: dict[str, float] | None = None
    try:
        if detector is not None and args.warmup_frames:
            warmup_dataset = TripDataset(args.practice_root / args.trips[0])
            warmup_records = list(warmup_dataset.iter_frames())
            for index in range(args.warmup_frames):
                frame = warmup_records[index % len(warmup_records)]
                left = warmup_dataset.load_left(frame.frame_id)
                right = warmup_dataset.load_right(frame.frame_id)
                if executor is not None:
                    stereo_future = executor.submit(backend.infer, left, right)
                    detector_future = executor.submit(detector.infer, left)
                    stereo_future.result()
                    detector_future.result()
                else:
                    backend.infer(left, right)
                    detector.infer(left)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass

        for trip_id in args.trips:
            dataset = TripDataset(args.practice_root / trip_id)
            calibration = dataset.load_calibration()
            image_shape = (
                int(calibration["image_height"]),
                int(calibration["image_width"]),
            )
            focal_length_px = float(calibration["K_left"][0][0])
            baseline_m = float(calibration["baseline_m"])
            corridor = collision_corridor_mask(
                image_shape,
                top_width_fraction=0.10,
                bottom_width_fraction=0.50,
            )
            detections_by_frame = (
                load_detections_csv(args.detections_dir / f"{trip_id}.csv")
                if detector is None
                else {}
            )
            policies = {
                "detector_owned": [],
                "detector_owned_ego_cap": [],
            }
            truth: list[float] = []
            stats = {
                "detections": 0,
                "depth_valid_components": 0,
                "frames_with_depth_component": 0,
                "frames_with_risk_track": 0,
            }
            frames = list(dataset.iter_frames())
            if args.start_frame_index:
                frames = frames[args.start_frame_index :]
            if args.max_frames_per_trip is not None:
                frames = frames[: args.max_frames_per_trip]
            for repeat_index in range(args.repeats):
                tracker = ComponentTracker(
                    image_shape,
                    depth_attribute="object_depth_m",
                    maximum_missed=3,
                    risk_top_width_fraction=0.10,
                    risk_bottom_width_fraction=0.50,
                    minimum_bottom_fraction=0.45,
                    minimum_height_fraction=0.025,
                )
                for index, frame in enumerate(frames):
                    load_started = time.perf_counter()
                    left = dataset.load_left(frame.frame_id)
                    right = dataset.load_right(frame.frame_id)
                    image_load_ms = (time.perf_counter() - load_started) * 1000.0
                    pipeline_started = time.perf_counter()
                    inference_started = time.perf_counter()
                    if executor is not None:
                        stereo_future = executor.submit(backend.infer, left, right)
                        detector_future = executor.submit(detector.infer, left)
                        stereo = stereo_future.result()
                        detection_result = detector_future.result()
                    else:
                        stereo = backend.infer(left, right)
                        detection_result = (
                            detector.infer(left) if detector is not None else None
                        )
                    inference_wall_ms = (
                        time.perf_counter() - inference_started
                    ) * 1000.0
                    detections = (
                        list(detection_result.detections)
                        if detection_result is not None
                        else detections_by_frame.get(int(frame.frame_id), [])
                    )

                    postprocess_started = time.perf_counter()
                    ground_model, _ = estimate_ground_model(stereo.disparity_px)
                    ground_confidence = (
                        float(ground_model.confidence)
                        if ground_model is not None
                        else 0.0
                    )
                    components = [
                        component
                        for component_id, detection in enumerate(
                            detections, start=1
                        )
                        if (
                            component := detection_component(
                                detection,
                                stereo.disparity_px,
                                stereo.valid_mask,
                                focal_length_px,
                                baseline_m,
                                corridor,
                                component_id=component_id,
                            )
                        )
                        is not None
                    ]
                    tracks = tracker.update(components, float(frame.timestamp))
                    risk_tracks = tracker.risk_tracks(tracks)
                    common = {
                        "minimum_track_confidence": 0.65,
                        "maximum_depth_m": 20.0,
                        "maximum_motion_residual_m": 0.8,
                    }
                    plain_ttc, _, _, _ = select_minimum_ttc(
                        risk_tracks,
                        ground_confidence,
                        maximum_closing_speed_mps=20.0,
                        **common,
                    )
                    ego_cap = min(
                        20.0,
                        max(3.0, float(frame.speed_kmh) / 3.6 + 3.0),
                    )
                    capped_ttc, _, _, _ = select_minimum_ttc(
                        risk_tracks,
                        ground_confidence,
                        maximum_closing_speed_mps=ego_cap,
                        **common,
                    )
                    postprocess_ms = (
                        time.perf_counter() - postprocess_started
                    ) * 1000.0
                    pipeline_compute_ms = (
                        time.perf_counter() - pipeline_started
                    ) * 1000.0
                    detector_preprocess_ms = (
                        detection_result.preprocess_ms
                        if detection_result is not None
                        else 0.0
                    )
                    detector_inference_ms = (
                        detection_result.inference_ms
                        if detection_result is not None
                        else 0.0
                    )
                    detector_postprocess_ms = (
                        detection_result.postprocess_ms
                        if detection_result is not None
                        else 0.0
                    )
                    runtime_rows.append(
                        {
                            "repeat": repeat_index + 1,
                            "trip_id": trip_id,
                            "frame_id": int(frame.frame_id),
                            "image_load_ms": image_load_ms,
                            "inference_wall_ms": inference_wall_ms,
                            "stereo_ms": float(
                                stereo.timings_ms.get(
                                    "stereo_total",
                                    sum(stereo.timings_ms.values()),
                                )
                            ),
                            "detector_preprocess_ms": detector_preprocess_ms,
                            "detector_inference_ms": detector_inference_ms,
                            "detector_postprocess_ms": detector_postprocess_ms,
                            "postprocess_ttc_ms": postprocess_ms,
                            "pipeline_compute_ms": pipeline_compute_ms,
                            "detections": len(detections),
                            "depth_valid_components": len(components),
                            "risk_tracks": len(risk_tracks),
                        }
                    )
                    if repeat_index == 0:
                        stats["detections"] += len(detections)
                        stats["depth_valid_components"] += len(components)
                        stats["frames_with_depth_component"] += int(
                            bool(components)
                        )
                        stats["frames_with_risk_track"] += int(bool(risk_tracks))
                        policies["detector_owned"].append(float(plain_ttc))
                        policies["detector_owned_ego_cap"].append(
                            float(capped_ttc)
                        )
                        truth.append(float(frame.min_ttc))
                    else:
                        prediction_comparisons += 2
                        nondeterministic_predictions += int(
                            not _prediction_equal(
                                policies["detector_owned"][index],
                                float(plain_ttc),
                            )
                        )
                        nondeterministic_predictions += int(
                            not _prediction_equal(
                                policies["detector_owned_ego_cap"][index],
                                float(capped_ttc),
                            )
                        )
                    if args.progress_every and (
                        (index + 1) % args.progress_every == 0
                        or index + 1 == len(dataset)
                    ):
                        print(
                            f"repeat {repeat_index + 1}/{args.repeats} "
                            f"{trip_id}: {index + 1}/{len(dataset)}",
                            flush=True,
                        )

            trip_report: dict[str, object] = {"coverage": stats, "metrics": {}}
            truth_array = np.asarray(truth, dtype=float)
            for policy_name, predictions in policies.items():
                metrics = score(np.asarray(predictions, dtype=float), truth_array)
                trip_report["metrics"][policy_name] = asdict(metrics)
                rows = [
                    {
                        "frame_id": int(frame.frame_id),
                        "timestamp": f"{float(frame.timestamp):.6f}",
                        "predicted_ttc": _ttc_text(prediction),
                    }
                    for frame, prediction in zip(
                        frames, predictions, strict=True
                    )
                ]
                _write_predictions(
                    args.output_dir / policy_name / f"{trip_id}.csv", rows
                )
            report["trips"][trip_id] = trip_report
        if detector is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_memory = {
                        "peak_allocated_mb": float(
                            torch.cuda.max_memory_allocated() / (1024**2)
                        ),
                        "peak_reserved_mb": float(
                            torch.cuda.max_memory_reserved() / (1024**2)
                        ),
                        "process_peak_mb": (
                            gpu_sampler.stop()
                            if gpu_sampler is not None
                            else None
                        ),
                        "process_peak_source": (
                            gpu_sampler.source
                            if gpu_sampler is not None
                            else None
                        ),
                    }
                    gpu_sampler = None
            except ImportError:
                pass
    finally:
        if gpu_sampler is not None:
            gpu_sampler.stop()
        if executor is not None:
            executor.shutdown(wait=True)
        if detector is not None:
            detector.close()
        backend.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_runtime_rows(args.output_dir / "runtime_frames.csv", runtime_rows)
    live_latency = [
        float(row["pipeline_compute_ms"]) for row in runtime_rows
    ]
    report["latency_ms"] = {
        "pipeline_compute": _latency_summary(live_latency),
        "inference_wall": _latency_summary(
            [float(row["inference_wall_ms"]) for row in runtime_rows]
        ),
        "stereo": _latency_summary(
            [float(row["stereo_ms"]) for row in runtime_rows]
        ),
        "detector_inference": _latency_summary(
            [float(row["detector_inference_ms"]) for row in runtime_rows]
        ),
        "postprocess_ttc": _latency_summary(
            [float(row["postprocess_ttc_ms"]) for row in runtime_rows]
        ),
    }
    report["latency_gate"] = {
        "target_p95_ms": args.latency_target_ms,
        "observed_p95_ms": report["latency_ms"]["pipeline_compute"]["p95"],
        "passed": (
            report["latency_ms"]["pipeline_compute"]["p95"]
            <= args.latency_target_ms
        ),
        "valid_for_deployment": detector is not None,
    }
    report["repeat_determinism"] = {
        "comparisons": prediction_comparisons,
        "prediction_mismatches": nondeterministic_predictions,
        "passed": nondeterministic_predictions == 0,
    }
    report["gpu_memory"] = gpu_memory
    report_path = args.output_dir / "detector_owned_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path(
            "ai_cv/phases/02_detection_tracking/artifacts/"
            "yolo26_reference/detections"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai_cv/outputs/phase05_detector_owned_t03_t05"),
    )
    parser.add_argument(
        "--trips", nargs="+", default=["T03-Sample", "T05-Sample"]
    )
    parser.add_argument(
        "--detector-backend",
        choices=["cached", "yolo26-pytorch", "yolo26-onnx"],
        default="cached",
    )
    parser.add_argument("--model-path", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument(
        "--parallel-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--start-frame-index", type=int, default=0)
    parser.add_argument("--max-frames-per-trip", type=int)
    parser.add_argument("--latency-target-ms", type=float, default=75.0)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument("--stereo-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
