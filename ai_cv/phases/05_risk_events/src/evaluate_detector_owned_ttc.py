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


def _ttc_text(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.6f}"


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
    report: dict[str, object] = {
        "experiment": "phase05a_detector_owned_ttc",
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
    try:
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
            detections_by_frame = load_detections_csv(
                args.detections_dir / f"{trip_id}.csv"
            )
            tracker = ComponentTracker(
                image_shape,
                depth_attribute="object_depth_m",
                maximum_missed=3,
                risk_top_width_fraction=0.10,
                risk_bottom_width_fraction=0.50,
                minimum_bottom_fraction=0.45,
                minimum_height_fraction=0.025,
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
            for index, frame in enumerate(dataset.iter_frames()):
                left = dataset.load_left(frame.frame_id)
                right = dataset.load_right(frame.frame_id)
                stereo = backend.infer(left, right)
                ground_model, _ = estimate_ground_model(stereo.disparity_px)
                ground_confidence = (
                    float(ground_model.confidence)
                    if ground_model is not None
                    else 0.0
                )
                detections = detections_by_frame.get(int(frame.frame_id), [])
                stats["detections"] += len(detections)
                components = [
                    component
                    for component_id, detection in enumerate(detections, start=1)
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
                stats["depth_valid_components"] += len(components)
                stats["frames_with_depth_component"] += int(bool(components))
                tracks = tracker.update(components, float(frame.timestamp))
                risk_tracks = tracker.risk_tracks(tracks)
                stats["frames_with_risk_track"] += int(bool(risk_tracks))

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
                ego_cap = min(20.0, max(3.0, float(frame.speed_kmh) / 3.6 + 3.0))
                capped_ttc, _, _, _ = select_minimum_ttc(
                    risk_tracks,
                    ground_confidence,
                    maximum_closing_speed_mps=ego_cap,
                    **common,
                )
                policies["detector_owned"].append(float(plain_ttc))
                policies["detector_owned_ego_cap"].append(float(capped_ttc))
                truth.append(float(frame.min_ttc))
                if args.progress_every and (
                    (index + 1) % args.progress_every == 0
                    or index + 1 == len(dataset)
                ):
                    print(f"{trip_id}: {index + 1}/{len(dataset)}", flush=True)

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
                        dataset.iter_frames(), predictions, strict=True
                    )
                ]
                _write_predictions(
                    args.output_dir / policy_name / f"{trip_id}.csv", rows
                )
            report["trips"][trip_id] = trip_report
    finally:
        backend.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument("--stereo-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
