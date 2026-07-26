"""Full TTC experiment for the classical object-centric vertical slice."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_stereo_confidence import (
    compute_cropped_disparities_with_timing,
    configure_opencv_threads,
    create_left_matcher,
    create_right_matcher,
    left_right_consistency,
)
from classical_geometry import (
    collision_corridor_mask,
    estimate_ground_model,
    extract_obstacle_components,
    ground_and_obstacle_masks,
)
from classical_tracking import ComponentTracker, select_minimum_ttc


TRIPS = [f"T0{index}-Sample" for index in range(1, 7)]
VARIANTS = ("scene_p20", "track_p20", "track_p35", "track_median")


@dataclass(frozen=True)
class RuntimeRecord:
    """Per-frame milliseconds; total_compute_ms explicitly excludes image I/O."""

    trip_id: str
    frame_id: int
    io_ms: float
    stereo_pair_ms: float
    left_match_ms: float
    right_match_ms: float
    lr_consistency_ms: float
    ground_ms: float
    components_ms: float
    tracking_ms: float
    total_compute_ms: float
    end_to_end_with_io_ms: float


class SceneDepthPolicy:
    """Unassociated nearest-component history for the tracking ablation."""

    def __init__(self) -> None:
        self.history = deque(maxlen=11)

    def update(self, timestamp: float, depth_m: float | None) -> float:
        if depth_m is None or not math.isfinite(depth_m):
            return math.inf
        self.history.append((timestamp, depth_m))
        if len(self.history) < 3:
            return math.inf
        samples = list(self.history)
        slopes = [
            (samples[j][1] - samples[i][1])
            / (samples[j][0] - samples[i][0])
            for i in range(len(samples) - 1)
            for j in range(i + 1, len(samples))
            if samples[j][0] > samples[i][0]
        ]
        closing_speed = -float(np.median(slopes))
        if closing_speed <= 0.3 or closing_speed > 40.0:
            return math.inf
        return float(depth_m / closing_speed)


def component_in_risk_corridor(component, corridor: np.ndarray) -> bool:
    height, width = corridor.shape
    center_x = int(np.clip(component.center_x, 0, width - 1))
    bottom_y = int(np.clip(component.bottom_y - 1, 0, height - 1))
    return bool(corridor[bottom_y, center_x])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_trip(
    trip_id: str,
    practice_root: Path,
    output_root: Path,
    starter_root: Path,
    stereo_executor: Executor | None = None,
    stereo_roi_top: int = 0,
) -> None:
    sys.path.insert(0, str(starter_root.resolve()))
    from team_kit.dataset_loader import TripDataset

    dataset = TripDataset(practice_root / trip_id)
    calibration = dataset.load_calibration()
    focal_length = float(calibration["K_left"][0][0])
    baseline_m = float(calibration["baseline_m"])
    left_matcher = create_left_matcher()
    right_matcher = create_right_matcher()
    image_shape = (int(calibration["image_height"]), int(calibration["image_width"]))
    risk_corridor = collision_corridor_mask(
        image_shape,
        top_width_fraction=0.16,
        bottom_width_fraction=0.55,
    )
    scene_policy = SceneDepthPolicy()
    trackers = {
        "track_p20": ComponentTracker(image_shape, depth_attribute="depth_p20_m"),
        "track_p35": ComponentTracker(image_shape, depth_attribute="depth_p35_m"),
        "track_median": ComponentTracker(image_shape, depth_attribute="depth_m"),
    }
    prediction_rows = {variant: [] for variant in VARIANTS}
    diagnostic_rows = []
    runtime_rows = []

    for index, frame in enumerate(dataset.iter_frames()):
        frame_started = time.perf_counter()
        left = dataset.load_left(frame.frame_id)
        right = dataset.load_right(frame.frame_id)
        io_ms = (time.perf_counter() - frame_started) * 1000.0

        compute_started = time.perf_counter()
        started = time.perf_counter()
        (
            left_disparity,
            right_disparity,
            left_match_ms,
            right_match_ms,
        ) = compute_cropped_disparities_with_timing(
            left,
            right,
            left_matcher,
            right_matcher,
            roi_top=stereo_roi_top,
            executor=stereo_executor,
        )
        stereo_pair_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        _, lr_consistent, _ = left_right_consistency(
            left_disparity, right_disparity
        )
        lr_consistency_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        ground_model, _ = estimate_ground_model(left_disparity)
        ground_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        components = []
        if ground_model is not None:
            _, obstacle_evidence, _ = ground_and_obstacle_masks(
                left_disparity, ground_model
            )
            components, _, _ = extract_obstacle_components(
                left_disparity,
                obstacle_evidence,
                lr_consistent,
                focal_length,
                baseline_m,
            )
        components_ms = (time.perf_counter() - started) * 1000.0
        ground_confidence = (
            ground_model.confidence if ground_model is not None else 0.0
        )

        started = time.perf_counter()
        relevant_components = [
            component
            for component in components
            if component_in_risk_corridor(component, risk_corridor)
        ]
        nearest_depth = (
            min(component.depth_p20_m for component in relevant_components)
            if relevant_components
            else None
        )
        predictions = {
            "scene_p20": scene_policy.update(frame.timestamp, nearest_depth)
        }
        selected = {
            "scene_p20": (
                None,
                0.0,
                0.0,
                nearest_depth if nearest_depth is not None else math.nan,
            )
        }
        for variant, tracker in trackers.items():
            current_tracks = tracker.update(components, frame.timestamp)
            risk_tracks = tracker.risk_tracks(current_tracks)
            ttc, track_id, confidence, closing_speed = select_minimum_ttc(
                risk_tracks, ground_confidence
            )
            predictions[variant] = ttc
            selected_track = next(
                (track for track in risk_tracks if track.track_id == track_id),
                None,
            )
            selected[variant] = (
                track_id,
                confidence,
                closing_speed,
                (
                    selected_track.latest.depth_m
                    if selected_track is not None
                    else math.nan
                ),
            )
        tracking_ms = (time.perf_counter() - started) * 1000.0
        compute_finished = time.perf_counter()
        total_compute_ms = (compute_finished - compute_started) * 1000.0
        end_to_end_with_io_ms = (compute_finished - frame_started) * 1000.0

        gt_value = (
            "inf" if not math.isfinite(frame.min_ttc) else round(frame.min_ttc, 3)
        )
        for variant in VARIANTS:
            prediction = predictions[variant]
            prediction_rows[variant].append(
                {
                    "frame_id": frame.frame_id,
                    "timestamp": round(frame.timestamp, 3),
                    "predicted_ttc": (
                        "inf"
                        if not math.isfinite(prediction)
                        else round(prediction, 3)
                    ),
                    "ground_truth_ttc": gt_value,
                }
            )
            track_id, confidence, closing_speed, depth_m = selected[variant]
            diagnostic_rows.append(
                {
                    "frame_id": frame.frame_id,
                    "timestamp": round(frame.timestamp, 3),
                    "variant": variant,
                    "component_count": len(components),
                    "relevant_component_count": len(relevant_components),
                    "ground_confidence": round(ground_confidence, 6),
                    "selected_track_id": track_id,
                    "selected_depth_m": (
                        "" if not math.isfinite(depth_m) else round(depth_m, 4)
                    ),
                    "closing_speed_mps": round(closing_speed, 4),
                    "prediction_confidence": round(confidence, 6),
                    "predicted_ttc": (
                        "inf"
                        if not math.isfinite(prediction)
                        else round(prediction, 4)
                    ),
                }
            )

        runtime_rows.append(
            asdict(
                RuntimeRecord(
                    trip_id=trip_id,
                    frame_id=frame.frame_id,
                    io_ms=io_ms,
                    stereo_pair_ms=stereo_pair_ms,
                    left_match_ms=left_match_ms,
                    right_match_ms=right_match_ms,
                    lr_consistency_ms=lr_consistency_ms,
                    ground_ms=ground_ms,
                    components_ms=components_ms,
                    tracking_ms=tracking_ms,
                    total_compute_ms=total_compute_ms,
                    end_to_end_with_io_ms=end_to_end_with_io_ms,
                )
            )
        )
        if index % 100 == 0:
            print(
                f"{trip_id}: {index}/599 components={len(components)} "
                f"ground_q={ground_confidence:.2f}",
                flush=True,
            )

    for variant in VARIANTS:
        write_csv(
            output_root / "predictions" / variant / f"{trip_id}.csv",
            prediction_rows[variant],
            ["frame_id", "timestamp", "predicted_ttc", "ground_truth_ttc"],
        )
    write_csv(
        output_root / "diagnostics" / f"{trip_id}.csv",
        diagnostic_rows,
        list(diagnostic_rows[0].keys()),
    )
    write_csv(
        output_root / "runtime" / f"{trip_id}.csv",
        runtime_rows,
        list(runtime_rows[0].keys()),
    )


def evaluate_variants(
    practice_root: Path,
    output_root: Path,
    starter_root: Path,
) -> pd.DataFrame:
    sys.path.insert(0, str(starter_root.resolve()))
    from team_kit.evaluation import evaluate

    records = []
    for variant in VARIANTS:
        report = evaluate(
            output_root / "predictions" / variant,
            practice_root,
            output_root / "reports" / f"{variant}.json",
        )
        scores = [metric.composite_score for metric in report.per_trip]
        records.append(
            {
                "variant": variant,
                "overall_mae_critical": report.overall_mae_critical,
                "overall_inv_ttc_mae": report.overall_inv_ttc_mae,
                "overall_f1": report.overall_f1,
                "overall_composite": report.overall_composite_score,
                "worst_trip_composite": min(scores),
                **{
                    metric.trip_id: metric.composite_score
                    for metric in report.per_trip
                },
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(output_root / "variant_summary.csv", index=False)
    return summary


def aggregate_runtime(output_root: Path) -> dict:
    frame = pd.concat(
        [pd.read_csv(path) for path in sorted((output_root / "runtime").glob("*.csv"))],
        ignore_index=True,
    )
    columns = [
        "io_ms",
        "stereo_pair_ms",
        "left_match_ms",
        "right_match_ms",
        "lr_consistency_ms",
        "ground_ms",
        "components_ms",
        "tracking_ms",
        "total_compute_ms",
        "end_to_end_with_io_ms",
    ]
    report = {
        column: {
            "p50": float(frame[column].quantile(0.50)),
            "p95": float(frame[column].quantile(0.95)),
            "p99": float(frame[column].quantile(0.99)),
            "mean": float(frame[column].mean()),
        }
        for column in columns
    }
    (output_root / "runtime_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def plot_results(
    summary: pd.DataFrame,
    output_root: Path,
) -> None:
    reference = pd.DataFrame(
        [
            {
                "variant": "official_reference",
                "overall_composite": 19.7,
                "worst_trip_composite": 5.0,
                "T01-Sample": 30.6,
                "T02-Sample": 12.2,
                "T03-Sample": 5.0,
                "T04-Sample": 38.2,
                "T05-Sample": 16.0,
                "T06-Sample": 16.2,
            },
            {
                "variant": "robust_roi_reference",
                "overall_composite": 32.7,
                "worst_trip_composite": 22.1,
                "T01-Sample": 33.8,
                "T02-Sample": 57.2,
                "T03-Sample": 22.1,
                "T04-Sample": 36.5,
                "T05-Sample": 22.2,
                "T06-Sample": 24.5,
            },
        ]
    )
    combined = pd.concat([reference, summary], ignore_index=True)
    figure, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    positions = np.arange(len(combined))
    axes[0].bar(
        positions - 0.18,
        combined.overall_composite,
        0.36,
        label="mean",
    )
    axes[0].bar(
        positions + 0.18,
        combined.worst_trip_composite,
        0.36,
        label="worst trip",
    )
    axes[0].set_xticks(positions, combined.variant, rotation=18)
    axes[0].set_ylabel("Composite score")
    axes[0].set_title("Vertical slice versus frozen references")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.13
    trip_positions = np.arange(len(TRIPS))
    for index, row in combined.iterrows():
        axes[1].bar(
            trip_positions + (index - (len(combined) - 1) / 2) * width,
            [row[trip] for trip in TRIPS],
            width,
            label=row.variant,
        )
    axes[1].set_xticks(
        trip_positions, [trip.replace("-Sample", "") for trip in TRIPS]
    )
    axes[1].set_ylabel("Composite score")
    axes[1].set_title("Per-trip effect")
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Stage 2A high-impact vertical slice: components → tracks → TTC",
        fontsize=16,
    )
    figure.savefig(output_root / "variant_comparison.png", dpi=170)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ai_cv/outputs/benchmarks/phase02a_vertical_slice"),
    )
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument(
        "--stereo-workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="Set to 2 to compute left and right disparity concurrently.",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=6,
        help="OpenCV worker threads available to each SGBM matcher.",
    )
    parser.add_argument(
        "--stereo-roi-top",
        type=int,
        default=0,
        help=(
            "Top crop in native-image rows; 0 preserves the frozen full-frame "
            "reference, and 96 is the single Phase 2B ROI candidate."
        ),
    )
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be positive")
    if args.stereo_roi_top < 0 or args.stereo_roi_top >= 360:
        parser.error("--stereo-roi-top must be in [0, 359]")
    configure_opencv_threads(args.opencv_threads)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_configuration.json").write_text(
        json.dumps(
            {
                "opencv_threads": args.opencv_threads,
                "stereo_workers": args.stereo_workers,
                "stereo_roi_top": args.stereo_roi_top,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.reuse_predictions:
        executor_context = (
            ThreadPoolExecutor(max_workers=2)
            if args.stereo_workers == 2
            else nullcontext(None)
        )
        with executor_context as stereo_executor:
            for trip_id in TRIPS:
                process_trip(
                    trip_id,
                    args.practice_root,
                    args.output_root,
                    args.starter_root,
                    stereo_executor=stereo_executor,
                    stereo_roi_top=args.stereo_roi_top,
                )
    summary = evaluate_variants(
        args.practice_root, args.output_root, args.starter_root
    )
    runtime = aggregate_runtime(args.output_root)
    plot_results(summary, args.output_root)
    print(summary.to_string(index=False))
    print(json.dumps(runtime, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
