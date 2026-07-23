"""End-to-end TTC ablation for explicit SGBM left-right consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_stereo_confidence import (
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    compute_disparities,
    create_left_matcher,
    create_right_matcher,
    left_right_consistency,
    roi_slices,
)


TRIPS = [f"T0{index}-Sample" for index in range(1, 7)]
MIN_CLOSING_SPEED_MPS = 0.3


@dataclass(frozen=True)
class Variant:
    name: str
    use_lr_consistency: bool
    window: int
    slope_estimator: str
    max_closing_speed_mps: float | None


VARIANTS = (
    Variant("official_replay", False, 5, "ols", None),
    Variant("lr_official_temporal", True, 5, "ols", None),
    Variant("lr_robust_temporal", True, 11, "theil_sen", 20.0),
)


class CausalDepthPolicy:
    def __init__(self, variant: Variant) -> None:
        self.variant = variant
        self.history: Deque[tuple[float, float]] = deque(maxlen=variant.window)

    def update(self, timestamp: float, depth_m: float | None) -> float:
        if depth_m is None or not math.isfinite(depth_m):
            return math.inf
        self.history.append((timestamp, depth_m))
        if len(self.history) < 2:
            return math.inf
        closing_speed = self._closing_speed()
        if closing_speed <= MIN_CLOSING_SPEED_MPS:
            return math.inf
        maximum = self.variant.max_closing_speed_mps
        if maximum is not None and closing_speed > maximum:
            return math.inf
        return float(depth_m / closing_speed)

    def _closing_speed(self) -> float:
        samples = list(self.history)
        times = np.asarray([timestamp for timestamp, _ in samples], dtype=float)
        depths = np.asarray([depth for _, depth in samples], dtype=float)
        if times.max() - times.min() < 1e-3:
            return 0.0
        if self.variant.slope_estimator == "ols":
            matrix = np.vstack([times, np.ones_like(times)]).T
            slope, _ = np.linalg.lstsq(matrix, depths, rcond=None)[0]
            return float(-slope)
        slopes = [
            (depths[j] - depths[i]) / (times[j] - times[i])
            for i in range(len(samples) - 1)
            for j in range(i + 1, len(samples))
            if times[j] > times[i]
        ]
        return float(-np.median(slopes)) if slopes else 0.0


def roi_median_depth(
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    consistency_mask: np.ndarray | None,
) -> float | None:
    roi_y, roi_x = roi_slices(disparity.shape)
    roi_disparity = disparity[roi_y, roi_x]
    valid = np.isfinite(roi_disparity) & (roi_disparity > 0.5)
    if consistency_mask is not None:
        valid &= consistency_mask[roi_y, roi_x]
    depths = focal_length_px * baseline_m / roi_disparity[valid]
    depths = depths[(depths >= MIN_DEPTH_M) & (depths <= MAX_DEPTH_M)]
    if depths.size < 100:
        return None
    return float(np.median(depths))


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_id",
                "timestamp",
                "predicted_ttc",
                "ground_truth_ttc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def process_trip(
    trip_id: str,
    practice_root: Path,
    output_root: Path,
    starter_root: Path,
) -> None:
    sys.path.insert(0, str(starter_root.resolve()))
    from team_kit.dataset_loader import TripDataset

    dataset = TripDataset(practice_root / trip_id)
    calibration = dataset.load_calibration()
    focal_length = float(calibration["K_left"][0][0])
    baseline_m = float(calibration["baseline_m"])
    left_matcher = create_left_matcher()
    right_matcher = create_right_matcher()
    policies = {variant.name: CausalDepthPolicy(variant) for variant in VARIANTS}
    rows = {variant.name: [] for variant in VARIANTS}

    for index, frame in enumerate(dataset.iter_frames()):
        left = dataset.load_left(frame.frame_id)
        right = dataset.load_right(frame.frame_id)
        left_disparity, right_disparity = compute_disparities(
            left, right, left_matcher, right_matcher
        )
        _, consistent, _ = left_right_consistency(
            left_disparity, right_disparity
        )
        raw_depth = roi_median_depth(
            left_disparity, focal_length, baseline_m, None
        )
        consistent_depth = roi_median_depth(
            left_disparity, focal_length, baseline_m, consistent
        )
        for variant in VARIANTS:
            feature = consistent_depth if variant.use_lr_consistency else raw_depth
            prediction = policies[variant.name].update(frame.timestamp, feature)
            rows[variant.name].append(
                {
                    "frame_id": frame.frame_id,
                    "timestamp": round(frame.timestamp, 3),
                    "predicted_ttc": (
                        "inf" if not math.isfinite(prediction) else round(prediction, 3)
                    ),
                    "ground_truth_ttc": (
                        "inf"
                        if not math.isfinite(frame.min_ttc)
                        else round(frame.min_ttc, 3)
                    ),
                }
            )
        if index % 100 == 0:
            print(f"{trip_id}: {index}/599", flush=True)

    for variant in VARIANTS:
        write_predictions(
            output_root / "predictions" / variant.name / f"{trip_id}.csv",
            rows[variant.name],
        )


def evaluate(
    practice_root: Path,
    output_root: Path,
    starter_root: Path,
) -> pd.DataFrame:
    sys.path.insert(0, str(starter_root.resolve()))
    from team_kit.evaluation import evaluate as official_evaluate

    records = []
    for variant in VARIANTS:
        report = official_evaluate(
            output_root / "predictions" / variant.name,
            practice_root,
            output_root / "reports" / f"{variant.name}.json",
        )
        scores = [metric.composite_score for metric in report.per_trip]
        records.append(
            {
                "variant": variant.name,
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
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "variant_summary.csv", index=False)
    (output_root / "variant_manifest.json").write_text(
        json.dumps([variant.__dict__ for variant in VARIANTS], indent=2),
        encoding="utf-8",
    )
    return summary


def verify_official_replay(output_root: Path) -> None:
    reference_root = Path("ai_cv/outputs/predictions/baseline_official")
    if not reference_root.is_dir():
        return
    for trip_id in TRIPS:
        expected = pd.read_csv(reference_root / f"{trip_id}.csv")[
            "predicted_ttc"
        ].to_numpy(float)
        replay = pd.read_csv(
            output_root / "predictions" / "official_replay" / f"{trip_id}.csv"
        )["predicted_ttc"].to_numpy(float)
        finite = np.isfinite(expected)
        if not np.array_equal(finite, np.isfinite(replay)):
            raise AssertionError(f"{trip_id}: official finite/inf mask changed")
        if not np.allclose(expected[finite], replay[finite], atol=0.001):
            raise AssertionError(f"{trip_id}: official replay values changed")


def plot_summary(summary: pd.DataFrame, output_root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    positions = np.arange(len(summary))
    axes[0].bar(
        positions - 0.18,
        summary.overall_composite,
        0.36,
        label="Mean composite",
    )
    axes[0].bar(
        positions + 0.18,
        summary.worst_trip_composite,
        0.36,
        label="Worst trip",
    )
    axes[0].set_xticks(positions, summary.variant, rotation=12)
    axes[0].set_ylabel("Composite score")
    axes[0].set_title("Does LR consistency improve end-to-end TTC?")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.25
    trip_positions = np.arange(len(TRIPS))
    for index, row in summary.iterrows():
        axes[1].bar(
            trip_positions + (index - 1) * width,
            [row[trip_id] for trip_id in TRIPS],
            width,
            label=row.variant,
        )
    axes[1].set_xticks(
        trip_positions, [trip_id.replace("-Sample", "") for trip_id in TRIPS]
    )
    axes[1].set_ylabel("Composite score")
    axes[1].set_title("Per-trip effect")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Stage 2A Component 1 ablation — explicit left-right consistency",
        fontsize=16,
    )
    figure.savefig(output_root / "variant_comparison.png", dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ai_cv/outputs/benchmarks/phase02a_lr_consistency"),
    )
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()

    if not args.reuse_predictions:
        for trip_id in TRIPS:
            process_trip(
                trip_id, args.practice_root, args.output_root, args.starter_root
            )
    verify_official_replay(args.output_root)
    summary = evaluate(args.practice_root, args.output_root, args.starter_root)
    plot_summary(summary, args.output_root)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
