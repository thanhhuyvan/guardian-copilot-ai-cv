"""Evaluate small causal changes to the organizer SGBM TTC baseline.

The expensive SGBM disparity is computed once per frame. Several lightweight
temporal/depth policies then consume the same disparity, which makes comparisons
fast and fair. Ground truth is written only for local evaluation and is never an
input to a prediction policy.
"""

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

import numpy as np
import pandas as pd


TRIPS = [f"T0{i}-Sample" for i in range(1, 7)]
MIN_DEPTH_M = 1.5
MAX_DEPTH_M = 80.0
MIN_CLOSING_SPEED_MPS = 0.3


@dataclass(frozen=True)
class Variant:
    name: str
    feature: str
    window: int
    slope_estimator: str
    max_closing_speed_mps: float | None


VARIANTS = [
    Variant("official_replay", "official_median", 5, "ols", None),
    Variant("robust_median", "official_median", 11, "theil_sen", 20.0),
    Variant("robust_near", "official_p35", 11, "theil_sen", 20.0),
    Variant("robust_corridor", "corridor_p35", 11, "theil_sen", 20.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ai_cv/outputs/benchmarks/lightweight_baseline"),
    )
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Skip SGBM and regenerate reports/charts from existing variant CSVs.",
    )
    return parser.parse_args()


def roi_depth_stat(
    disparity: np.ndarray,
    fx: float,
    baseline_m: float,
    bounds: tuple[float, float, float, float],
    percentile: float,
) -> float | None:
    height, width = disparity.shape
    x0 = int(width * bounds[0])
    x1 = int(width * bounds[1])
    y0 = int(height * bounds[2])
    y1 = int(height * bounds[3])
    roi_disparity = disparity[y0:y1, x0:x1]
    valid = roi_disparity > 0.5
    depths = (fx * baseline_m) / roi_disparity[valid]
    depths = depths[(depths >= MIN_DEPTH_M) & (depths <= MAX_DEPTH_M)]
    if depths.size < 100:
        return None
    return float(np.percentile(depths, percentile))


def extract_features(disparity: np.ndarray, fx: float, baseline_m: float) -> dict[str, float | None]:
    official_bounds = (0.35, 0.65, 0.50, 0.85)
    # The corridor excludes the nearest road strip and gives slightly more width
    # for cut-in objects. It remains a fixed causal ROI, not a detector.
    corridor_bounds = (0.30, 0.70, 0.42, 0.75)
    return {
        "official_median": roi_depth_stat(disparity, fx, baseline_m, official_bounds, 50.0),
        "official_p35": roi_depth_stat(disparity, fx, baseline_m, official_bounds, 35.0),
        "corridor_p35": roi_depth_stat(disparity, fx, baseline_m, corridor_bounds, 35.0),
    }


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
        max_speed = self.variant.max_closing_speed_mps
        if max_speed is not None and closing_speed > max_speed:
            # A discontinuous stereo jump is not accepted as physical motion.
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
        if self.variant.slope_estimator == "theil_sen":
            slopes = [
                (depths[j] - depths[i]) / (times[j] - times[i])
                for i in range(len(samples) - 1)
                for j in range(i + 1, len(samples))
                if times[j] > times[i]
            ]
            return float(-np.median(slopes)) if slopes else 0.0
        raise ValueError(f"Unknown slope estimator: {self.variant.slope_estimator}")


def write_prediction_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_id", "timestamp", "predicted_ttc", "ground_truth_ttc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def process_trip(trip_id: str, practice_root: Path, output_root: Path, starter_root: Path) -> None:
    sys.path.insert(0, str(starter_root.resolve()))
    from team_kit.baseline_ttc_predictor import BaselineTTCPredictor
    from team_kit.dataset_loader import TripDataset

    dataset = TripDataset(practice_root / trip_id)
    calibration = dataset.load_calibration()
    matcher_owner = BaselineTTCPredictor(calibration)
    fx = float(calibration["K_left"][0][0])
    baseline_m = float(calibration["baseline_m"])
    policies = {variant.name: CausalDepthPolicy(variant) for variant in VARIANTS}
    rows = {variant.name: [] for variant in VARIANTS}

    for index, frame in enumerate(dataset.iter_frames()):
        left = dataset.load_left(frame.frame_id)
        right = dataset.load_right(frame.frame_id)
        disparity = matcher_owner._compute_disparity(left, right)
        features = extract_features(disparity, fx, baseline_m)
        for variant in VARIANTS:
            prediction = policies[variant.name].update(frame.timestamp, features[variant.feature])
            rows[variant.name].append(
                {
                    "frame_id": frame.frame_id,
                    "timestamp": round(frame.timestamp, 3),
                    "predicted_ttc": "inf" if not math.isfinite(prediction) else round(prediction, 3),
                    "ground_truth_ttc": "inf" if not math.isfinite(frame.min_ttc) else round(frame.min_ttc, 3),
                }
            )
        if index % 100 == 0:
            print(f"{trip_id}: processed frame {index}/599", flush=True)

    for variant in VARIANTS:
        write_prediction_csv(output_root / "predictions" / variant.name / f"{trip_id}.csv", rows[variant.name])


def evaluate_variants(practice_root: Path, output_root: Path, starter_root: Path) -> pd.DataFrame:
    sys.path.insert(0, str(starter_root.resolve()))
    from team_kit.evaluation import evaluate

    records: list[dict[str, object]] = []
    for variant in VARIANTS:
        report = evaluate(
            output_root / "predictions" / variant.name,
            practice_root,
            output_root / "reports" / f"{variant.name}.json",
        )
        scores = [metric.composite_score for metric in report.per_trip]
        records.append(
            {
                "variant": variant.name,
                "feature": variant.feature,
                "window": variant.window,
                "slope_estimator": variant.slope_estimator,
                "max_closing_speed_mps": variant.max_closing_speed_mps,
                "overall_mae_critical": report.overall_mae_critical,
                "overall_inv_ttc_mae": report.overall_inv_ttc_mae,
                "overall_f1": report.overall_f1,
                "overall_composite": report.overall_composite_score,
                "worst_trip_composite": min(scores),
                "best_trip_composite": max(scores),
                **{metric.trip_id: metric.composite_score for metric in report.per_trip},
            }
        )
    frame = pd.DataFrame(records).sort_values("overall_composite", ascending=False)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "variant_summary.csv", index=False)
    (output_root / "variant_manifest.json").write_text(
        json.dumps([variant.__dict__ for variant in VARIANTS], indent=2),
        encoding="utf-8",
    )
    return frame


def verify_official_replay(output_root: Path) -> None:
    official_root = Path("ai_cv/outputs/predictions/baseline_official")
    if not official_root.exists():
        return
    for trip_id in TRIPS:
        expected = pd.read_csv(official_root / f"{trip_id}.csv")["predicted_ttc"].to_numpy(float)
        replayed = pd.read_csv(output_root / "predictions" / "official_replay" / f"{trip_id}.csv")[
            "predicted_ttc"
        ].to_numpy(float)
        finite = np.isfinite(expected)
        if not np.array_equal(finite, np.isfinite(replayed)):
            raise AssertionError(f"{trip_id}: replay finite/inf mask differs from official run")
        if not np.allclose(expected[finite], replayed[finite], atol=0.001):
            raise AssertionError(f"{trip_id}: replay numeric predictions differ from official run")


def plot_comparison(summary: pd.DataFrame, output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = summary.sort_values("overall_composite")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    colors = ["#888888" if name == "official_replay" else "#2c7fb8" for name in ordered.variant]
    axes[0].barh(ordered.variant, ordered.overall_composite, color=colors, label="Mean composite")
    axes[0].scatter(ordered.worst_trip_composite, ordered.variant, color="#d62728", s=60, label="Worst trip")
    for index, row in ordered.reset_index(drop=True).iterrows():
        axes[0].text(row.overall_composite + 0.4, index, f"{row.overall_composite:.1f}", va="center")
    axes[0].set_xlim(0, max(45, ordered.overall_composite.max() + 6))
    axes[0].set_xlabel("Composite score")
    axes[0].set_title("Mean and worst-trip composite")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.2)

    x = np.arange(len(TRIPS))
    width = 0.19
    for index, row in summary.sort_values("variant").reset_index(drop=True).iterrows():
        axes[1].bar(x + (index - 1.5) * width, [row[trip] for trip in TRIPS], width, label=row.variant)
    axes[1].set_xticks(x, [trip.replace("-Sample", "") for trip in TRIPS])
    axes[1].set_ylabel("Composite score")
    axes[1].set_title("Per-trip score: improvement is not uniform")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Lightweight causal changes versus organizer baseline", fontsize=16)
    fig.tight_layout()
    chart_dir = output_root / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_dir / "variant_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_timeline_comparison(output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(17, 12), sharex=True, sharey=True)
    for axis, trip_id in zip(axes.flat, TRIPS, strict=True):
        official = pd.read_csv(output_root / "predictions" / "official_replay" / f"{trip_id}.csv")
        improved = pd.read_csv(output_root / "predictions" / "robust_corridor" / f"{trip_id}.csv")
        seconds = official.timestamp.to_numpy(float)

        def display(series: pd.Series) -> np.ndarray:
            values = series.to_numpy(float)
            return np.where(np.isfinite(values), np.minimum(values, 10.0), 10.0)

        axis.axhspan(0, 2, color="#ffdddd", alpha=0.8)
        axis.axhspan(2, 3, color="#fff1cc", alpha=0.7)
        axis.plot(seconds, display(official.ground_truth_ttc), color="#111111", linewidth=2, label="GT")
        axis.plot(seconds, display(official.predicted_ttc), color="#d62728", linewidth=0.9, alpha=0.55, label="Official")
        axis.plot(seconds, display(improved.predicted_ttc), color="#1f77b4", linewidth=1.1, alpha=0.9, label="Robust corridor")
        axis.set_title(trip_id)
        axis.set_xlim(0, 30)
        axis.set_ylim(0, 10.2)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="upper left", ncols=3, fontsize=8)
    fig.supxlabel("Trip time (seconds)")
    fig.supylabel("TTC capped at 10 s; inf shown at 10 s")
    fig.suptitle("Official baseline versus lightweight robust-corridor variant", fontsize=16)
    fig.tight_layout()
    chart_dir = output_root / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_dir / "robust_corridor_timelines.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.reuse_predictions:
        for trip_id in TRIPS:
            process_trip(trip_id, args.practice_root, args.output_root, args.starter_root)
    verify_official_replay(args.output_root)
    summary = evaluate_variants(args.practice_root, args.output_root, args.starter_root)
    plot_comparison(summary, args.output_root)
    plot_timeline_comparison(args.output_root)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
