"""Analyze SGBM disparity quality before obstacle extraction.

This is Stage 2A component 1. It reproduces the organizer's left disparity,
computes an explicit right disparity, applies a causal left-right consistency
check, and creates visual/CSV evidence. Ground-truth TTC is used only as a
failure-case annotation; provided depth keyframes are validation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_DISPARITIES = 96
BLOCK_SIZE = 11
MIN_DEPTH_M = 1.5
MAX_DEPTH_M = 80.0
LR_THRESHOLD_PX = 1.0
BASELINE_ROI = (0.35, 0.65, 0.50, 0.85)


@dataclass(frozen=True)
class FailureCase:
    trip_id: str
    frame_id: int
    outcome: str
    ground_truth_ttc: float
    predicted_ttc: float
    description: str


@dataclass(frozen=True)
class FrameMetrics:
    trip_id: str
    frame_id: int
    valid_fraction: float
    consistent_fraction: float
    consistent_of_valid: float
    roi_valid_fraction: float
    roi_consistent_fraction: float
    roi_consistent_of_valid: float
    roi_raw_depth_median_m: float
    roi_consistent_depth_median_m: float
    roi_lr_residual_median_px: float
    roi_lr_residual_p95_px: float
    reference_valid_pixels: int
    raw_depth_abs_rel: float
    consistent_depth_abs_rel: float
    left_match_ms: float
    right_match_ms: float
    stereo_pair_ms: float


DEFAULT_CASES = (
    FailureCase(
        "T01-Sample",
        324,
        "FN",
        1.060,
        math.inf,
        "Pedestrian threat is erased by the mixed-ROI median.",
    ),
    FailureCase(
        "T03-Sample",
        293,
        "FP",
        math.inf,
        0.172,
        "Empty-road disparity jump becomes an impossible closing speed.",
    ),
    FailureCase(
        "T04-Sample",
        265,
        "TP",
        1.751,
        1.706,
        "Stable coherent depth trend is a successful reference.",
    ),
    FailureCase(
        "T05-Sample",
        314,
        "FP",
        math.inf,
        0.285,
        "Mixed pixels create a false 21.30 m/s closing speed.",
    ),
    FailureCase(
        "T05-Sample",
        469,
        "FN",
        1.441,
        math.inf,
        "Noisy depth trend estimates a receding target.",
    ),
    FailureCase(
        "T06-Sample",
        146,
        "FN",
        0.836,
        math.inf,
        "Motorcycle depth reverses across the five-frame history.",
    ),
)


def create_left_matcher() -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def create_right_matcher() -> cv2.StereoSGBM:
    # Right-view disparity is x_right - x_left and is therefore negative.
    return cv2.StereoSGBM_create(
        minDisparity=-NUM_DISPARITIES,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparities(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_matcher: cv2.StereoSGBM,
    right_matcher: cv2.StereoSGBM,
) -> tuple[np.ndarray, np.ndarray]:
    left_disparity, right_disparity, _, _ = compute_disparities_with_timing(
        left_bgr, right_bgr, left_matcher, right_matcher
    )
    return left_disparity, right_disparity


def compute_disparities_with_timing(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_matcher: cv2.StereoSGBM,
    right_matcher: cv2.StereoSGBM,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    started = time.perf_counter()
    left_disparity = (
        left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    )
    left_match_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    right_disparity = (
        right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
    )
    right_match_ms = (time.perf_counter() - started) * 1000.0
    return left_disparity, right_disparity, left_match_ms, right_match_ms


def left_right_consistency(
    left_disparity: np.ndarray,
    right_disparity: np.ndarray,
    threshold_px: float = LR_THRESHOLD_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid-left mask, consistency mask and residual in left coordinates."""
    if left_disparity.shape != right_disparity.shape:
        raise ValueError("Left and right disparity shapes must match")
    height, width = left_disparity.shape
    x_left = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    x_right = np.rint(x_left - left_disparity).astype(np.int32)

    valid_left = np.isfinite(left_disparity) & (left_disparity > 0.5)
    in_bounds = (x_right >= 0) & (x_right < width)
    sample_x = np.clip(x_right, 0, width - 1)
    rows = np.arange(height)[:, None]
    sampled_right = right_disparity[rows, sample_x]
    valid_right = np.isfinite(sampled_right) & (sampled_right < -0.5)

    residual = np.full(left_disparity.shape, np.nan, dtype=np.float32)
    comparable = valid_left & in_bounds & valid_right
    residual[comparable] = np.abs(
        left_disparity[comparable] + sampled_right[comparable]
    )
    consistent = comparable & (residual <= threshold_px)
    return valid_left, consistent, residual


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
) -> np.ndarray:
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0.5)
    depth[valid] = focal_length_px * baseline_m / disparity[valid]
    depth[(depth < MIN_DEPTH_M) | (depth > MAX_DEPTH_M)] = np.nan
    return depth


def roi_slices(shape: tuple[int, int]) -> tuple[slice, slice]:
    height, width = shape
    x0, x1, y0, y1 = BASELINE_ROI
    return (
        slice(int(height * y0), int(height * y1)),
        slice(int(width * x0), int(width * x1)),
    )


def finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else math.nan


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else math.nan


def absolute_relative_error(
    estimated_depth: np.ndarray,
    reference_depth: np.ndarray,
    extra_mask: np.ndarray | None = None,
) -> tuple[int, float]:
    mask = (
        np.isfinite(estimated_depth)
        & np.isfinite(reference_depth)
        & (reference_depth >= MIN_DEPTH_M)
        & (reference_depth <= MAX_DEPTH_M)
        & (reference_depth < 999.0)
    )
    if extra_mask is not None:
        mask &= extra_mask
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0, math.nan
    error = np.abs(estimated_depth[mask] - reference_depth[mask]) / reference_depth[mask]
    return count, float(np.median(error))


def load_calibration(trip_dir: Path) -> tuple[float, float]:
    path = trip_dir / "kitti" / "calibration_info.txt"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    return float(calibration["K_left"][0][0]), float(calibration["baseline_m"])


def read_stereo(trip_dir: Path, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    left_path = trip_dir / "kitti" / "image_2" / f"{frame_id:06d}.jpg"
    right_path = trip_dir / "kitti" / "image_3" / f"{frame_id:06d}.jpg"
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Missing stereo pair: {left_path}, {right_path}")
    return left, right


def load_reference_depth(trip_dir: Path, frame_id: int) -> np.ndarray | None:
    path = trip_dir / "kitti" / "depth" / f"{frame_id:06d}.npy"
    return np.load(path) if path.is_file() else None


def compute_frame_metrics(
    trip_id: str,
    frame_id: int,
    left_disparity: np.ndarray,
    right_disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    reference_depth: np.ndarray | None,
    left_match_ms: float = math.nan,
    right_match_ms: float = math.nan,
) -> tuple[FrameMetrics, dict[str, np.ndarray]]:
    valid, consistent, residual = left_right_consistency(
        left_disparity, right_disparity
    )
    depth = disparity_to_depth(left_disparity, focal_length_px, baseline_m)
    depth_consistent = np.where(consistent, depth, np.nan)
    roi_y, roi_x = roi_slices(left_disparity.shape)
    roi_valid = valid[roi_y, roi_x]
    roi_consistent = consistent[roi_y, roi_x]
    roi_residual = residual[roi_y, roi_x]
    roi_depth = depth[roi_y, roi_x]
    roi_depth_consistent = depth_consistent[roi_y, roi_x]

    valid_count = int(np.count_nonzero(valid))
    roi_valid_count = int(np.count_nonzero(roi_valid))
    reference_pixels = 0
    raw_abs_rel = math.nan
    consistent_abs_rel = math.nan
    if reference_depth is not None:
        reference_pixels, raw_abs_rel = absolute_relative_error(depth, reference_depth)
        _, consistent_abs_rel = absolute_relative_error(
            depth, reference_depth, consistent
        )

    metrics = FrameMetrics(
        trip_id=trip_id,
        frame_id=frame_id,
        valid_fraction=float(np.mean(valid)),
        consistent_fraction=float(np.mean(consistent)),
        consistent_of_valid=(
            float(np.count_nonzero(consistent) / valid_count)
            if valid_count
            else math.nan
        ),
        roi_valid_fraction=float(np.mean(roi_valid)),
        roi_consistent_fraction=float(np.mean(roi_consistent)),
        roi_consistent_of_valid=(
            float(np.count_nonzero(roi_consistent) / roi_valid_count)
            if roi_valid_count
            else math.nan
        ),
        roi_raw_depth_median_m=finite_median(roi_depth),
        roi_consistent_depth_median_m=finite_median(roi_depth_consistent),
        roi_lr_residual_median_px=finite_median(roi_residual),
        roi_lr_residual_p95_px=finite_percentile(roi_residual, 95.0),
        reference_valid_pixels=reference_pixels,
        raw_depth_abs_rel=raw_abs_rel,
        consistent_depth_abs_rel=consistent_abs_rel,
        left_match_ms=left_match_ms,
        right_match_ms=right_match_ms,
        stereo_pair_ms=left_match_ms + right_match_ms,
    )
    arrays = {
        "valid": valid,
        "consistent": consistent,
        "residual": residual,
        "depth": depth,
        "depth_consistent": depth_consistent,
    }
    return metrics, arrays


def format_ttc(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.3f}s"


def save_case_visual(
    output_path: Path,
    case: FailureCase,
    left_bgr: np.ndarray,
    left_disparity: np.ndarray,
    arrays: dict[str, np.ndarray],
    metrics: FrameMetrics,
) -> None:
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    roi_y, roi_x = roi_slices(left_disparity.shape)
    y0, y1 = roi_y.start, roi_y.stop
    x0, x1 = roi_x.start, roi_x.stop

    confidence_overlay = np.zeros((*left_disparity.shape, 3), dtype=np.uint8)
    confidence_overlay[arrays["valid"]] = (235, 80, 70)
    confidence_overlay[arrays["consistent"]] = (45, 200, 90)
    confidence_view = cv2.addWeighted(
        left_rgb, 0.55, confidence_overlay, 0.45, 0.0
    )

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    axes[0, 0].imshow(left_rgb)
    axes[0, 0].add_patch(
        plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, color="yellow", linewidth=2
        )
    )
    axes[0, 0].set_title("Left image + organizer ROI")

    disparity_view = np.ma.masked_where(left_disparity <= 0.5, left_disparity)
    image = axes[0, 1].imshow(disparity_view, cmap="turbo", vmin=0, vmax=64)
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046, label="disparity (px)")
    axes[0, 1].set_title("Raw left SGBM disparity")

    residual_view = np.ma.masked_invalid(arrays["residual"])
    image = axes[0, 2].imshow(residual_view, cmap="magma", vmin=0, vmax=4)
    figure.colorbar(image, ax=axes[0, 2], fraction=0.046, label="LR residual (px)")
    axes[0, 2].set_title("Left-right residual")

    axes[1, 0].imshow(confidence_view)
    axes[1, 0].set_title("Consistency overlay: green=pass, red=reject")

    depth_view = np.ma.masked_invalid(arrays["depth_consistent"])
    image = axes[1, 1].imshow(depth_view, cmap="viridis_r", vmin=MIN_DEPTH_M, vmax=40)
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, label="depth (m)")
    axes[1, 1].set_title("Depth after LR consistency")

    raw_roi = arrays["depth"][roi_y, roi_x]
    consistent_roi = arrays["depth_consistent"][roi_y, roi_x]
    raw_values = raw_roi[np.isfinite(raw_roi)]
    consistent_values = consistent_roi[np.isfinite(consistent_roi)]
    bins = np.linspace(MIN_DEPTH_M, 40.0, 60)
    if raw_values.size:
        axes[1, 2].hist(
            raw_values,
            bins=bins,
            alpha=0.45,
            density=True,
            label=f"raw n={raw_values.size}",
        )
    if consistent_values.size:
        axes[1, 2].hist(
            consistent_values,
            bins=bins,
            alpha=0.55,
            density=True,
            label=f"consistent n={consistent_values.size}",
        )
    axes[1, 2].axvline(
        metrics.roi_raw_depth_median_m, color="tab:blue", linestyle="--"
    )
    axes[1, 2].axvline(
        metrics.roi_consistent_depth_median_m, color="tab:orange", linestyle="--"
    )
    axes[1, 2].set_xlim(MIN_DEPTH_M, 40.0)
    axes[1, 2].set_xlabel("depth (m)")
    axes[1, 2].set_title("Organizer ROI depth distribution")
    axes[1, 2].legend(fontsize=8)

    for axis in axes.flat[:5]:
        axis.axis("off")
    figure.suptitle(
        f"{case.trip_id} frame {case.frame_id} [{case.outcome}]  "
        f"GT={format_ttc(case.ground_truth_ttc)}  "
        f"baseline={format_ttc(case.predicted_ttc)}\n"
        f"ROI valid={metrics.roi_valid_fraction:.1%}, "
        f"LR-pass/valid={metrics.roi_consistent_of_valid:.1%}, "
        f"raw median={metrics.roi_raw_depth_median_m:.2f}m, "
        f"consistent median={metrics.roi_consistent_depth_median_m:.2f}m",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_summary_visual(
    output_path: Path,
    records: list[tuple[FailureCase, np.ndarray, dict[str, np.ndarray], FrameMetrics]],
) -> None:
    figure, axes = plt.subplots(
        len(records), 2, figsize=(14, 3.25 * len(records)), constrained_layout=True
    )
    for row, (case, left_bgr, arrays, metrics) in enumerate(records):
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        overlay = np.zeros((*arrays["valid"].shape, 3), dtype=np.uint8)
        overlay[arrays["valid"]] = (235, 80, 70)
        overlay[arrays["consistent"]] = (45, 200, 90)
        confidence_view = cv2.addWeighted(left_rgb, 0.55, overlay, 0.45, 0.0)

        axes[row, 0].imshow(left_rgb)
        axes[row, 0].set_title(
            f"{case.trip_id} #{case.frame_id} {case.outcome}: "
            f"GT {format_ttc(case.ground_truth_ttc)}, "
            f"pred {format_ttc(case.predicted_ttc)}"
        )
        axes[row, 1].imshow(confidence_view)
        axes[row, 1].set_title(
            f"LR pass/valid {metrics.consistent_of_valid:.1%}; "
            f"ROI {metrics.roi_consistent_of_valid:.1%}"
        )
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    figure.suptitle(
        "Stage 2A Component 1 — explicit SGBM left-right consistency\n"
        "green = geometrically consistent, red = left disparity rejected",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_sampled_summary(output_path: Path, metrics: list[FrameMetrics]) -> None:
    frame = pd.DataFrame([asdict(item) for item in metrics])
    trip_ids = sorted(frame["trip_id"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(trip_ids)))
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].boxplot(
        [
            frame.loc[frame.trip_id == trip_id, "roi_consistent_of_valid"]
            for trip_id in trip_ids
        ],
        tick_labels=[trip_id.replace("-Sample", "") for trip_id in trip_ids],
    )
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("LR-consistent / valid ROI pixels")
    axes[0, 0].set_title("ROI consistency varies strongly by trip/frame")
    axes[0, 0].grid(axis="y", alpha=0.25)

    raw_by_trip = frame.groupby("trip_id")["raw_depth_abs_rel"].median()
    filtered_by_trip = frame.groupby("trip_id")["consistent_depth_abs_rel"].median()
    positions = np.arange(len(trip_ids))
    width = 0.36
    axes[0, 1].bar(
        positions - width / 2,
        [raw_by_trip[trip_id] for trip_id in trip_ids],
        width,
        label="raw SGBM",
    )
    axes[0, 1].bar(
        positions + width / 2,
        [filtered_by_trip[trip_id] for trip_id in trip_ids],
        width,
        label="LR-consistent",
    )
    axes[0, 1].set_xticks(
        positions, [trip_id.replace("-Sample", "") for trip_id in trip_ids]
    )
    axes[0, 1].set_ylabel("Median pixel abs-relative depth error")
    axes[0, 1].set_title("Depth validation against provided keyframes")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    for color, trip_id in zip(colors, trip_ids):
        subset = frame[frame.trip_id == trip_id]
        axes[1, 0].scatter(
            subset.raw_depth_abs_rel,
            subset.consistent_depth_abs_rel,
            s=45,
            alpha=0.8,
            color=color,
            label=trip_id.replace("-Sample", ""),
        )
    upper = float(
        np.nanmax(
            frame[["raw_depth_abs_rel", "consistent_depth_abs_rel"]].to_numpy()
        )
    )
    axes[1, 0].plot([0, upper], [0, upper], "--", color="#333333")
    axes[1, 0].set_xlabel("Raw depth abs-relative error")
    axes[1, 0].set_ylabel("LR-consistent depth abs-relative error")
    improved = int(
        np.count_nonzero(
            frame.consistent_depth_abs_rel < frame.raw_depth_abs_rel
        )
    )
    axes[1, 0].set_title(f"LR filtering improves {improved}/{len(frame)} sampled frames")
    axes[1, 0].legend(ncols=2, fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    shifts = (
        frame.roi_consistent_depth_median_m - frame.roi_raw_depth_median_m
    )
    axes[1, 1].hist(shifts, bins=30, color="#5b8ff9", edgecolor="white")
    axes[1, 1].axvline(0, color="#222222", linestyle="--")
    axes[1, 1].set_xlabel("Filtered ROI median − raw ROI median (m)")
    axes[1, 1].set_ylabel("Sampled frames")
    axes[1, 1].set_title(
        f"ROI median changes >0.5 m in {int(np.count_nonzero(np.abs(shifts) > 0.5))}"
        f"/{len(frame)} frames"
    )
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Stage 2A Component 1 — SGBM left-right consistency audit "
        f"({len(frame)} frames)",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def write_metrics(path: Path, rows: Iterable[FrameMetrics]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai_cv/outputs/reports/phase02a/stereo_confidence"),
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=50,
        help="Broad audit stride across all practice trips; 0 disables.",
    )
    args = parser.parse_args()

    if args.sample_stride < 0:
        parser.error("--sample-stride must be non-negative")
    left_matcher = create_left_matcher()
    right_matcher = create_right_matcher()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_metrics = []
    visual_records = []
    for case in DEFAULT_CASES:
        trip_dir = args.practice_root / case.trip_id
        focal_length, baseline = load_calibration(trip_dir)
        left, right = read_stereo(trip_dir, case.frame_id)
        left_disparity, right_disparity, left_match_ms, right_match_ms = (
            compute_disparities_with_timing(
            left, right, left_matcher, right_matcher
            )
        )
        metrics, arrays = compute_frame_metrics(
            case.trip_id,
            case.frame_id,
            left_disparity,
            right_disparity,
            focal_length,
            baseline,
            load_reference_depth(trip_dir, case.frame_id),
            left_match_ms,
            right_match_ms,
        )
        case_metrics.append(metrics)
        visual_records.append((case, left, arrays, metrics))
        save_case_visual(
            args.output_dir / "cases" / f"{case.trip_id}_{case.frame_id:06d}.png",
            case,
            left,
            left_disparity,
            arrays,
            metrics,
        )

    write_metrics(args.output_dir / "failure_case_metrics.csv", case_metrics)
    save_summary_visual(args.output_dir / "failure_case_summary.png", visual_records)

    sampled_metrics = []
    if args.sample_stride:
        for trip_dir in sorted(args.practice_root.glob("T*-Sample")):
            focal_length, baseline = load_calibration(trip_dir)
            frame_paths = sorted((trip_dir / "kitti" / "image_2").glob("*.jpg"))
            for frame_path in frame_paths[:: args.sample_stride]:
                frame_id = int(frame_path.stem)
                left, right = read_stereo(trip_dir, frame_id)
                (
                    left_disparity,
                    right_disparity,
                    left_match_ms,
                    right_match_ms,
                ) = compute_disparities_with_timing(
                    left, right, left_matcher, right_matcher
                )
                metrics, _ = compute_frame_metrics(
                    trip_dir.name,
                    frame_id,
                    left_disparity,
                    right_disparity,
                    focal_length,
                    baseline,
                    load_reference_depth(trip_dir, frame_id),
                    left_match_ms,
                    right_match_ms,
                )
                sampled_metrics.append(metrics)
        write_metrics(args.output_dir / "sampled_frame_metrics.csv", sampled_metrics)
        save_sampled_summary(
            args.output_dir / "sampled_frame_summary.png", sampled_metrics
        )

    print(
        f"Wrote {len(case_metrics)} failure-case analyses and "
        f"{len(sampled_metrics)} sampled-frame rows to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
