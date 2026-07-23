"""Visual falsification of ground removal and classical obstacle components."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_stereo_confidence import (
    DEFAULT_CASES,
    compute_disparities,
    create_left_matcher,
    create_right_matcher,
    left_right_consistency,
    load_calibration,
    read_stereo,
)
from classical_geometry import (
    GroundModel,
    collision_corridor_mask,
    estimate_ground_model,
    extract_obstacle_components,
    ground_and_obstacle_masks,
)


@dataclass(frozen=True)
class CaseMetrics:
    trip_id: str
    frame_id: int
    outcome: str
    ground_model_found: bool
    ground_slope_px_per_row: float
    ground_intercept_px: float
    ground_confidence: float
    ground_median_residual_px: float
    ground_fraction: float
    obstacle_evidence_fraction: float
    component_count: int
    nearest_component_depth_m: float
    nearest_component_quality: float


def colored_component_overlay(
    left_rgb: np.ndarray,
    components,
    labels: np.ndarray,
    corridor: np.ndarray,
) -> np.ndarray:
    overlay = left_rgb.copy()
    tint = np.zeros_like(left_rgb)
    tint[corridor] = (40, 80, 180)
    overlay = cv2.addWeighted(overlay, 0.82, tint, 0.18, 0.0)
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(components))))[:, :3]
    for color, component in zip(palette, components):
        component_mask = labels == component.component_id
        rgb_color = (color * 255).astype(np.uint8)
        overlay[component_mask] = (
            0.45 * overlay[component_mask] + 0.55 * rgb_color
        ).astype(np.uint8)
        x0, y0, x1, y1 = component.bbox
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            tuple(int(value) for value in rgb_color),
            2,
        )
        cv2.putText(
            overlay,
            f"{component.depth_m:.1f}m q{component.quality:.2f}",
            (x0, max(14, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            tuple(int(value) for value in rgb_color),
            1,
            cv2.LINE_AA,
        )
    return overlay


def save_case_visual(
    path: Path,
    case,
    left_bgr: np.ndarray,
    disparity: np.ndarray,
    histogram: np.ndarray,
    model: GroundModel,
    ground: np.ndarray,
    obstacle_evidence: np.ndarray,
    components,
    labels: np.ndarray,
    corridor: np.ndarray,
    metrics: CaseMetrics,
) -> None:
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    ground_overlay = left_rgb.copy()
    ground_tint = np.zeros_like(left_rgb)
    ground_tint[ground] = (30, 150, 255)
    ground_overlay = cv2.addWeighted(ground_overlay, 0.62, ground_tint, 0.38, 0)

    obstacle_overlay = left_rgb.copy()
    obstacle_tint = np.zeros_like(left_rgb)
    obstacle_tint[obstacle_evidence & corridor] = (255, 60, 40)
    obstacle_overlay = cv2.addWeighted(
        obstacle_overlay, 0.58, obstacle_tint, 0.42, 0
    )
    component_overlay = colored_component_overlay(
        left_rgb, components, labels, corridor
    )

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title("Input frame")

    disparity_view = np.ma.masked_where(disparity <= 0.5, disparity)
    image = axes[0, 1].imshow(disparity_view, cmap="turbo", vmin=0, vmax=64)
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046, label="disparity (px)")
    axes[0, 1].set_title("Raw SGBM disparity")

    axes[0, 2].imshow(
        np.log1p(histogram),
        cmap="magma",
        aspect="auto",
        extent=(0, 96, histogram.shape[0], 0),
    )
    rows = np.arange(histogram.shape[0])
    axes[0, 2].plot(
        model.disparity_at(rows),
        rows,
        color="cyan",
        linewidth=2,
        label="fitted ground line",
    )
    axes[0, 2].set_xlim(0, 64)
    axes[0, 2].set_xlabel("disparity (px)")
    axes[0, 2].set_ylabel("image row")
    axes[0, 2].set_title("V-disparity + ground fit")
    axes[0, 2].legend()

    axes[1, 0].imshow(ground_overlay)
    axes[1, 0].set_title("Estimated ground support (blue)")
    axes[1, 1].imshow(obstacle_overlay)
    axes[1, 1].set_title("Closer-than-ground evidence (red)")
    axes[1, 2].imshow(component_overlay)
    axes[1, 2].set_title(
        f"Corridor components: {len(components)} "
        "(box label = depth, quality)"
    )

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]):
        axis.axis("off")
    figure.suptitle(
        f"{case.trip_id} frame {case.frame_id} [{case.outcome}] — "
        f"ground confidence={metrics.ground_confidence:.2f}, "
        f"components={metrics.component_count}, "
        f"nearest={metrics.nearest_component_depth_m:.2f}m",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=155)
    plt.close(figure)


def save_summary(path: Path, records) -> None:
    figure, axes = plt.subplots(
        len(records), 2, figsize=(14, 3.3 * len(records)), constrained_layout=True
    )
    for row, (case, left_bgr, overlay, metrics) in enumerate(records):
        axes[row, 0].imshow(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(
            f"{case.trip_id} #{case.frame_id} [{case.outcome}]"
        )
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"ground q={metrics.ground_confidence:.2f}; "
            f"components={metrics.component_count}; "
            f"nearest={metrics.nearest_component_depth_m:.1f}m"
        )
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    figure.suptitle(
        "Stage 2A vertical slice — ground removal and obstacle components",
        fontsize=16,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=155)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai_cv/outputs/reports/phase02a/ground_obstacles"),
    )
    args = parser.parse_args()

    left_matcher = create_left_matcher()
    right_matcher = create_right_matcher()
    metrics_rows = []
    component_rows = []
    summary_records = []
    for case in DEFAULT_CASES:
        trip_dir = args.practice_root / case.trip_id
        focal_length, baseline_m = load_calibration(trip_dir)
        left, right = read_stereo(trip_dir, case.frame_id)
        left_disparity, right_disparity = compute_disparities(
            left, right, left_matcher, right_matcher
        )
        _, consistent, _ = left_right_consistency(
            left_disparity, right_disparity
        )
        model, histogram = estimate_ground_model(left_disparity)
        if model is None:
            print(f"WARN: no ground model for {case.trip_id} #{case.frame_id}")
            continue
        ground, obstacle_evidence, _ = ground_and_obstacle_masks(
            left_disparity, model
        )
        components, labels, corridor = extract_obstacle_components(
            left_disparity,
            obstacle_evidence,
            consistent,
            focal_length,
            baseline_m,
        )
        component_rows.extend(
            {
                "trip_id": case.trip_id,
                "frame_id": case.frame_id,
                "outcome": case.outcome,
                **asdict(component),
            }
            for component in components
        )
        nearest = components[0] if components else None
        metrics = CaseMetrics(
            trip_id=case.trip_id,
            frame_id=case.frame_id,
            outcome=case.outcome,
            ground_model_found=True,
            ground_slope_px_per_row=model.disparity_per_row,
            ground_intercept_px=model.intercept,
            ground_confidence=model.confidence,
            ground_median_residual_px=model.median_residual_px,
            ground_fraction=float(np.mean(ground)),
            obstacle_evidence_fraction=float(np.mean(obstacle_evidence)),
            component_count=len(components),
            nearest_component_depth_m=(
                nearest.depth_m if nearest is not None else math.nan
            ),
            nearest_component_quality=(
                nearest.quality if nearest is not None else math.nan
            ),
        )
        metrics_rows.append(metrics)
        component_overlay = colored_component_overlay(
            cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
            components,
            labels,
            corridor,
        )
        summary_records.append((case, left, component_overlay, metrics))
        save_case_visual(
            args.output_dir / "cases" / f"{case.trip_id}_{case.frame_id:06d}.png",
            case,
            left,
            left_disparity,
            histogram,
            model,
            ground,
            obstacle_evidence,
            components,
            labels,
            corridor,
            metrics,
        )

    if not metrics_rows:
        raise RuntimeError("No failure case produced a ground model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "failure_case_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(metrics_rows[0]).keys())
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(asdict(row))
    if component_rows:
        with (args.output_dir / "component_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(component_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(component_rows)
    save_summary(args.output_dir / "failure_case_summary.png", summary_records)
    print(f"Wrote {len(metrics_rows)} case analyses to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
