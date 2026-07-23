"""Classical ground/obstacle geometry for the Stage 2A vertical slice."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GroundModel:
    disparity_per_row: float
    intercept: float
    confidence: float
    median_residual_px: float
    supporting_rows: int
    total_rows: int

    def disparity_at(self, rows: np.ndarray | float) -> np.ndarray:
        return self.disparity_per_row * np.asarray(rows) + self.intercept


@dataclass(frozen=True)
class ObstacleComponent:
    component_id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    center_x: float
    center_y: float
    bottom_y: int
    depth_m: float
    depth_p20_m: float
    depth_p35_m: float
    depth_mad_m: float
    lr_support: float
    corridor_overlap: float
    quality: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


def v_disparity_histogram(
    disparity: np.ndarray,
    *,
    max_disparity: int = 96,
    bin_size: float = 0.5,
    x_margin_fraction: float = 0.10,
) -> np.ndarray:
    height, width = disparity.shape
    bins = int(max_disparity / bin_size)
    histogram = np.zeros((height, bins), dtype=np.float32)
    x0 = int(width * x_margin_fraction)
    x1 = int(width * (1.0 - x_margin_fraction))
    for row in range(height):
        values = disparity[row, x0:x1]
        values = values[np.isfinite(values) & (values > 0.5) & (values < max_disparity)]
        if values.size:
            indices = np.clip((values / bin_size).astype(np.int32), 0, bins - 1)
            histogram[row] = np.bincount(indices, minlength=bins)
    return histogram


def row_disparity_modes(
    histogram: np.ndarray,
    *,
    bin_size: float = 0.5,
    y_start_fraction: float = 0.48,
    minimum_peak_count: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = histogram.shape[0]
    rows = np.arange(height)
    mode_indices = np.argmax(histogram, axis=1)
    peak_counts = histogram[rows, mode_indices]
    selected = (
        (rows >= int(height * y_start_fraction))
        & (peak_counts >= minimum_peak_count)
        & (mode_indices > 0)
    )
    return (
        rows[selected].astype(np.float32),
        ((mode_indices[selected] + 0.5) * bin_size).astype(np.float32),
        peak_counts[selected].astype(np.float32),
    )


def fit_ground_line(
    rows: np.ndarray,
    disparities: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    residual_threshold_px: float = 1.5,
    minimum_slope: float = 0.025,
    maximum_slope: float = 0.40,
    minimum_supporting_rows: int = 30,
) -> GroundModel | None:
    """Deterministic RANSAC-like fit of disparity = slope * row + intercept."""
    if rows.size < minimum_supporting_rows:
        return None
    if weights is None:
        weights = np.ones_like(rows, dtype=np.float32)
    weights = np.maximum(weights.astype(np.float64), 1.0)
    rows64 = rows.astype(np.float64)
    disparities64 = disparities.astype(np.float64)

    candidate_indices = np.linspace(
        0, rows.size - 1, min(rows.size, 48), dtype=np.int32
    )
    best_score = -math.inf
    best_inliers: np.ndarray | None = None
    for left_index_position, first in enumerate(candidate_indices[:-1]):
        for second in candidate_indices[left_index_position + 1 :]:
            delta_row = rows64[second] - rows64[first]
            if delta_row < 24:
                continue
            slope = (disparities64[second] - disparities64[first]) / delta_row
            if not minimum_slope <= slope <= maximum_slope:
                continue
            intercept = disparities64[first] - slope * rows64[first]
            residual = np.abs(disparities64 - (slope * rows64 + intercept))
            inliers = residual <= residual_threshold_px
            if np.count_nonzero(inliers) < minimum_supporting_rows:
                continue
            score = float(np.sum(weights[inliers])) - 0.25 * float(
                np.sum(residual[inliers] * weights[inliers])
            )
            if score > best_score:
                best_score = score
                best_inliers = inliers

    if best_inliers is None:
        return None

    inliers = best_inliers
    for _ in range(3):
        design = np.column_stack([rows64[inliers], np.ones(np.count_nonzero(inliers))])
        weighted_design = design * np.sqrt(weights[inliers])[:, None]
        weighted_target = disparities64[inliers] * np.sqrt(weights[inliers])
        slope, intercept = np.linalg.lstsq(
            weighted_design, weighted_target, rcond=None
        )[0]
        if not minimum_slope <= slope <= maximum_slope:
            return None
        residual = np.abs(disparities64 - (slope * rows64 + intercept))
        inliers = residual <= residual_threshold_px
        if np.count_nonzero(inliers) < minimum_supporting_rows:
            return None

    inlier_residual = residual[inliers]
    support = int(np.count_nonzero(inliers))
    confidence = float(
        (support / rows.size)
        * min(1.0, support / max(1.0, minimum_supporting_rows * 2.0))
        * math.exp(-float(np.median(inlier_residual)) / residual_threshold_px)
    )
    return GroundModel(
        disparity_per_row=float(slope),
        intercept=float(intercept),
        confidence=confidence,
        median_residual_px=float(np.median(inlier_residual)),
        supporting_rows=support,
        total_rows=int(rows.size),
    )


def estimate_ground_model(
    disparity: np.ndarray,
) -> tuple[GroundModel | None, np.ndarray]:
    histogram = v_disparity_histogram(disparity)
    rows, modes, weights = row_disparity_modes(histogram)
    model = fit_ground_line(rows, modes, weights)
    return model, histogram


def collision_corridor_mask(
    shape: tuple[int, int],
    *,
    top_y_fraction: float = 0.36,
    top_width_fraction: float = 0.24,
    bottom_width_fraction: float = 0.82,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    top_y = int(height * top_y_fraction)
    center_x = width // 2
    points = np.array(
        [
            [center_x - int(width * top_width_fraction / 2), top_y],
            [center_x + int(width * top_width_fraction / 2), top_y],
            [center_x + int(width * bottom_width_fraction / 2), height - 1],
            [center_x - int(width * bottom_width_fraction / 2), height - 1],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, points, 1)
    return mask.astype(bool)


def ground_and_obstacle_masks(
    disparity: np.ndarray,
    ground_model: GroundModel,
    *,
    ground_tolerance_px: float = 1.5,
    obstacle_margin_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, _ = disparity.shape
    rows = np.arange(height, dtype=np.float32)[:, None]
    predicted_ground = ground_model.disparity_at(rows)
    valid = np.isfinite(disparity) & (disparity > 0.5)
    analysis_region = rows >= height * 0.34

    ground = (
        valid
        & analysis_region
        & (np.abs(disparity - predicted_ground) <= ground_tolerance_px)
    )
    obstacle_evidence = (
        valid
        & analysis_region
        & (disparity >= predicted_ground + obstacle_margin_px)
    )
    return ground, obstacle_evidence, predicted_ground


def extract_obstacle_components(
    disparity: np.ndarray,
    obstacle_evidence: np.ndarray,
    lr_consistent: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    *,
    minimum_area: int = 90,
    minimum_height: int = 12,
) -> tuple[list[ObstacleComponent], np.ndarray, np.ndarray]:
    corridor = collision_corridor_mask(disparity.shape)
    candidate = obstacle_evidence & corridor
    binary = candidate.astype(np.uint8) * 255
    # A road-disparity error usually forms a thin horizontal band. Require
    # vertical support first (Stixel-lite) so those bands cannot become the
    # nearest "obstacle" merely because they span many columns.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
        iterations=1,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        iterations=1,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components: list[ObstacleComponent] = []
    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area < minimum_area or height < minimum_height:
            continue
        if width / max(1, height) > 6.0:
            continue
        if width > disparity.shape[1] * 0.75 or height > disparity.shape[0] * 0.75:
            continue

        component_region = labels == label
        evidence_region = component_region & obstacle_evidence
        disparities = disparity[evidence_region]
        disparities = disparities[
            np.isfinite(disparities) & (disparities > 0.5)
        ]
        if disparities.size < minimum_area // 2:
            continue
        depths = focal_length_px * baseline_m / disparities
        depths = depths[(depths >= 1.5) & (depths <= 80.0)]
        if depths.size < minimum_area // 2:
            continue

        depth_m = float(np.median(depths))
        depth_p20 = float(np.percentile(depths, 20.0))
        depth_p35 = float(np.percentile(depths, 35.0))
        depth_mad = float(np.median(np.abs(depths - depth_m)))
        support_pixels = int(np.count_nonzero(evidence_region))
        lr_support = float(
            np.count_nonzero(evidence_region & lr_consistent)
            / max(1, support_pixels)
        )
        corridor_overlap = float(
            np.count_nonzero(component_region & corridor)
            / max(1, np.count_nonzero(component_region))
        )
        density = min(1.0, support_pixels / max(1.0, area))
        dispersion_score = math.exp(-depth_mad / max(1.0, depth_m * 0.15))
        quality = float(
            0.40 * density
            + 0.25 * lr_support
            + 0.20 * corridor_overlap
            + 0.15 * dispersion_score
        )
        components.append(
            ObstacleComponent(
                component_id=label,
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                center_x=float(centroids[label][0]),
                center_y=float(centroids[label][1]),
                bottom_y=y + height,
                depth_m=depth_m,
                depth_p20_m=depth_p20,
                depth_p35_m=depth_p35,
                depth_mad_m=depth_mad,
                lr_support=lr_support,
                corridor_overlap=corridor_overlap,
                quality=quality,
            )
        )

    components.sort(key=lambda component: (component.depth_m, -component.area))
    return components, labels, corridor
