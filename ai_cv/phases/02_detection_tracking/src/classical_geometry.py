"""Classical ground/obstacle geometry for the Stage 2A vertical slice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


_VERTICAL_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
_HORIZONTAL_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
_CLEANUP_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


@lru_cache(maxsize=8)
def _row_coordinates(height: int) -> np.ndarray:
    rows = np.arange(height, dtype=np.float32)[:, None]
    rows.flags.writeable = False
    return rows


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
    object_depth_m: float = math.nan
    object_depth_mad_m: float = math.nan
    object_depth_confidence: float = 0.0
    object_depth_mode_count: int = 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass(frozen=True)
class ObjectDepthEstimate:
    """Robust depth estimate from the inner support of one obstacle ROI."""

    depth_m: float
    depth_mad_m: float
    confidence: float
    mode_count: int


def estimate_object_depth(
    disparity_roi: np.ndarray,
    evidence_roi: np.ndarray,
    lr_consistent_roi: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    *,
    inner_width_fraction: float = 0.64,
    inner_height_fraction: float = 0.76,
    disparity_bin_px: float = 0.5,
) -> ObjectDepthEstimate | None:
    """Estimate foreground depth from a component's inner, disparity-modal ROI.

    The inner crop reduces boundary/background contamination. Up to two strong
    disparity modes are retained; the nearer significant mode is selected so a
    broad component containing road/background support does not pull depth away
    from the foreground obstacle.
    """
    if disparity_roi.shape != evidence_roi.shape:
        raise ValueError("disparity and evidence ROI shapes must match")
    if disparity_roi.shape != lr_consistent_roi.shape:
        raise ValueError("LR-consistency ROI shape must match disparity")
    height, width = disparity_roi.shape
    if height < 3 or width < 3:
        return None

    keep_width = int(round(width * inner_width_fraction))
    keep_height = int(round(height * inner_height_fraction))
    keep_width = min(width, max(3, keep_width))
    keep_height = min(height, max(3, keep_height))
    x0 = (width - keep_width) // 2
    x1 = x0 + keep_width
    y0 = max(0, int(round(height * 0.10)))
    y1 = min(height, y0 + keep_height)

    core_evidence = evidence_roi[y0:y1, x0:x1]
    core_disparity = disparity_roi[y0:y1, x0:x1]
    core_lr = lr_consistent_roi[y0:y1, x0:x1]
    valid = (
        core_evidence
        & np.isfinite(core_disparity)
        & (core_disparity > 0.5)
        & (core_disparity < 96.0)
    )
    minimum_support = max(18, int(0.03 * keep_width * keep_height))
    if np.count_nonzero(valid) < minimum_support:
        return None

    values = core_disparity[valid].astype(np.float64)
    bin_count = int(math.ceil(96.0 / disparity_bin_px))
    histogram, _ = np.histogram(
        values,
        bins=bin_count,
        range=(0.0, 96.0),
    )
    smoothed = np.convolve(
        histogram.astype(np.float64),
        np.asarray([0.25, 0.50, 0.25]),
        mode="same",
    )
    local_peaks = np.flatnonzero(
        (smoothed >= np.roll(smoothed, 1))
        & (smoothed >= np.roll(smoothed, -1))
    )
    local_peaks = local_peaks[(local_peaks > 0) & (local_peaks < bin_count - 1)]
    if local_peaks.size == 0:
        return None

    peak_floor = max(4.0, 0.18 * float(np.max(smoothed)))
    ranked = sorted(
        (
            (float(smoothed[index]), int(index))
            for index in local_peaks
            if smoothed[index] >= peak_floor
        ),
        reverse=True,
    )
    selected_peaks: list[int] = []
    minimum_separation_bins = max(2, int(round(2.0 / disparity_bin_px)))
    for _, peak in ranked:
        if all(
            abs(peak - existing) >= minimum_separation_bins
            for existing in selected_peaks
        ):
            selected_peaks.append(peak)
        if len(selected_peaks) == 2:
            break
    if not selected_peaks:
        return None

    # Higher disparity is the nearer surface. Require local support around the
    # selected mode so a single noisy maximum cannot become object depth.
    selected_peak = max(selected_peaks)
    center_disparity = (selected_peak + 0.5) * disparity_bin_px
    mode_mask = np.abs(values - center_disparity) <= 1.25
    mode_values = values[mode_mask]
    if mode_values.size < minimum_support // 2:
        return None

    disparities = mode_values
    depths = focal_length_px * baseline_m / disparities
    depths = depths[(depths >= 1.5) & (depths <= 80.0)]
    if depths.size < minimum_support // 2:
        return None
    depth_m = float(np.median(depths))
    depth_mad = float(np.median(np.abs(depths - depth_m)))
    mode_support = float(mode_values.size / values.size)
    lr_support = float(np.count_nonzero(core_lr[valid]) / values.size)
    dispersion = math.exp(-depth_mad / max(0.25, depth_m * 0.08))
    confidence = float(
        np.clip(
            0.45 * mode_support + 0.30 * lr_support + 0.25 * dispersion,
            0.0,
            1.0,
        )
    )
    return ObjectDepthEstimate(
        depth_m=depth_m,
        depth_mad_m=depth_mad,
        confidence=confidence,
        mode_count=len(selected_peaks),
    )


def v_disparity_histogram(
    disparity: np.ndarray,
    *,
    max_disparity: int = 96,
    bin_size: float = 0.5,
    x_margin_fraction: float = 0.10,
) -> np.ndarray:
    height, width = disparity.shape
    bins = int(max_disparity / bin_size)
    x0 = int(width * x_margin_fraction)
    x1 = int(width * (1.0 - x_margin_fraction))
    values = disparity[:, x0:x1]
    valid = np.isfinite(values) & (values > 0.5) & (values < max_disparity)
    valid_rows = np.nonzero(valid)[0]
    if not valid_rows.size:
        return np.zeros((height, bins), dtype=np.float32)

    bin_indices = np.clip(
        (values[valid] / bin_size).astype(np.int32),
        0,
        bins - 1,
    )
    flat_indices = valid_rows * bins + bin_indices
    return np.bincount(
        flat_indices,
        minlength=height * bins,
    ).reshape(height, bins).astype(np.float32)


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
    """Return a caller-owned collision-corridor mask."""
    return _cached_collision_corridor_mask(
        shape,
        top_y_fraction,
        top_width_fraction,
        bottom_width_fraction,
    ).copy()


@lru_cache(maxsize=16)
def _cached_collision_corridor_mask(
    shape: tuple[int, int],
    top_y_fraction: float,
    top_width_fraction: float,
    bottom_width_fraction: float,
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
    corridor = mask.astype(bool)
    corridor.flags.writeable = False
    return corridor


def ground_and_obstacle_masks(
    disparity: np.ndarray,
    ground_model: GroundModel,
    *,
    ground_tolerance_px: float = 1.5,
    obstacle_margin_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, _ = disparity.shape
    rows = _row_coordinates(height)
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
    compute_object_depth: bool = False,
) -> tuple[list[ObstacleComponent], np.ndarray, np.ndarray]:
    corridor = _cached_collision_corridor_mask(
        disparity.shape,
        0.36,
        0.24,
        0.82,
    )
    binary = np.logical_and(obstacle_evidence, corridor).astype(np.uint8)
    # A road-disparity error usually forms a thin horizontal band. Require
    # vertical support first (Stixel-lite) so those bands cannot become the
    # nearest "obstacle" merely because they span many columns.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        _VERTICAL_OPEN_KERNEL,
        iterations=1,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        _HORIZONTAL_CLOSE_KERNEL,
        iterations=1,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        _CLEANUP_OPEN_KERNEL,
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

        y1 = y + height
        x1 = x + width
        component_region = labels[y:y1, x:x1] == label
        evidence_region = (
            component_region & obstacle_evidence[y:y1, x:x1]
        )
        disparities = disparity[y:y1, x:x1][evidence_region]
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
            np.count_nonzero(
                evidence_region & lr_consistent[y:y1, x:x1]
            )
            / max(1, support_pixels)
        )
        corridor_overlap = float(
            np.count_nonzero(
                component_region & corridor[y:y1, x:x1]
            )
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
        object_depth = (
            estimate_object_depth(
                disparity[y:y1, x:x1],
                evidence_region,
                lr_consistent[y:y1, x:x1],
                focal_length_px,
                baseline_m,
            )
            if compute_object_depth
            else None
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
                object_depth_m=(
                    object_depth.depth_m
                    if object_depth is not None
                    else depth_p35
                ),
                object_depth_mad_m=(
                    object_depth.depth_mad_m
                    if object_depth is not None
                    else depth_mad
                ),
                object_depth_confidence=(
                    object_depth.confidence
                    if object_depth is not None
                    else 0.0
                ),
                object_depth_mode_count=(
                    object_depth.mode_count
                    if object_depth is not None
                    else 0
                ),
            )
        )

    components.sort(key=lambda component: (component.depth_m, -component.area))
    return components, labels, corridor
