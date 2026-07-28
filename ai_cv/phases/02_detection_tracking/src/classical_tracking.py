"""Causal component association and robust per-track TTC."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence

import numpy as np

from classical_geometry import ObstacleComponent, collision_corridor_mask

# Optional semantic fusion — imported lazily so the none-backend path has
# zero dependency on YOLO / Ultralytics at import time.
try:
    from detector_interfaces import Detection
    from semantic_fusion import (
        TemporalSemanticState,
        associate_component_with_detections,
    )
    _SEMANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover — only absent in stripped environments
    _SEMANTIC_AVAILABLE = False


@dataclass(frozen=True)
class TrackObservation:
    timestamp: float
    depth_m: float
    center_x: float
    center_y: float
    quality: float
    depth_mad_m: float = math.nan
    lr_support: float = math.nan
    corridor_overlap: float = math.nan
    depth_sigma_m: float = math.nan
    depth_confidence: float = 0.0


@dataclass
class DistanceVelocityFilter:
    """Small constant-velocity filter for causal object distance and speed."""

    state: np.ndarray | None = None
    covariance: np.ndarray | None = None
    timestamp: float | None = None
    innovation_ema_m: float = 0.0
    updates: int = 0

    def update(
        self,
        timestamp: float,
        distance_m: float,
        sigma_m: float,
    ) -> None:
        sigma_m = float(np.clip(sigma_m, 0.15, 12.0))
        if (
            self.state is None
            or self.covariance is None
            or self.timestamp is None
            or timestamp <= self.timestamp
            or timestamp - self.timestamp > 0.5
        ):
            self.state = np.asarray([distance_m, 0.0], dtype=np.float64)
            self.covariance = np.diag(
                [sigma_m * sigma_m, 16.0]
            ).astype(np.float64)
            self.timestamp = timestamp
            self.innovation_ema_m = 0.0
            self.updates = 1
            return

        dt = float(timestamp - self.timestamp)
        transition = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        process_acceleration = 5.0
        process_noise = process_acceleration**2 * np.asarray(
            [
                [0.25 * dt**4, 0.5 * dt**3],
                [0.5 * dt**3, dt**2],
            ],
            dtype=np.float64,
        )
        predicted_state = transition @ self.state
        predicted_covariance = (
            transition @ self.covariance @ transition.T + process_noise
        )
        measurement_variance = sigma_m * sigma_m
        innovation = float(distance_m - predicted_state[0])
        innovation_variance = float(
            predicted_covariance[0, 0] + measurement_variance
        )
        gain = predicted_covariance[:, 0] / innovation_variance
        self.state = predicted_state + gain * innovation
        self.covariance = predicted_covariance - np.outer(
            gain,
            predicted_covariance[0, :],
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.timestamp = timestamp
        self.innovation_ema_m = (
            0.75 * self.innovation_ema_m + 0.25 * abs(innovation)
            if self.updates > 1
            else abs(innovation)
        )
        self.updates += 1

    def motion_state(
        self,
        timestamp: float,
        *,
        missed_frames: int = 0,
    ) -> tuple[float, float, float]:
        if (
            self.state is None
            or self.covariance is None
            or self.timestamp is None
            or self.updates < 3
        ):
            return 0.0, math.inf, math.inf
        dt = max(0.0, float(timestamp - self.timestamp))
        distance = float(self.state[0] + dt * self.state[1])
        closing_speed = float(-self.state[1])
        residual = float(
            self.innovation_ema_m
            + 0.20 * missed_frames
            + math.sqrt(max(0.0, self.covariance[0, 0])) * 0.10
        )
        if distance <= 0.0 or closing_speed <= 0.3:
            return closing_speed, math.inf, residual
        return closing_speed, distance / closing_speed, residual


@dataclass
class ComponentTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    observations: Deque[TrackObservation] = field(
        default_factory=lambda: deque(maxlen=11)
    )
    hits: int = 0
    age: int = 0
    missed: int = 0
    current_timestamp: float | None = None
    filtered_motion: DistanceVelocityFilter | None = field(
        default=None,
        repr=False,
    )
    # Semantic state — None when semantic detector is 'none' (bit-identical path).
    semantic_state: "TemporalSemanticState | None" = field(default=None, repr=False)

    def enable_filtered_motion(self) -> None:
        if self.filtered_motion is None:
            self.filtered_motion = DistanceVelocityFilter()

    def enable_semantic(self) -> None:
        """Activate per-track semantic state (call once after construction when detector != none)."""
        if _SEMANTIC_AVAILABLE and self.semantic_state is None:
            self.semantic_state = TemporalSemanticState()

    def update_semantic(
        self,
        detections: "Sequence[Detection]",
        image_shape: tuple[int, int],
    ) -> None:
        """Associate current bbox with YOLO detections and update EMA score.

        No-op when semantic state is disabled (none backend).
        """
        if self.semantic_state is None or not _SEMANTIC_AVAILABLE:
            return
        assoc = associate_component_with_detections(self.bbox, detections, image_shape)
        self.semantic_state.update(assoc)

    def is_semantically_suppressed(
        self,
        *,
        score_threshold: float = 0.25,
        max_misses: int = 3,
        fallback_depth_m: float = 5.0,
    ) -> bool:
        """Return True only when soft-guard conditions all hold.

        Always False when semantic state is disabled (none backend bit-parity).
        """
        if self.semantic_state is None:
            return False
        return self.semantic_state.is_suppressed(
            latest_depth_m=self.latest.depth_m,
            score_threshold=score_threshold,
            max_misses=max_misses,
            fallback_depth_m=fallback_depth_m,
        )

    def update(
        self,
        component: ObstacleComponent,
        timestamp: float,
        depth_m: float,
    ) -> None:
        self.bbox = component.bbox
        self.current_timestamp = timestamp
        object_depth_mad = float(
            getattr(component, "object_depth_mad_m", component.depth_mad_m)
        )
        object_depth_confidence = float(
            getattr(component, "object_depth_confidence", 0.0)
        )
        depth_mad = (
            object_depth_mad
            if math.isfinite(object_depth_mad)
            else component.depth_mad_m
        )
        confidence_scale = max(0.15, object_depth_confidence)
        depth_sigma = float(
            np.clip(
                max(0.15, depth_mad)
                * (1.35 - 0.55 * component.lr_support)
                / math.sqrt(confidence_scale),
                0.15,
                12.0,
            )
        )
        self.observations.append(
            TrackObservation(
                timestamp=timestamp,
                depth_m=depth_m,
                center_x=component.center_x,
                center_y=component.center_y,
                quality=component.quality,
                depth_mad_m=depth_mad,
                lr_support=component.lr_support,
                corridor_overlap=component.corridor_overlap,
                depth_sigma_m=depth_sigma,
                depth_confidence=object_depth_confidence,
            )
        )
        if self.filtered_motion is not None:
            self.filtered_motion.update(timestamp, depth_m, depth_sigma)
        self.hits += 1
        self.age += 1
        self.missed = 0

    @property
    def latest(self) -> TrackObservation:
        return self.observations[-1]

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3 and len(self.observations) >= 3

    def motion_state(self) -> tuple[float, float, float]:
        """Return closing speed, TTC and robust fit residual."""
        if len(self.observations) < 3:
            return 0.0, math.inf, math.inf
        times = np.asarray(
            [observation.timestamp for observation in self.observations],
            dtype=float,
        )
        depths = np.asarray(
            [observation.depth_m for observation in self.observations],
            dtype=float,
        )
        slopes = [
            (depths[j] - depths[i]) / (times[j] - times[i])
            for i in range(len(times) - 1)
            for j in range(i + 1, len(times))
            if times[j] > times[i]
        ]
        if not slopes:
            return 0.0, math.inf, math.inf
        slope = float(np.median(slopes))
        intercept = float(np.median(depths - slope * times))
        residual = float(np.median(np.abs(depths - (slope * times + intercept))))
        closing_speed = -slope
        if closing_speed <= 0.3:
            return closing_speed, math.inf, residual
        return closing_speed, float(depths[-1] / closing_speed), residual

    def filtered_motion_state(self) -> tuple[float, float, float]:
        if self.filtered_motion is None or not self.observations:
            return self.motion_state()
        return self.filtered_motion.motion_state(
            (
                self.current_timestamp
                if self.current_timestamp is not None
                else self.latest.timestamp
            ),
            missed_frames=self.missed,
        )

    def confidence(
        self,
        ground_confidence: float,
        *,
        use_filtered_motion: bool = False,
    ) -> float:
        if not self.observations:
            return 0.0
        closing_speed, _, residual = (
            self.filtered_motion_state()
            if use_filtered_motion
            else self.motion_state()
        )
        history_score = min(1.0, len(self.observations) / 7.0)
        observation_score = float(
            np.mean([observation.quality for observation in self.observations])
        )
        residual_scale = max(0.5, self.latest.depth_m * 0.10)
        residual_score = (
            math.exp(-residual / residual_scale)
            if math.isfinite(residual)
            else 0.0
        )
        motion_score = 1.0 if closing_speed > 0.3 else 0.5
        base_confidence = float(
            0.25 * history_score
            + 0.30 * observation_score
            + 0.20 * residual_score
            + 0.15 * ground_confidence
            + 0.10 * motion_score
        )
        if use_filtered_motion:
            depth_confidence = float(
                np.mean(
                    [
                        observation.depth_confidence
                        for observation in self.observations
                    ]
                )
            )
            base_confidence = (
                0.80 * base_confidence + 0.20 * depth_confidence
            )
            base_confidence *= 0.82**self.missed
        return float(base_confidence)


def bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return float(intersection / max(1, first_area + second_area - intersection))


class ComponentTracker:
    def __init__(
        self,
        image_shape: tuple[int, int],
        *,
        depth_attribute: str = "depth_p20_m",
        maximum_missed: int = 3,
        risk_top_width_fraction: float = 0.16,
        risk_bottom_width_fraction: float = 0.55,
        minimum_bottom_fraction: float = 0.0,
        minimum_height_fraction: float = 0.0,
        # Semantic fusion — None disables it entirely (none backend, bit-identical output).
        semantic_score_threshold: float | None = None,
        semantic_max_misses: int = 3,
        semantic_fallback_depth_m: float = 5.0,
        use_uncertainty_filter: bool = False,
        include_predicted_tracks: bool = False,
    ) -> None:
        self.image_shape = image_shape
        self.depth_attribute = depth_attribute
        self.maximum_missed = maximum_missed
        self.minimum_bottom_fraction = minimum_bottom_fraction
        self.minimum_height_fraction = minimum_height_fraction
        # Semantic fusion config — all None means disabled.
        self._semantic_enabled = (
            semantic_score_threshold is not None and _SEMANTIC_AVAILABLE
        )
        self._semantic_score_threshold = semantic_score_threshold or 0.25
        self._semantic_max_misses = semantic_max_misses
        self._semantic_fallback_depth_m = semantic_fallback_depth_m
        self.use_uncertainty_filter = use_uncertainty_filter
        self.include_predicted_tracks = include_predicted_tracks
        self.tracks: dict[int, ComponentTrack] = {}
        self.next_track_id = 1
        self._risk_corridor = collision_corridor_mask(
            self.image_shape,
            top_width_fraction=risk_top_width_fraction,
            bottom_width_fraction=risk_bottom_width_fraction,
        )
        self._risk_corridor.flags.writeable = False

    def _depth(self, component: ObstacleComponent) -> float:
        depth = float(getattr(component, self.depth_attribute))
        if not math.isfinite(depth):
            return float(component.depth_p35_m)
        return depth

    def _association_cost(
        self, track: ComponentTrack, component: ObstacleComponent
    ) -> float | None:
        previous = track.latest
        diagonal = math.hypot(*self.image_shape)
        center_distance = math.hypot(
            previous.center_x - component.center_x,
            previous.center_y - component.center_y,
        )
        normalized_center = center_distance / diagonal
        overlap = bbox_iou(track.bbox, component.bbox)
        depth = self._depth(component)
        relative_depth = abs(previous.depth_m - depth) / max(
            1.0, previous.depth_m, depth
        )
        if overlap < 0.02 and normalized_center > 0.12:
            return None
        if relative_depth > 0.65:
            return None
        return float(
            0.45 * (1.0 - overlap)
            + 0.35 * min(1.0, normalized_center / 0.12)
            + 0.20 * min(1.0, relative_depth / 0.65)
        )

    def update(
        self,
        components: Iterable[ObstacleComponent],
        timestamp: float,
        detections: "Sequence[Detection] | None" = None,
    ) -> list[ComponentTrack]:
        """Update tracks with new obstacle components.

        Args:
            components: Obstacle components from the current stereo frame.
            timestamp: Frame timestamp in seconds.
            detections: Optional YOLO detections for this frame. Pass None (or
                omit) when using the ``none`` backend — this preserves
                bit-identical output with the pre-semantic pipeline.
        """
        components = list(components)
        # Semantic: use empty list when detections omitted — treated as all-miss.
        frame_dets: Sequence[Detection] = detections if detections is not None else []

        candidates = []
        for track_id, track in self.tracks.items():
            for component_index, component in enumerate(components):
                cost = self._association_cost(track, component)
                if cost is not None:
                    candidates.append((cost, track_id, component_index))
        candidates.sort()

        matched_tracks: set[int] = set()
        matched_components: set[int] = set()
        for _, track_id, component_index in candidates:
            if track_id in matched_tracks or component_index in matched_components:
                continue
            component = components[component_index]
            self.tracks[track_id].update(
                component, timestamp, self._depth(component)
            )
            matched_tracks.add(track_id)
            matched_components.add(component_index)

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track.age += 1
                track.missed += 1
                track.current_timestamp = timestamp
            if track.missed > self.maximum_missed:
                del self.tracks[track_id]

        for component_index, component in enumerate(components):
            if component_index in matched_components:
                continue
            track = ComponentTrack(
                track_id=self.next_track_id,
                bbox=component.bbox,
            )
            if self.use_uncertainty_filter:
                track.enable_filtered_motion()
            track.update(component, timestamp, self._depth(component))
            if self._semantic_enabled:
                track.enable_semantic()
            self.tracks[track.track_id] = track
            self.next_track_id += 1

        # Update semantic state for all active tracks this frame.
        # When semantic is disabled, update_semantic() is a no-op — no change
        # to TTC output, preserving bit-identical none-backend behaviour.
        if self._semantic_enabled:
            for track in self.tracks.values():
                if track.missed == 0:
                    track.update_semantic(frame_dets, self.image_shape)
                # Missed tracks still accumulate consecutive_misses via EMA decay
                # (matched_confidence=0 path), handled inside TemporalSemanticState.update.
                elif track.semantic_state is not None:
                    from semantic_fusion import SemanticAssociation
                    track.semantic_state.update(SemanticAssociation(matched=False))

        return [
            track
            for track in self.tracks.values()
            if (
                track.missed <= min(2, self.maximum_missed)
                if self.include_predicted_tracks
                else track.missed == 0
            )
        ]

    def reset(self) -> None:
        """Reset all tracker state between trips.

        Clears all active tracks (including their semantic states) and resets
        the track ID counter so successive trip runs are fully independent.
        """
        self.tracks.clear()
        self.next_track_id = 1

    def risk_tracks(self, tracks: Iterable[ComponentTrack]) -> list[ComponentTrack]:
        height, width = self.image_shape
        selected = []
        for track in tracks:
            if not track.confirmed:
                continue
            x0, y0, x1, y1 = track.bbox
            if y1 / height < self.minimum_bottom_fraction:
                continue
            if (y1 - y0) / height < self.minimum_height_fraction:
                continue
            center_x = int(np.clip((x0 + x1) / 2, 0, width - 1))
            bottom_y = int(np.clip(y1 - 1, 0, height - 1))
            if self._risk_corridor[bottom_y, center_x]:
                selected.append(track)
        return selected


def select_minimum_ttc(
    tracks: Iterable[ComponentTrack],
    ground_confidence: float,
    *,
    minimum_track_confidence: float = 0.55,
    maximum_closing_speed_mps: float = 40.0,
    maximum_depth_m: float = math.inf,
    maximum_motion_residual_m: float = math.inf,
    # Semantic soft-guard — only active when ComponentTracker was initialised
    # with semantic_score_threshold != None.  Defaults keep none-backend
    # bit-identical (is_semantically_suppressed() returns False when
    # semantic_state is None).
    semantic_score_threshold: float = 0.25,
    semantic_max_misses: int = 3,
    semantic_fallback_depth_m: float = 5.0,
    use_filtered_motion: bool = False,
) -> tuple[float, int | None, float, float]:
    best = (math.inf, None, 0.0, 0.0)
    for track in tracks:
        closing_speed, ttc, residual = (
            track.filtered_motion_state()
            if use_filtered_motion
            else track.motion_state()
        )
        confidence = track.confidence(
            ground_confidence,
            use_filtered_motion=use_filtered_motion,
        )
        if confidence < minimum_track_confidence:
            continue
        if closing_speed > maximum_closing_speed_mps:
            continue
        if track.latest.depth_m > maximum_depth_m:
            continue
        if residual > maximum_motion_residual_m:
            continue
        # Semantic soft-guard: skip this candidate if suppressed.
        # When semantic_state is None (none backend), this is always False —
        # output is bit-identical to the pre-semantic pipeline.
        if track.is_semantically_suppressed(
            score_threshold=semantic_score_threshold,
            max_misses=semantic_max_misses,
            fallback_depth_m=semantic_fallback_depth_m,
        ):
            continue
        if ttc < best[0]:
            best = (ttc, track.track_id, confidence, closing_speed)
    return best
