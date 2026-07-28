"""Deterministic, in-memory robustness inputs and fallback contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np


VISUAL_PERTURBATIONS = ("blur", "darkness", "noise", "occlusion")
FAULT_REASONS = (
    "missing_left_camera",
    "missing_right_camera",
    "corrupt_stereo_pair",
    "invalid_calibration",
    "detector_exception",
    "tracker_exception",
    "frame_drop",
    "irregular_timestamp",
)


@dataclass(frozen=True)
class Perturbation:
    kind: str
    severity: int

    def __post_init__(self) -> None:
        if self.kind not in VISUAL_PERTURBATIONS:
            raise ValueError(f"unsupported visual perturbation: {self.kind}")
        if self.severity not in {1, 2, 3}:
            raise ValueError("severity must be 1, 2, or 3")


def _seed(trip_id: str, frame_id: int, kind: str, severity: int) -> int:
    text = f"{trip_id}:{frame_id}:{kind}:{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def apply_perturbation(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    trip_id: str,
    frame_id: int,
    perturbation: Perturbation,
) -> tuple[np.ndarray, np.ndarray]:
    """Return modified copies; never mutate or persist source frames."""
    left = left_bgr.copy()
    right = right_bgr.copy()
    severity = perturbation.severity

    if perturbation.kind == "blur":
        kernel = (3, 7, 11)[severity - 1]
        return (
            cv2.GaussianBlur(left, (kernel, kernel), 0),
            cv2.GaussianBlur(right, (kernel, kernel), 0),
        )

    if perturbation.kind == "darkness":
        alpha = (0.70, 0.45, 0.25)[severity - 1]
        return (
            cv2.convertScaleAbs(left, alpha=alpha, beta=0),
            cv2.convertScaleAbs(right, alpha=alpha, beta=0),
        )

    if perturbation.kind == "noise":
        sigma = (8.0, 18.0, 32.0)[severity - 1]
        generator = np.random.default_rng(
            _seed(trip_id, frame_id, perturbation.kind, severity)
        )
        # One noise field per camera preserves deterministic but not identical
        # stereo corruption, matching independently noisy image sensors.
        left_noise = generator.normal(0.0, sigma, left.shape)
        right_noise = generator.normal(0.0, sigma, right.shape)
        return (
            np.clip(left.astype(np.float32) + left_noise, 0, 255).astype(np.uint8),
            np.clip(right.astype(np.float32) + right_noise, 0, 255).astype(np.uint8),
        )

    height, width = left.shape[:2]
    fraction = (0.15, 0.25, 0.35)[severity - 1]
    box_width = int(width * fraction)
    box_height = int(height * fraction)
    x0 = width // 2 - box_width // 2
    y0 = height - box_height
    left[y0:height, x0 : x0 + box_width] = 0
    right[y0:height, x0 : x0 + box_width] = 0
    return left, right


def screening_selector(frame: object, *, safe_stride: int = 8) -> bool:
    """Keep every danger frame plus a deterministic representative safe set."""
    if safe_stride < 1:
        raise ValueError("safe_stride must be positive")
    return float(frame.min_ttc) < 3.0 or int(frame.frame_id) % safe_stride == 0


def unknown_perception_document(
    *,
    trip_id: str,
    frame_id: int,
    timestamp: float,
    latency_ms: float,
    reason: str,
) -> dict[str, object]:
    if reason not in FAULT_REASONS:
        raise ValueError(f"unsupported fault reason: {reason}")
    return {
        "schema_version": "perception.v1",
        "run_id": "phase06-robustness",
        "trip_id": trip_id,
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "image_width": 640,
        "image_height": 360,
        "status": "unknown",
        "objects": [],
        "min_ttc_sec": None,
        "risk_level": "UNKNOWN",
        "perception_quality": 0.0,
        "latency_ms": float(latency_ms),
        "degraded_reasons": [reason],
    }
