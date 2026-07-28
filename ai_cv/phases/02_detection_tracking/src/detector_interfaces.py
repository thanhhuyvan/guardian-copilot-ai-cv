"""Detector interfaces and data structures for Phase 04B YOLO26 Semantic Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    backend: str
    precision: str
    input_shape: tuple[int, int, int, int]
    model_sha256: str
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class ObjectDetector(Protocol):
    def infer(self, left_bgr: np.ndarray) -> DetectionResult:
        ...

    def close(self) -> None:
        ...
