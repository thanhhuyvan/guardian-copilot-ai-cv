"""Semantic association and temporal track state for Phase 04B YOLO26 Fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from detector_interfaces import Detection


RETAINED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True)
class SemanticAssociation:
    matched: bool
    class_id: int | None = None
    class_name: str | None = None
    confidence: float = 0.0
    iou: float = 0.0


def expand_and_clip_box(
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int],
    expand_fraction: float = 0.10,
) -> Tuple[float, float, float, float]:
    """Expand box by fraction (10%) in width and height, clipped to image bounds."""
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    dx = width * expand_fraction / 2.0
    dy = height * expand_fraction / 2.0

    img_h, img_w = image_shape[:2]
    new_x0 = max(0.0, x0 - dx)
    new_y0 = max(0.0, y0 - dy)
    new_x1 = min(float(img_w), x1 + dx)
    new_y1 = min(float(img_h), y1 + dy)

    return (new_x0, new_y0, new_x1, new_y1)


def compute_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x0, y0, x1, y1) bounding boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def compute_vertical_overlap(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute vertical overlap fraction relative to smaller box height."""
    _, ay0, _, ay1 = box_a
    _, by0, _, by1 = box_b

    iy0 = max(ay0, by0)
    iy1 = min(ay1, by1)
    inter_h = max(0.0, iy1 - iy0)

    h_a = max(1e-5, ay1 - ay0)
    h_b = max(1e-5, by1 - by0)
    min_h = min(h_a, h_b)

    return inter_h / min_h


def compute_intersection_over_box(
    container_candidate: Tuple[float, float, float, float],
    reference_box: Tuple[float, float, float, float],
) -> float:
    """Return intersection area divided by ``reference_box`` area.

    Unlike IoU, this remains high when a small detection is contained inside a
    much larger stereo component. That geometry occurs when disparity merges a
    road user with nearby road or guardrail pixels.
    """
    ax0, ay0, ax1, ay1 = container_candidate
    bx0, by0, bx1, by1 = reference_box
    intersection_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    reference_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if reference_area <= 0.0:
        return 0.0
    return intersection_width * intersection_height / reference_area


def point_in_box(
    px: float,
    py: float,
    box: Tuple[float, float, float, float],
) -> bool:
    """Check if point (px, py) is strictly inside box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    return x0 <= px <= x1 and y0 <= py <= y1


def associate_component_with_detections(
    component_bbox: Tuple[int, int, int, int],
    detections: Sequence[Detection],
    image_shape: Tuple[int, int],
    retained_class_ids: set[int] | None = None,
) -> SemanticAssociation:
    """
    Associate a disparity component/track bbox with detections according to Phase 04B rules:
    1. Expand accepted detection box by 10% clipped to image.
    2. Match if IoU >= 0.15, component center lies in the expanded detection
       with vertical overlap >= 0.50, or the expanded detection center lies in
       the component with >= 0.50 detection-area coverage and vertical overlap.
       The symmetric containment branch handles disparity components merged
       around a valid object without accepting a mere edge contact.
    3. Select match with highest score: 0.60 * confidence + 0.40 * IoU.
    """
    if retained_class_ids is None:
        retained_class_ids = set(RETAINED_CLASSES.keys())

    comp_x0, comp_y0, comp_x1, comp_y1 = component_bbox
    comp_box = (float(comp_x0), float(comp_y0), float(comp_x1), float(comp_y1))
    comp_cx = (comp_x0 + comp_x1) / 2.0
    comp_cy = (comp_y0 + comp_y1) / 2.0

    best_score = -1.0
    best_match: SemanticAssociation | None = None

    for det in detections:
        if det.class_id not in retained_class_ids:
            continue

        exp_det_box = expand_and_clip_box(det.bbox_xyxy, image_shape, expand_fraction=0.10)
        iou = compute_iou(comp_box, exp_det_box)
        vert_overlap = compute_vertical_overlap(comp_box, exp_det_box)
        center_inside = point_in_box(comp_cx, comp_cy, exp_det_box)
        det_cx = (exp_det_box[0] + exp_det_box[2]) / 2.0
        det_cy = (exp_det_box[1] + exp_det_box[3]) / 2.0
        detection_center_inside = point_in_box(det_cx, det_cy, comp_box)
        detection_coverage = compute_intersection_over_box(comp_box, exp_det_box)

        matched = (
            (iou >= 0.15)
            or (center_inside and vert_overlap >= 0.50)
            or (
                detection_center_inside
                and detection_coverage >= 0.50
                and vert_overlap >= 0.50
            )
        )

        if matched:
            match_score = 0.60 * det.confidence + 0.40 * iou
            if match_score > best_score:
                best_score = match_score
                best_match = SemanticAssociation(
                    matched=True,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    iou=iou,
                )

    if best_match is not None:
        return best_match

    return SemanticAssociation(matched=False)


@dataclass
class TemporalSemanticState:
    score: float = 0.0
    consecutive_misses: int = 0
    matched_class_id: int | None = None
    matched_class_name: str | None = None
    last_matched_confidence: float = 0.0

    def update(self, assoc: SemanticAssociation) -> None:
        matched_conf = assoc.confidence if assoc.matched else 0.0
        self.score = 0.4 * matched_conf + 0.6 * self.score

        if assoc.matched:
            self.consecutive_misses = 0
            self.matched_class_id = assoc.class_id
            self.matched_class_name = assoc.class_name
            self.last_matched_confidence = assoc.confidence
        else:
            self.consecutive_misses += 1

    def has_semantic_support(self, threshold: float = 0.25) -> bool:
        return self.score >= threshold

    def is_suppressed(
        self,
        latest_depth_m: float,
        score_threshold: float = 0.25,
        max_misses: int = 3,
        fallback_depth_m: float = 5.0,
    ) -> bool:
        """
        soft-guard logic:
        Reject TTC candidate only when ALL conditions hold:
        - no semantic support (score < score_threshold)
        - at least max_misses consecutive misses
        - depth > fallback_depth_m
        """
        if latest_depth_m <= fallback_depth_m:
            return False  # Close-range fallback preserves candidate

        no_support = self.score < score_threshold
        sufficient_misses = self.consecutive_misses >= max_misses

        return no_support and sufficient_misses
