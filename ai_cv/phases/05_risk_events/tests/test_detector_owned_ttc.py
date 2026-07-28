from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
PHASE02_SRC = Path(__file__).resolve().parents[2] / "02_detection_tracking" / "src"
for path in (SRC, PHASE02_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from classical_geometry import collision_corridor_mask
from detector_interfaces import Detection
from evaluate_detector_owned_ttc import (
    _apply_suppressed_ttc_floor,
    _best_road_user_detection_iou,
    _path_intersection_geometry,
    _prefixed_evidence,
    _track_measurements_json,
    detection_component,
    detection_evidence_json,
)


def test_path_intersection_rejects_a_side_track_but_keeps_an_entering_track() -> None:
    class Observation:
        def __init__(self, timestamp: float, depth_m: float, center_x: float):
            self.timestamp = timestamp
            self.depth_m = depth_m
            self.center_x = center_x

    class Track:
        def __init__(self, centers: list[float]):
            self.observations = [
                Observation(index * 0.1, 10.0 - index * 0.4, center)
                for index, center in enumerate(centers)
            ]

    # Fixed 3.6 m lateral offset: image centre shifts outward as depth shrinks.
    side, separation = _path_intersection_geometry(
        Track([500.0, 508.0, 517.0, 527.0, 534.0]),
        ttc_sec=1.5,
        focal_length_px=500.0,
        principal_x_px=320.0,
        ego_lateral_accel_mps2=0.0,
        corridor_half_width_m=1.75,
    )
    entering, _ = _path_intersection_geometry(
        Track([415.0, 414.0, 413.0, 411.0, 409.0]),
        ttc_sec=1.5,
        focal_length_px=500.0,
        principal_x_px=320.0,
        ego_lateral_accel_mps2=0.0,
        corridor_half_width_m=1.75,
    )

    assert side is False
    assert separation is not None and separation > 1.75
    assert entering is True


def test_classical_track_yolo_association_uses_only_road_users() -> None:
    class Track:
        bbox = (10, 10, 30, 30)

    detections = [
        Detection((10.0, 10.0, 30.0, 30.0), 56, "chair", 0.9),
        Detection((15.0, 10.0, 35.0, 30.0), 2, "car", 0.9),
    ]

    assert _best_road_user_detection_iou(Track(), detections) == 0.6
    assert _best_road_user_detection_iou(None, detections) == 0.0


def test_shadow_measurements_keep_only_latest_track_observation() -> None:
    class Latest:
        timestamp = 1.25
        depth_m = 8.0
        center_x = 300.0
        depth_sigma_m = 0.5

    class Track:
        track_id = 4
        latest = Latest()

    assert json.loads(_track_measurements_json([Track()])) == [
        {"track_id": 4, "timestamp": 1.25, "depth_m": 8.0, "center_x": 300.0, "depth_sigma_m": 0.5}
    ]
from diagnose_temporal_depth_consistency import bbox_iou as diagnostic_bbox_iou


def test_detection_evidence_retains_raw_yolo_identity_and_box() -> None:
    evidence = json.loads(
        detection_evidence_json(
            [
                Detection((1.25, 2.0, 30.5, 41.0), 0, "person", 0.81234567),
                Detection((5.0, 6.0, 50.0, 60.0), 2, "car", 0.4),
            ]
        )
    )

    assert evidence == [
        {
            "bbox_xyxy": [1.25, 2.0, 30.5, 41.0],
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.812346,
        },
        {
            "bbox_xyxy": [5.0, 6.0, 50.0, 60.0],
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.4,
        },
    ]


def test_prefixed_evidence_keeps_detector_and_classical_fields_distinct() -> None:
    assert _prefixed_evidence("classical", {"track_id": 7, "quality": 0.8}) == {
        "classical_track_id": 7,
        "classical_quality": 0.8,
    }


def test_suppressed_ttc_floor_keeps_non_danger_output_finite() -> None:
    assert _apply_suppressed_ttc_floor(
        raw_ttc=1.2, gated_ttc=float("inf"), floor_ttc=2.0
    ) == (2.0, True)
    assert _apply_suppressed_ttc_floor(
        raw_ttc=1.2, gated_ttc=1.5, floor_ttc=2.0
    ) == (1.5, False)


def test_temporal_depth_diagnostic_bbox_iou_is_symmetric() -> None:
    first = [0.0, 0.0, 10.0, 10.0]
    second = [5.0, 0.0, 15.0, 10.0]

    assert diagnostic_bbox_iou(first, second) == diagnostic_bbox_iou(second, first)
    assert diagnostic_bbox_iou(first, second) == 1 / 3


def test_detection_component_uses_nearer_supported_disparity_mode() -> None:
    disparity = np.zeros((100, 160), dtype=np.float32)
    valid = np.zeros_like(disparity, dtype=bool)
    # A broad background surface and a smaller, nearer object surface.
    disparity[35:75, 55:105] = 8.0
    disparity[45:70, 68:94] = 20.0
    valid[35:75, 55:105] = True
    detection = Detection(
        bbox_xyxy=(55.0, 35.0, 105.0, 75.0),
        class_id=2,
        class_name="car",
        confidence=0.8,
    )

    component = detection_component(
        detection,
        disparity,
        valid,
        focal_length_px=100.0,
        baseline_m=0.5,
        corridor_mask=collision_corridor_mask((100, 160)),
        component_id=1,
    )

    assert component is not None
    assert component.object_depth_m == component.depth_m
    assert 2.3 < component.depth_m < 2.7


def test_detection_component_rejects_non_road_user_and_sparse_depth() -> None:
    disparity = np.full((80, 120), 12.0, dtype=np.float32)
    valid = np.zeros_like(disparity, dtype=bool)
    corridor = collision_corridor_mask((80, 120))
    chair = Detection((40, 30, 80, 70), 56, "chair", 0.9)
    car = Detection((40, 30, 80, 70), 2, "car", 0.9)

    assert detection_component(
        chair, disparity, valid, 100.0, 0.5, corridor, component_id=1
    ) is None
    assert detection_component(
        car, disparity, valid, 100.0, 0.5, corridor, component_id=2
    ) is None


def test_detection_component_penalizes_lr_inconsistent_depth() -> None:
    disparity = np.full((100, 160), 20.0, dtype=np.float32)
    valid = np.ones_like(disparity, dtype=bool)
    confidence = np.zeros_like(disparity, dtype=np.float32)
    detection = Detection((45, 30, 115, 90), 2, "car", 0.9)

    component = detection_component(
        detection,
        disparity,
        valid,
        100.0,
        0.5,
        collision_corridor_mask((100, 160)),
        component_id=1,
        stereo_confidence=confidence,
    )

    assert component is not None
    assert component.object_depth_confidence < 0.45
