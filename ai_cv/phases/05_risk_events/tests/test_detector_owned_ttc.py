from __future__ import annotations

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
from evaluate_detector_owned_ttc import detection_component


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
