from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate_path_only_gate import direct_path_offset_m, gated_ttc, iou


def _row(*, source: str = "classical") -> dict[str, str]:
    return {
        "frame_id": "1",
        "union_predicted_ttc": "1.0",
        "union_source": source,
        "classical_selected_bbox_xyxy": "[90, 100, 110, 200]",
        "v2_shadow_updates_json": "[]",
        "ego_speed_mps": "10.0",
        "lateral_accel_mps2": "0.0",
    }


def test_iou_and_direct_offset_are_current_frame_geometry() -> None:
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    row = _row()
    update = {"bbox_xyxy": [90, 100, 110, 200], "depth_m": 10.0}
    assert direct_path_offset_m(row, update, focal_px=320.0, principal_x_px=320.0) == -6.875


def test_path_only_gate_is_finite_and_never_suppresses_without_association() -> None:
    row = _row()
    row["v2_shadow_updates_json"] = '[{"measurement_source":"yolo_box_median_disparity","bbox_xyxy":[90,100,110,200],"depth_m":10.0}]'
    gated, audit = gated_ttc(
        row, focal_px=320.0, principal_x_px=320.0,
        minimum_iou=0.30, corridor_half_width_m=1.75,
    )
    assert gated == 2.0
    assert audit["suppressed"]
    row["v2_shadow_updates_json"] = "[]"
    preserved, audit = gated_ttc(
        row, focal_px=320.0, principal_x_px=320.0,
        minimum_iou=0.30, corridor_half_width_m=1.75,
    )
    assert preserved == 1.0
    assert not audit["suppressed"]
