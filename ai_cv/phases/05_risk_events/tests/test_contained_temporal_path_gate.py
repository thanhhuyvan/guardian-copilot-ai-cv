from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate_contained_temporal_path_gate import gate_trip


def _row(frame_id: int) -> dict[str, str]:
    return {
        "frame_id": str(frame_id), "union_predicted_ttc": "1.0",
        "union_source": "classical", "ego_speed_mps": "10.0",
        "lateral_accel_mps2": "0.0",
        "classical_selected_bbox_xyxy": "[80, 100, 140, 220]",
        "v2_shadow_updates_json": (
            '[{"measurement_source":"yolo_box_median_disparity",'
            '"track_id":4,"bbox_xyxy":[90,100,110,200],"depth_m":10.0}]'
        ),
    }


def test_contained_temporal_gate_fails_open_then_gates_off_path() -> None:
    predictions, audit = gate_trip(
        [_row(10), _row(11)], focal_px=320.0, principal_x_px=320.0,
        corridor_half_width_m=1.75,
    )
    assert predictions.tolist() == [1.0, 2.0]
    assert audit[0]["association_status"] == "episode_start_or_no_continuity"
    assert audit[1]["suppressed"]
