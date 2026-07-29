from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_temporal_multicue_association import _rank_candidates


def test_rank_prefers_contained_continuous_track_without_depth() -> None:
    classical = (0.0, 0.0, 100.0, 100.0)
    updates = [
        {"track_id": 1, "measurement_source": "yolo_box_median_disparity", "bbox_xyxy": [45, 45, 55, 55], "depth_m": 1000},
        {"track_id": 2, "measurement_source": "yolo_box_median_disparity", "bbox_xyxy": [48, 48, 52, 52], "depth_m": 1},
    ]
    ranked = _rank_candidates(classical, updates, previous_track=1)
    assert ranked[0]["track_id"] == 1
    assert ranked[0]["continues"] is True
