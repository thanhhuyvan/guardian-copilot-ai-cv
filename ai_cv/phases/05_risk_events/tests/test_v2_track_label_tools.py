from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"


def test_label_audit_refuses_incomplete_labels(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "trip_id,frame_id,left_image_path,track_id,mahalanobis_squared,occupancy_probability,path_relation,cpa_distance_m,occluded,notes\n"
        "T01-Sample,0,x.jpg,1,0.0,0.1,,,,\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(SRC / "audit_v2_track_labels.py"), "--labels", str(labels), "--output", str(report)],
        check=True,
    )
    assert json.loads(report.read_text(encoding="utf-8"))["decision"] == "labels_incomplete_no_risk_gate_decision"


def test_association_audit_refuses_incomplete_labels(tmp_path: Path) -> None:
    labels = tmp_path / "association.csv"
    labels.write_text(
        "trip_id,frame_id,sample_stratum,classical_bbox_xyxy,classical_depth_m,proposed_track_id,proposed_bbox_xyxy,proposed_depth_m,bbox_iou,centre_distance_px,depth_delta_m,rank,same_object,notes\n"
        "T05-Sample,1,false_alert_unmatched,[],1,1,[],1,0,1,0,1,,\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(SRC / "audit_t05_association_labels.py"), "--labels", str(labels), "--output", str(report)],
        check=True,
    )
    assert json.loads(report.read_text(encoding="utf-8"))["decision"] == "incomplete_labels_no_association_decision"
