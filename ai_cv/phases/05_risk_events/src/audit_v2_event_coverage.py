"""Audit V2 eligibility coverage before attempting another score experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path


ROAD_USERS = {"car", "truck", "bus", "motorcycle", "bicycle", "person", "pedestrian"}


def _finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ground_truth(trip_path: Path) -> dict[int, float]:
    with gzip.open(trip_path / f"{trip_path.name}.json.gz", "rt", encoding="utf-8") as handle:
        return {
            int(frame["frame_id"]): float(frame["min_ttc"])
            for frame in json.load(handle)["frames"]
        }


def _t05_category(row: dict[str, str]) -> str:
    if not row["union_source"].startswith("classical"):
        return "not_classical_candidate"
    detections = json.loads(row["raw_detections_json"])
    if not any(item["class_name"].lower() in ROAD_USERS for item in detections):
        return "no_yolo_road_user"
    iou = _finite_float(row["v2_event_match_iou"]) or 0.0
    if iou < 0.30:
        return "no_classical_yolo_iou_match"
    occupancy = _finite_float(row["v2_event_occupancy"])
    if occupancy is None:
        return "matched_occupancy_unavailable"
    if occupancy < 0.50:
        return "eligible_low_occupancy"
    return "eligible_not_low_occupancy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    total = Counter()
    trips: dict[str, dict[str, int]] = {}
    t05_taxonomy = Counter()
    # Evidence is normally kept under an ignored run-specific subdirectory
    # (for example ``.../phase13/evidence``).  Searching recursively keeps
    # the audit independent of that layout; callers should pass the one
    # evidence directory, not the whole output tree containing alternatives.
    for evidence_path in sorted(args.evidence_root.rglob("T*-Sample.csv")):
        trip_id = evidence_path.stem
        truth = _ground_truth(args.practice_root / trip_id)
        counts = Counter()
        with evidence_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_ttc = _finite_float(row["union_predicted_ttc"])
                if raw_ttc is None or raw_ttc >= 2.0:
                    continue
                counts["v1_danger"] += 1
                is_tp = truth[int(row["frame_id"])] < 2.0
                counts["v1_tp" if is_tp else "v1_fp"] += 1
                iou = _finite_float(row["v2_event_match_iou"]) or 0.0
                occupancy = _finite_float(row["v2_event_occupancy"])
                if iou >= 0.30 and occupancy is not None:
                    counts["v2_eligible"] += 1
                    counts["v2_eligible_tp" if is_tp else "v2_eligible_fp"] += 1
                if trip_id == "T05-Sample" and not is_tp:
                    t05_taxonomy[_t05_category(row)] += 1
        total.update(counts)
        trips[trip_id] = dict(counts)

    report = {
        "contract": {
            "v1_danger": "conservative union TTC < 2.0 s",
            "v2_eligible": "classical-to-YOLO IoU >= 0.30 and finite occupancy",
            "ground_truth_danger": "trusted TTC < 2.0 s",
        },
        "overall": dict(total),
        "per_trip": trips,
        "t05_false_positive_taxonomy": dict(t05_taxonomy),
        "decision": "coverage_only_do_not_change_v2_parameters",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
