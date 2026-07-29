"""Partition frozen T05 false-danger frames by association and path geometry."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE02_SRC = REPOSITORY_ROOT / "ai_cv" / "phases" / "02_detection_tracking" / "src"
if str(PHASE02_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE02_SRC))

from evaluate_path_only_gate import (  # noqa: E402
    best_yolo_update,
    direct_path_offset_m,
    load_truth,
    parse_ttc,
)


def classify(
    row: dict[str, str], *, minimum_iou: float, corridor_half_width_m: float,
    focal_px: float, principal_x_px: float,
) -> dict[str, object]:
    """Classify one false-danger frame; no result is used to change a TTC."""
    update, overlap = best_yolo_update(row, minimum_iou=minimum_iou)
    result: dict[str, object] = {
        "frame_id": int(row["frame_id"]),
        "raw_ttc": parse_ttc(row["union_predicted_ttc"]),
        "union_source": row["union_source"],
        "matched_iou": overlap,
        "path_offset_m": None,
        "category": "unassociated",
    }
    if update is None:
        return result
    offset = direct_path_offset_m(
        row, update, focal_px=focal_px, principal_x_px=principal_x_px
    )
    if offset is None:
        result["category"] = "geometry_unavailable"
        return result
    result["path_offset_m"] = offset
    result["category"] = (
        "off_path" if abs(offset) > corridor_half_width_m else "on_path_nonclosing"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.30)
    parser.add_argument("--corridor-half-width-m", type=float, default=1.75)
    parser.add_argument("--focal-px", type=float, default=320.0)
    parser.add_argument("--principal-x-px", type=float, default=320.0)
    args = parser.parse_args()

    with args.evidence.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    trip_id = args.evidence.stem
    truth = load_truth(args.practice_root, trip_id)
    if len(rows) != len(truth):
        raise ValueError("evidence and truth frame counts differ")
    false_danger = [
        row for row, target in zip(rows, truth, strict=True)
        if parse_ttc(row["union_predicted_ttc"]) < 2.0 and target >= 2.0
    ]
    audited = [
        classify(
            row, minimum_iou=args.minimum_iou,
            corridor_half_width_m=args.corridor_half_width_m,
            focal_px=args.focal_px, principal_x_px=args.principal_x_px,
        )
        for row in false_danger
    ]
    report = {
        "trip_id": trip_id,
        "policy": {
            "minimum_iou": args.minimum_iou,
            "corridor_half_width_m": args.corridor_half_width_m,
        },
        "false_danger_frames": len(audited),
        "category_counts": dict(Counter(str(item["category"]) for item in audited)),
        "rows": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
