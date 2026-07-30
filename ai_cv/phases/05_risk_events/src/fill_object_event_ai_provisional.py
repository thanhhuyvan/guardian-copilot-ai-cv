"""Create an explicitly non-ground-truth AI provisional object-event label copy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {}
    for path in args.evidence_root.glob("T*-Sample.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            evidence.update({(path.stem, row["frame_id"]): row for row in csv.DictReader(handle)})
    with args.labels.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        source = evidence[(row["trip_id"], row["frame_id"])]
        raw_box = source.get("union_selected_bbox_xyxy") or source.get("classical_selected_bbox_xyxy")
        box = json.loads(raw_box) if raw_box else [0, 0, 0, 0]
        centre_x = (float(box[0]) + float(box[2])) / 2.0
        relation = "on_path" if 256.0 <= centre_x <= 384.0 else "adjacent"
        row.update({
            "object_id_window": f"ai-{row['trip_id']}-{row['selected_track_id'] or row['frame_id']}",
            "event_owner": "yes" if relation == "on_path" else "no",
            "path_relation": relation,
            "relative_motion": "closing",
            "cpa_distance_m": "unknown",
            "occluded": "unknown",
            "review_confidence": "low",
            "notes": "PROVISIONAL_AI_ASSUMPTION; camera-position heuristic only; not ground truth",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} explicitly provisional AI rows to {args.output}")


if __name__ == "__main__":
    main()
