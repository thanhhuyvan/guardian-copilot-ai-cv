"""Sample containment-association candidates across all trips for visual review."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import cv2

from audit_containment_association import _bbox, _contains_centre, _number, _truth


FIELDS = [
    "trip_id", "frame_id", "sample_stratum", "classical_bbox_xyxy", "proposed_track_id",
    "proposed_bbox_xyxy", "same_object", "notes",
]


def _spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    indices = {round(index * (len(rows) - 1) / (count - 1)) for index in range(count)}
    return [row for index, row in enumerate(rows) if index in indices]


def _draw(image, bbox, color, label):
    x0, y0, x1, y1 = (round(value) for value in bbox)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, label, (x0, max(16, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=2)
    args = parser.parse_args()
    selected: list[dict[str, str]] = []
    for evidence_path in sorted(args.evidence_root.glob("T*-Sample.csv")):
        trip_id, truth = evidence_path.stem, _truth(args.practice_root / evidence_path.stem)
        grouped = {"false_alert": [], "true_danger": []}
        with evidence_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["union_source"].startswith("classical") or _number(row["union_predicted_ttc"]) >= 2.0:
                    continue
                classical = _bbox(row["classical_selected_bbox_xyxy"])
                if classical is None:
                    continue
                contained = [
                    update for update in json.loads(row.get("v2_shadow_updates_json", "[]"))
                    if update.get("measurement_source") == "yolo_box_median_disparity"
                    and (box := _bbox(update.get("bbox_xyxy", []))) is not None
                    and _contains_centre(classical, box)
                ]
                if len(contained) != 1:
                    continue
                update = contained[0]
                stratum = "true_danger" if truth[int(row["frame_id"])] < 2.0 else "false_alert"
                grouped[stratum].append({
                    "trip_id": trip_id, "frame_id": row["frame_id"], "sample_stratum": stratum,
                    "classical_bbox_xyxy": row["classical_selected_bbox_xyxy"],
                    "proposed_track_id": str(update["track_id"]),
                    "proposed_bbox_xyxy": json.dumps(update["bbox_xyxy"]),
                    "same_object": "", "notes": "",
                })
        for stratum, rows in grouped.items():
            selected.extend(_spaced(rows, args.per_stratum))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "containment_validation_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(selected)
    for row in selected:
        image_path = args.practice_root / row["trip_id"] / "kitti" / "image_2" / f"{int(row['frame_id']):06d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        _draw(image, _bbox(row["classical_bbox_xyxy"]), (255, 0, 255), "classical")
        _draw(image, _bbox(row["proposed_bbox_xyxy"]), (0, 255, 255), "YOLO containment")
        output = args.output_dir / "overlays" / row["trip_id"] / row["sample_stratum"] / f"{int(row['frame_id']):06d}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"could not write {output}")
    print(json.dumps({"label_rows": len(selected), "per_stratum": args.per_stratum}, indent=2))


if __name__ == "__main__":
    main()
