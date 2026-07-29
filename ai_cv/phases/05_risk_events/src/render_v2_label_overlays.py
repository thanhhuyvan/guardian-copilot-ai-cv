"""Render review images for human V2 stereo-track path labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    evidence: dict[tuple[str, str], dict[str, str]] = {}
    for evidence_file in args.evidence_root.glob("T*-Sample.csv"):
        with evidence_file.open(encoding="utf-8", newline="") as handle:
            evidence.update({(evidence_file.stem, row["frame_id"]): row for row in csv.DictReader(handle)})

    with args.labels.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    for label in labels:
        row = evidence[(label["trip_id"], label["frame_id"])]
        track_id = int(label["track_id"])
        tracks = json.loads(row["classical_track_measurements_json"])
        track = next(track for track in tracks if int(track["track_id"]) == track_id)
        image = cv2.imread(label["left_image_path"])
        if image is None:
            raise FileNotFoundError(label["left_image_path"])
        x = round(float(track["center_x"]))
        height, width = image.shape[:2]
        cv2.line(image, (x, 0), (x, height - 1), (255, 255, 0), 2)
        cv2.putText(
            image,
            f"V2 stereo track {track_id}: x={x}, depth={float(track['depth_m']):.1f}m",
            (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA,
        )
        for detection in json.loads(row["raw_detections_json"]):
            x0, y0, x1, y1 = (round(value) for value in detection["bbox_xyxy"])
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 1)
        output = args.output_dir / label["trip_id"] / f"{int(label['frame_id']):06d}_track_{track_id}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"could not write {output}")


if __name__ == "__main__":
    main()
