"""Render short, blind temporal clips for track/path validation labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from audit_containment_association import _bbox


def _draw(image, bbox, colour, text):
    x0, y0, x1, y1 = (round(value) for value in bbox)
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(image, text, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=5)
    args = parser.parse_args()

    evidence: dict[tuple[str, int], dict[str, str]] = {}
    for path in args.evidence_root.glob("T*-Sample.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            evidence.update({
                (path.stem, int(row["frame_id"])): row
                for row in csv.DictReader(handle)
            })
    with args.labels.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))

    written = 0
    for label in labels:
        trip_id, frame_id, track_id = label["trip_id"], int(label["frame_id"]), int(label["track_id"])
        output = args.output_dir / trip_id / f"{frame_id:06d}_track_{track_id}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        for current in range(max(0, frame_id - args.radius), frame_id + args.radius + 1):
            row = evidence.get((trip_id, current))
            if row is None:
                continue
            image_path = args.practice_root / trip_id / "kitti" / "image_2" / f"{current:06d}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"could not open {output}")
            classical = _bbox(row.get("classical_selected_bbox_xyxy", ""))
            if classical is not None:
                _draw(image, classical, (255, 0, 255), "classical")
            updates = json.loads(row.get("v2_shadow_updates_json", "[]"))
            target = next((item for item in updates if int(item.get("track_id", -1)) == track_id), None)
            if target and (box := _bbox(target.get("bbox_xyxy", []))) is not None:
                _draw(image, box, (0, 255, 255), f"YOLO track {track_id}")
                caption = f"frame {current}: depth={float(target.get('depth_m', float('nan'))):.1f}m"
            else:
                caption = f"frame {current}: track {track_id} not observed"
            cv2.putText(image, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
        if writer is not None:
            writer.release(); written += 1
    print(json.dumps({"clips": written, "radius_frames": args.radius, "blind_to_ground_truth": True}))


if __name__ == "__main__":
    main()
