"""Render camera video with TTC, fusion, track and YOLO evidence overlays."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def _text(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.2f}s"


def _load_track_boxes(path: Path) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        document = json.loads(line)
        if document.get("objects"):
            result[int(document["frame_id"])] = [
                float(value) for value in document["objects"][0]["bbox_xyxy"]
            ]
    return result


def run(args: argparse.Namespace) -> None:
    starter_root = args.starter_root.resolve()
    if str(starter_root) not in sys.path:
        sys.path.insert(0, str(starter_root))
    from team_kit.evaluation import load_ground_truth_from_trip, load_predictions

    output_root = args.output_root.resolve()
    predictions = load_predictions(
        output_root / "conservative_union" / f"{args.trip}.csv"
    )
    truth = load_ground_truth_from_trip(args.practice_root.resolve() / args.trip).ttc
    evidence = {
        int(row["frame_id"]): row
        for row in csv.DictReader(
            (output_root / "evidence" / f"{args.trip}.csv").open(
                encoding="utf-8", newline=""
            )
        )
    }
    boxes = _load_track_boxes(
        output_root / "contracts" / "perception" / f"{args.trip}.jsonl"
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 360)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output}")
    try:
        for frame_id in range(args.start_frame, args.end_frame + 1):
            image_path = (
                args.practice_root.resolve()
                / args.trip
                / "kitti"
                / "image_2"
                / f"{frame_id:06d}.jpg"
            )
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"cannot read {image_path}")
            prediction = predictions[frame_id].predicted_ttc
            actual = truth.get(frame_id, math.inf)
            row = evidence[frame_id]
            predicted_danger = prediction < 2.0
            actual_danger = actual < 2.0
            status = (
                "TP" if predicted_danger and actual_danger else
                "FP" if predicted_danger else
                "FN" if actual_danger else "TN"
            )
            color = {
                "TP": (0, 200, 0), "FP": (0, 0, 230),
                "FN": (0, 180, 255), "TN": (220, 220, 220),
            }[status]
            cv2.rectangle(image, (0, 0), (639, 66), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"{args.trip}  frame {frame_id}  {status}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            cv2.putText(
                image,
                f"pred {_text(prediction)} | GT {_text(actual)} | {row.get('union_source', 'conservative_union')}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            for detection in json.loads(row.get("raw_detections_json", "[]")):
                x0, y0, x1, y1 = (int(value) for value in detection["bbox_xyxy"])
                cv2.rectangle(image, (x0, y0), (x1, y1), (255, 180, 0), 1)
                cv2.putText(
                    image,
                    f"{detection['class_name']} {detection['confidence']:.2f}",
                    (x0, max(78, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 180, 0), 1,
                )
            if frame_id in boxes:
                x0, y0, x1, y1 = (int(value) for value in boxes[frame_id])
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    image, "selected TTC track", (x0, min(350, y1 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )
            writer.write(image)
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--starter-root", type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument("--trip", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.end_frame < args.start_frame:
        parser.error("end frame must be >= start frame")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
