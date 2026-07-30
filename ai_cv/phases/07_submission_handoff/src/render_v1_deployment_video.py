"""Render a causal V1 deployment video without accessing redacted TTC truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_ttc(value: str) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.inf
    except ValueError:
        return math.inf


def label_ttc(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.2f} s"


def load_predictions(path: Path) -> dict[int, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(row["frame_id"]): parse_ttc(row["predicted_ttc"]) for row in csv.DictReader(handle)}


def load_boxes(path: Path) -> dict[int, list[float]]:
    boxes: dict[int, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        document = json.loads(line)
        if document.get("objects"):
            boxes[int(document["frame_id"])] = [float(v) for v in document["objects"][0]["bbox_xyxy"]]
    return boxes


def run(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    predictions = load_predictions(output_root / "conservative_union" / f"{args.trip}.csv")
    with (output_root / "evidence" / f"{args.trip}.csv").open(encoding="utf-8", newline="") as handle:
        evidence = {int(row["frame_id"]): row for row in csv.DictReader(handle)}
    boxes = load_boxes(output_root / "contracts" / "perception" / f"{args.trip}.jsonl")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 360))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open {output}")
    try:
        for frame_id in range(args.start_frame, args.end_frame + 1):
            row = evidence[frame_id]
            degraded = row.get("input_degraded", "False").strip().lower() == "true"
            image_path = args.data_root / args.trip / "kitti" / "image_2" / f"{frame_id:06d}.jpg"
            image = None if degraded else cv2.imread(str(image_path))
            if image is None:
                image = np.zeros((360, 640, 3), dtype=np.uint8)
                degraded = True
            ttc = predictions[frame_id]
            danger = ttc < 2.0
            color = (0, 0, 235) if danger else (50, 220, 70)
            cv2.rectangle(image, (0, 0), (639, 68), (0, 0, 0), -1)
            cv2.putText(image, f"Guardian V1  {args.trip}  frame {frame_id}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(image, f"{'DANGER' if danger else 'CLEAR'} | predicted TTC {label_ttc(ttc)} | {row.get('union_source', 'unknown')}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1)
            for detection in json.loads(row.get("raw_detections_json", "[]")):
                x0, y0, x1, y1 = (int(v) for v in detection["bbox_xyxy"])
                cv2.rectangle(image, (x0, y0), (x1, y1), (255, 180, 0), 1)
                cv2.putText(image, f"{detection['class_name']} {detection['confidence']:.2f}", (x0, max(84, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 180, 0), 1)
            if frame_id in boxes:
                x0, y0, x1, y1 = (int(v) for v in boxes[frame_id])
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
                cv2.putText(image, "selected TTC track", (x0, min(350, y1 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if degraded:
                cv2.rectangle(image, (0, 330), (639, 359), (0, 0, 180), -1)
                cv2.putText(image, "DEGRADED INPUT: fail-safe inference; trackers reset", (10, 351), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            writer.write(image)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--trip", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.end_frame < args.start_frame:
        parser.error("end frame must be >= start frame")
    run(args)


if __name__ == "__main__":
    main()
