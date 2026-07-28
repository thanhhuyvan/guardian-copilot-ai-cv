"""Diagnose whether classical TTC depth motion agrees with YOLO box motion.

This is descriptive only. It uses already-produced evidence and trusted TTC
ground truth; it does not fit a gate or alter any prediction. A rigid object
whose range changes from d0 to d1 should have a roughly inverse-square box
area change. Large residuals indicate a stereo/association disagreement worth
testing as a future causal feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def bbox_iou(first: list[float], second: list[float]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _perception_boxes(path: Path) -> dict[int, list[float]]:
    boxes: dict[int, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        document = json.loads(line)
        objects = document.get("objects", [])
        if objects:
            boxes[int(document["frame_id"])] = [
                float(value) for value in objects[0]["bbox_xyxy"]
            ]
    return boxes


def run(args: argparse.Namespace) -> dict[str, Any]:
    starter_root = args.starter_root.resolve()
    if str(starter_root) not in sys.path:
        sys.path.insert(0, str(starter_root))
    from team_kit.evaluation import load_ground_truth_from_trip, load_predictions

    output_root = args.output_root.resolve()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trip_id in args.trips:
        predictions = load_predictions(
            output_root / "conservative_union" / f"{trip_id}.csv"
        )
        truth = load_ground_truth_from_trip(
            args.practice_root.resolve() / trip_id
        ).ttc
        evidence = {
            int(row["frame_id"]): row
            for row in csv.DictReader(
                (output_root / "evidence" / f"{trip_id}.csv").open(
                    encoding="utf-8", newline=""
                )
            )
        }
        boxes = _perception_boxes(
            output_root / "contracts" / "perception" / f"{trip_id}.jsonl"
        )
        for frame_id, prediction in predictions.items():
            row = evidence[frame_id]
            if row["union_source"] != "classical":
                continue
            predicted_danger = prediction.predicted_ttc < 2.0
            truth_danger = truth.get(frame_id, math.inf) < 2.0
            if not predicted_danger:
                continue
            group = "tp_classical" if truth_danger else "fp_classical"
            track_box = boxes.get(frame_id)
            detections = json.loads(row["raw_detections_json"])
            matched = None
            if track_box is not None and detections:
                matched = max(
                    detections,
                    key=lambda item: bbox_iou(track_box, item["bbox_xyxy"]),
                )
            groups[group].append(
                {
                    "trip_id": trip_id,
                    "frame_id": frame_id,
                    "track_id": row["classical_selected_track_id"],
                    "depth_m": float(row["classical_selected_depth_m"]),
                    "iou": (
                        bbox_iou(track_box, matched["bbox_xyxy"])
                        if matched is not None and track_box is not None
                        else 0.0
                    ),
                    "detection_area": (
                        (matched["bbox_xyxy"][2] - matched["bbox_xyxy"][0])
                        * (matched["bbox_xyxy"][3] - matched["bbox_xyxy"][1])
                        if matched is not None
                        else 0.0
                    ),
                    "matched_class": matched["class_name"] if matched else None,
                    "matched_confidence": (
                        float(matched["confidence"]) if matched else None
                    ),
                }
            )

    report: dict[str, Any] = {"groups": {}}
    for name, rows in groups.items():
        ious = [row["iou"] for row in rows]
        confidences = [
            row["matched_confidence"]
            for row in rows
            if row["matched_confidence"] is not None
        ]
        residuals = []
        previous: dict[tuple[str, str], dict[str, Any]] = {}
        for row in sorted(rows, key=lambda item: (item["trip_id"], item["frame_id"])):
            key = (row["trip_id"], str(row["track_id"]))
            prior = previous.get(key)
            if (
                prior is not None
                and row["frame_id"] == prior["frame_id"] + 1
                and row["detection_area"] > 0.0
                and prior["detection_area"] > 0.0
            ):
                residuals.append(
                    abs(
                        math.log(row["depth_m"] / prior["depth_m"])
                        + 0.5
                        * math.log(
                            row["detection_area"] / prior["detection_area"]
                        )
                    )
                )
            previous[key] = row
        report["groups"][name] = {
            "frames": len(rows),
            "matched_detection_fraction": float(np.mean(np.asarray(ious) > 0.05))
            if ious
            else 0.0,
            "iou": _summary(ious),
            "matched_confidence": _summary(confidences),
            "depth_box_motion_residual": _summary(residuals),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument(
        "--trips",
        nargs="+",
        default=[f"T0{index}-Sample" for index in range(1, 7)],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
