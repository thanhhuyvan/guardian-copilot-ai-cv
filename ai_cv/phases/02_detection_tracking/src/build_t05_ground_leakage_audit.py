"""Build reproducible visual labels for T05 classical ground-leakage audit.

Ground-truth TTC is used only to stratify offline audit examples.  It never
feeds stereo, ground removal, tracking, or a predicted TTC.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from analyze_stereo_confidence import (  # noqa: E402
    compute_disparities,
    create_left_matcher,
    create_right_matcher,
    left_right_consistency,
    load_calibration,
    read_stereo,
)
from classical_geometry import (  # noqa: E402
    collision_corridor_mask,
    estimate_ground_model,
    extract_obstacle_components,
    ground_and_obstacle_masks,
)


FIELDS = [
    "trip_id", "frame_id", "sample_stratum", "predicted_ttc", "ground_truth_ttc",
    "union_source", "classical_selected_bbox_xyxy", "ground_confidence",
    "classical_selected_depth_m", "raw_detections_json", "component_label", "notes",
]


def _finite(value: str) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.inf
    except (TypeError, ValueError):
        return math.inf


def _truth(trip_dir: Path) -> dict[int, float]:
    with gzip.open(trip_dir / f"{trip_dir.name}.json.gz", "rt", encoding="utf-8") as handle:
        return {int(frame["frame_id"]): float(frame["min_ttc"]) for frame in json.load(handle)["frames"]}


def _draw_bbox(image: np.ndarray, values: str, color: tuple[int, int, int], label: str) -> None:
    try:
        x0, y0, x1, y1 = (round(float(value)) for value in json.loads(values))
    except (ValueError, TypeError, json.JSONDecodeError):
        return
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, label, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def _render_case(
    row: dict[str, str], trip_dir: Path, output_path: Path, left_matcher, right_matcher
) -> None:
    frame_id = int(row["frame_id"])
    left, right = read_stereo(trip_dir, frame_id)
    focal_length_px, baseline_m = load_calibration(trip_dir)
    disparity, right_disparity = compute_disparities(left, right, left_matcher, right_matcher)
    _, consistent, _ = left_right_consistency(disparity, right_disparity)
    ground_model, _ = estimate_ground_model(disparity)
    image = left.copy()
    if ground_model is None:
        cv2.putText(image, "NO GROUND MODEL", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        ground, obstacle, _ = ground_and_obstacle_masks(disparity, ground_model)
        tint = image.copy()
        tint[ground] = (255, 150, 40)  # blue: ground-support pixels
        tint[obstacle] = (30, 30, 255)  # red: closer-than-ground evidence
        image = cv2.addWeighted(image, 0.64, tint, 0.36, 0)
        components, _, corridor = extract_obstacle_components(
            disparity, obstacle, consistent, focal_length_px, baseline_m
        )
        image[corridor & ~ground & ~obstacle] = (
            0.85 * image[corridor & ~ground & ~obstacle] + np.array([25, 25, 25])
        ).astype(np.uint8)
        for component in components:
            x0, y0, x1, y1 = component.bbox
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 180, 0), 1)
        cv2.putText(
            image,
            f"ground q={ground_model.confidence:.2f} residual={ground_model.median_residual_px:.2f}px comps={len(components)}",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA,
        )
    _draw_bbox(image, row["classical_selected_bbox_xyxy"], (255, 0, 255), "selected classical")
    for detection in json.loads(row["raw_detections_json"]):
        x0, y0, x1, y1 = (round(float(value)) for value in detection["bbox_xyxy"])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 1)
    cv2.putText(
        image,
        f"{row['sample_stratum']} frame={frame_id} pred={row['predicted_ttc']}s GT={row['ground_truth_ttc']}s",
        (8, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"could not write {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trip_id = args.evidence.stem
    if trip_id != "T05-Sample":
        raise ValueError("This fixed audit is only for T05-Sample")
    truth = _truth(args.practice_root / trip_id)
    candidates: list[dict[str, str]] = []
    with args.evidence.open(encoding="utf-8", newline="") as handle:
        for evidence in csv.DictReader(handle):
            if _finite(evidence["union_predicted_ttc"]) >= 2.0:
                continue
            frame_id = int(evidence["frame_id"])
            is_false_alert = truth[frame_id] >= 2.0
            candidates.append({
                "trip_id": trip_id,
                "frame_id": str(frame_id),
                "sample_stratum": "v1_false_alert" if is_false_alert else "v1_true_danger_anchor",
                "predicted_ttc": evidence["union_predicted_ttc"],
                "ground_truth_ttc": str(truth[frame_id]),
                "union_source": evidence["union_source"],
                "classical_selected_bbox_xyxy": evidence["classical_selected_bbox_xyxy"],
                "ground_confidence": evidence["ground_confidence"],
                "classical_selected_depth_m": evidence["classical_selected_depth_m"],
                "raw_detections_json": evidence["raw_detections_json"],
                "component_label": "",
                "notes": "",
            })
    false_rows = [row for row in candidates if row["sample_stratum"] == "v1_false_alert"]
    anchor_rows = [row for row in candidates if row["sample_stratum"] == "v1_true_danger_anchor"]
    selected = false_rows + anchor_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "t05_ground_leakage_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    left_matcher, right_matcher = create_left_matcher(), create_right_matcher()
    trip_dir = args.practice_root / trip_id
    for row in selected:
        _render_case(
            row, trip_dir,
            args.output_dir / "overlays" / row["sample_stratum"] / f"{int(row['frame_id']):06d}.jpg",
            left_matcher, right_matcher,
        )
    summary = {
        "contract": "offline diagnosis only; no prediction code or parameters changed",
        "false_alert_frames": len(false_rows),
        "true_danger_anchor_frames": len(anchor_rows),
        "label_values": ["road_leak", "real_object", "mixed", "unknown"],
    }
    (args.output_dir / "audit_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
