"""Create offline labels for T05 classical-to-YOLO association failures.

This ranks candidates with independent geometric cues but deliberately makes no
match decision.  Human labels establish whether multi-cue association would be
safe before any TTC policy can consume it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import cv2


FIELDS = [
    "trip_id", "frame_id", "sample_stratum", "classical_bbox_xyxy", "classical_depth_m",
    "proposed_track_id", "proposed_bbox_xyxy", "proposed_depth_m", "bbox_iou",
    "centre_distance_px", "depth_delta_m", "rank", "same_object", "notes",
]


def _number(value: str) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.inf
    except (TypeError, ValueError):
        return math.inf


def _truth(trip_dir: Path) -> dict[int, float]:
    with gzip.open(trip_dir / f"{trip_dir.name}.json.gz", "rt", encoding="utf-8") as handle:
        return {int(item["frame_id"]): float(item["min_ttc"]) for item in json.load(handle)["frames"]}


def _bbox(value: str | list[float]) -> tuple[float, float, float, float] | None:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
        if len(raw) != 4:
            return None
        return tuple(float(item) for item in raw)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - overlap
    return overlap / union if union > 0.0 else 0.0


def _centre_distance(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    first_centre = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_centre = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    return math.dist(first_centre, second_centre)


def _candidate_rows(row: dict[str, str], stratum: str) -> list[dict[str, str]]:
    classical_bbox = _bbox(row["classical_selected_bbox_xyxy"])
    if classical_bbox is None:
        return []
    classical_depth = _number(row["classical_selected_depth_m"])
    candidates = []
    for update in json.loads(row.get("v2_shadow_updates_json", "[]")):
        if update.get("measurement_source") != "yolo_box_median_disparity":
            continue
        proposal_bbox = _bbox(update.get("bbox_xyxy", []))
        if proposal_bbox is None:
            continue
        depth = float(update.get("depth_m", math.inf))
        iou = _iou(classical_bbox, proposal_bbox)
        centre_distance = _centre_distance(classical_bbox, proposal_bbox)
        # Ranking only.  No threshold or learned score is introduced here.
        rank_key = (-iou, centre_distance, abs(classical_depth - depth))
        candidates.append((rank_key, {
            "trip_id": "T05-Sample", "frame_id": row["frame_id"], "sample_stratum": stratum,
            "classical_bbox_xyxy": json.dumps(classical_bbox), "classical_depth_m": str(classical_depth),
            "proposed_track_id": str(update["track_id"]), "proposed_bbox_xyxy": json.dumps(proposal_bbox),
            "proposed_depth_m": str(depth), "bbox_iou": f"{iou:.6f}",
            "centre_distance_px": f"{centre_distance:.3f}", "depth_delta_m": f"{abs(classical_depth - depth):.3f}",
            "rank": "", "same_object": "", "notes": "",
        }))
    candidates.sort(key=lambda item: item[0])
    for rank, (_, candidate) in enumerate(candidates, start=1):
        candidate["rank"] = str(rank)
    return [candidate for _, candidate in candidates]


def _draw(image, bbox, color, label):
    x0, y0, x1, y1 = (round(value) for value in bbox)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, label, (x0, max(16, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)


def _render(rows: list[dict[str, str]], practice_root: Path, output_dir: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["sample_stratum"], row["frame_id"]), []).append(row)
    for (stratum, frame_text), candidates in grouped.items():
        frame_id = int(frame_text)
        image_path = practice_root / "T05-Sample" / "kitti" / "image_2" / f"{frame_id:06d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        _draw(image, _bbox(candidates[0]["classical_bbox_xyxy"]), (255, 0, 255), "classical")
        for candidate in candidates[:3]:
            _draw(
                image, _bbox(candidate["proposed_bbox_xyxy"]), (0, 255, 255),
                f"YOLO #{candidate['rank']} iou={candidate['bbox_iou']} d={candidate['depth_delta_m']}m",
            )
        output = output_dir / "overlays" / stratum / f"{frame_id:06d}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    truth = _truth(args.practice_root / "T05-Sample")
    audit_rows: list[dict[str, str]] = []
    skipped_no_yolo = 0
    with args.evidence.open(encoding="utf-8", newline="") as handle:
        for evidence in csv.DictReader(handle):
            frame_id = int(evidence["frame_id"])
            if not evidence["union_source"].startswith("classical") or _number(evidence["union_predicted_ttc"]) >= 2.0:
                continue
            if _number(evidence["v2_event_match_iou"]) >= 0.30:
                continue
            stratum = "false_alert_unmatched" if truth[frame_id] >= 2.0 else "true_danger_unmatched"
            candidates = _candidate_rows(evidence, stratum)
            if not candidates:
                skipped_no_yolo += 1
            audit_rows.extend(candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "t05_association_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(audit_rows)
    _render(audit_rows, args.practice_root, args.output_dir)
    report = {
        "contract": "offline association audit; no match decision, TTC, or model parameter changed",
        "frames_with_ranked_yolo_candidates": len({row["frame_id"] for row in audit_rows}),
        "candidate_pairs": len(audit_rows),
        "frames_without_yolo_candidate": skipped_no_yolo,
        "label_values": ["yes", "no", "uncertain"],
    }
    (args.output_dir / "audit_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
