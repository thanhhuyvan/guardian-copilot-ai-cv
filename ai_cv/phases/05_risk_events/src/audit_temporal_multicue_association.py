"""Audit temporal multi-cue classical-to-YOLO identity proposals offline.

The ranker is deliberately diagnostic-only.  It uses no depth agreement and
never changes TTC, the risk FSM, or a submission.  Its purpose is to reveal
whether image geometry plus immediate track continuity resolves the ambiguity
that makes containment-only association unsafe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from audit_containment_association import _bbox, _contains_centre, _number, _truth
from build_t05_association_audit import _iou


FIELDS = [
    "trip_id", "frame_id", "truth", "classical_bbox_xyxy", "track_id",
    "yolo_bbox_xyxy", "contains_yolo_centre", "normalized_centre_distance",
    "bbox_iou", "continues_previous_frame", "rank", "candidate_count",
]


def _normalised_centre_distance(
    classical: tuple[float, float, float, float], candidate: tuple[float, float, float, float]
) -> float:
    """Centre distance in component-width/component-height units."""
    cx = (classical[0] + classical[2]) / 2.0
    cy = (classical[1] + classical[3]) / 2.0
    tx = (candidate[0] + candidate[2]) / 2.0
    ty = (candidate[1] + candidate[3]) / 2.0
    width, height = max(classical[2] - classical[0], 1.0), max(classical[3] - classical[1], 1.0)
    return math.hypot((tx - cx) / width, (ty - cy) / height)


def _rank_candidates(
    classical: tuple[float, float, float, float], updates: list[dict], previous_track: int | None
) -> list[dict]:
    """Deterministic lexicographic rank; it is not a production association."""
    candidates = []
    for update in updates:
        if update.get("measurement_source") != "yolo_box_median_disparity":
            continue
        bbox = _bbox(update.get("bbox_xyxy", []))
        if bbox is None or "track_id" not in update:
            continue
        track_id = int(update["track_id"])
        contains = _contains_centre(classical, bbox)
        centre_distance = _normalised_centre_distance(classical, bbox)
        iou = _iou(classical, bbox)
        continues = previous_track == track_id
        # Prefer support inside the component, then a continuous identity,
        # then geometric closeness.  Depth is intentionally absent.
        rank_key = (-int(contains), -int(continues), centre_distance, -iou, track_id)
        candidates.append((rank_key, {
            "track_id": track_id, "bbox": bbox, "contains": contains,
            "centre_distance": centre_distance, "iou": iou, "continues": continues,
        }))
    candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in candidates]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    counts = Counter()
    for evidence_path in sorted(args.evidence_root.glob("T*-Sample.csv")):
        truth = _truth(args.practice_root / evidence_path.stem)
        previous_frame, previous_track = None, None
        with evidence_path.open(encoding="utf-8", newline="") as handle:
            for source in csv.DictReader(handle):
                frame_id = int(source["frame_id"])
                classical = _bbox(source["classical_selected_bbox_xyxy"])
                is_danger = source["union_source"].startswith("classical") and _number(source["union_predicted_ttc"]) < 2.0
                if not is_danger or classical is None:
                    previous_frame, previous_track = frame_id, None
                    continue
                updates = json.loads(source.get("v2_shadow_updates_json", "[]"))
                ranked = _rank_candidates(classical, updates, previous_track if previous_frame == frame_id - 1 else None)
                target = "tp" if truth[frame_id] < 2.0 else "fp"
                counts["classical_danger"] += 1
                counts[f"classical_danger_{target}"] += 1
                if ranked:
                    top = ranked[0]
                    counts["ranked"] += 1
                    counts[f"ranked_{target}"] += 1
                    counts["top_contains"] += int(top["contains"])
                    counts[f"top_contains_{target}"] += int(top["contains"])
                    counts["top_continues"] += int(top["continues"])
                    counts[f"top_continues_{target}"] += int(top["continues"])
                    previous_track = top["track_id"]
                else:
                    counts["no_candidate"] += 1
                    counts[f"no_candidate_{target}"] += 1
                    previous_track = None
                previous_frame = frame_id
                for rank, candidate in enumerate(ranked, start=1):
                    rows.append({
                        "trip_id": evidence_path.stem, "frame_id": str(frame_id), "truth": target,
                        "classical_bbox_xyxy": json.dumps(classical), "track_id": str(candidate["track_id"]),
                        "yolo_bbox_xyxy": json.dumps(candidate["bbox"]),
                        "contains_yolo_centre": str(candidate["contains"]),
                        "normalized_centre_distance": f"{candidate['centre_distance']:.6f}",
                        "bbox_iou": f"{candidate['iou']:.6f}",
                        "continues_previous_frame": str(candidate["continues"]),
                        "rank": str(rank), "candidate_count": str(len(ranked)),
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    report = {"contract": "offline candidate ranking only; no TTC/risk/submission change", "overall": dict(counts), "rows": len(rows)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
