"""Build deterministic V2 track snapshots for human path/CPA validation."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


FIELDS = [
    "trip_id", "frame_id", "left_image_path", "track_id",
    "mahalanobis_squared", "occupancy_probability", "path_relation",
    "cpa_distance_m", "occluded", "ground_truth_ttc", "sample_stratum", "notes",
]


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Choose full-trip temporal coverage instead of early-frame samples."""
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return rows
    indices = {round(index * (len(rows) - 1) / (count - 1)) for index in range(count)}
    return [row for index, row in enumerate(rows) if index in indices]


def ground_truth_ttc_by_frame(trip_path: Path) -> dict[int, float]:
    """Read organizer labels only to choose validation anchors, never runtime."""
    with gzip.open(trip_path / f"{trip_path.name}.json.gz", "rt", encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]
    return {int(frame["frame_id"]): float(frame["min_ttc"]) for frame in frames}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-trip", type=int, default=5)
    args = parser.parse_args()

    output_rows: list[dict[str, str]] = []
    for trip_file in sorted(args.evidence_root.glob("T*-Sample.csv")):
        trip_id = trip_file.stem
        ground_truth = ground_truth_ttc_by_frame(args.practice_root / trip_id)
        candidates: list[dict[str, str]] = []
        with trip_file.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                updates = json.loads(row.get("v2_shadow_updates_json", "[]"))
                if not updates:
                    continue
                selected_track_id = row.get("selected_track_id", "")
                update = next(
                    (
                        item for item in updates
                        if str(item.get("track_id", "")) == selected_track_id
                        and item.get("measurement_source") == "yolo_box_median_disparity"
                    ),
                    None,
                )
                if update is None:
                    # Known-failure frames may lack a detector TTC candidate;
                    # retain an object-level track rather than classical noise.
                    detector_updates = [
                        item for item in updates
                        if item.get("measurement_source") == "yolo_box_median_disparity"
                    ]
                    if not detector_updates:
                        continue
                    update = max(
                        detector_updates,
                        key=lambda item: item.get("depth_confidence", 0.0),
                    )
                frame_id = int(row["frame_id"])
                candidates.append(
                    {
                        "trip_id": trip_id,
                        "frame_id": str(frame_id),
                        "left_image_path": str(
                            args.practice_root / trip_file.stem / "kitti" / "image_2"
                            / f"{frame_id:06d}.jpg"
                        ),
                        "track_id": str(update["track_id"]),
                        "mahalanobis_squared": str(update["mahalanobis_squared"]),
                        "occupancy_probability": str(
                            update.get("corridor_occupancy_probability", "")
                        ),
                        "path_relation": "",
                        "cpa_distance_m": "",
                        "occluded": "",
                        "ground_truth_ttc": str(ground_truth.get(frame_id, math.inf)),
                        "sample_stratum": "known_failure" if trip_id in {"T01-Sample", "T05-Sample"} else "",
                        "notes": "",
                    }
                )
        # T01/T05 retain full temporal coverage of known failures.  Other
        # trips supply true-danger anchors (organizer TTC <2 s); fill only if
        # the trip has fewer such frames.  GT never enters filter or gate.
        if trip_id not in {"T01-Sample", "T05-Sample"}:
            anchors = [row for row in candidates if float(row["ground_truth_ttc"]) < 2.0]
            chosen = evenly_spaced(anchors, args.per_trip)
            for row in chosen:
                row["sample_stratum"] = "ground_truth_danger_anchor"
            if len(chosen) < args.per_trip:
                chosen_ids = {(row["frame_id"], row["track_id"]) for row in chosen}
                fill = [row for row in candidates if (row["frame_id"], row["track_id"]) not in chosen_ids]
                chosen.extend(evenly_spaced(fill, args.per_trip - len(chosen)))
        else:
            chosen = evenly_spaced(candidates, args.per_trip)
        output_rows.extend(chosen)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
