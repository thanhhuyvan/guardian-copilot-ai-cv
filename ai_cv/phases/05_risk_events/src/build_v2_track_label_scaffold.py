"""Build deterministic V2 track snapshots for human path/CPA validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "trip_id", "frame_id", "left_image_path", "track_id",
    "mahalanobis_squared", "occupancy_probability", "path_relation",
    "cpa_distance_m", "occluded", "notes",
]


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Choose full-trip temporal coverage instead of early-frame samples."""
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return rows
    indices = {round(index * (len(rows) - 1) / (count - 1)) for index in range(count)}
    return [row for index, row in enumerate(rows) if index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-trip", type=int, default=5)
    args = parser.parse_args()

    output_rows: list[dict[str, str]] = []
    for trip_file in sorted(args.evidence_root.glob("T*-Sample.csv")):
        candidates: list[dict[str, str]] = []
        with trip_file.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                updates = json.loads(row.get("v2_shadow_updates_json", "[]"))
                if not updates:
                    continue
                update = max(updates, key=lambda item: item.get("mahalanobis_squared", 0.0))
                frame_id = int(row["frame_id"])
                candidates.append(
                    {
                        "trip_id": trip_file.stem,
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
                        "notes": "",
                    }
                )
        output_rows.extend(evenly_spaced(candidates, args.per_trip))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
