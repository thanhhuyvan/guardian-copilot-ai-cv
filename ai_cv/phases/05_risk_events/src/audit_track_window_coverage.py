"""Measure whether a selected YOLO track exists across its local review window."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _visible_track_ids(row: dict[str, str]) -> set[int]:
    return {
        int(item["track_id"])
        for item in json.loads(row.get("v2_shadow_updates_json", "[]"))
        if "track_id" in item
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence: dict[tuple[str, int], dict[str, str]] = {}
    for path in args.evidence_root.glob("T*-Sample.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            evidence.update({(path.stem, int(row["frame_id"])): row for row in csv.DictReader(handle)})
    with args.labels.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))

    by_trip: dict[str, list[float]] = defaultdict(list)
    histogram: Counter[str] = Counter()
    for label in labels:
        trip_id, frame_id, track_id = label["trip_id"], int(label["frame_id"]), int(label["track_id"])
        window = range(max(0, frame_id - args.radius), frame_id + args.radius + 1)
        available = sum(track_id in _visible_track_ids(evidence.get((trip_id, current), {})) for current in window)
        possible = len(window)
        fraction = available / possible
        by_trip[trip_id].append(fraction)
        histogram[f"{available}/{possible}"] += 1
    result = {
        "contract": "descriptive track-lifecycle audit; no risk/TTC decision",
        "radius_frames": args.radius,
        "tracks": len(labels),
        "mean_visible_fraction": sum(sum(values) for values in by_trip.values()) / len(labels),
        "single_frame_tracks": histogram.get(f"1/{2 * args.radius + 1}", 0),
        "coverage_histogram": dict(sorted(histogram.items())),
        "mean_visible_fraction_by_trip": {trip: sum(values) / len(values) for trip, values in sorted(by_trip.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
