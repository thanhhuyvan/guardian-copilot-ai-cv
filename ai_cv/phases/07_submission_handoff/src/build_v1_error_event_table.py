"""Build organizer-style FP/FN event episodes for the frozen V1 practice run."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


TRIPS = tuple(f"T{index:02d}-Sample" for index in range(1, 7))


def finite(value: str) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.inf
    except ValueError:
        return math.inf


def runs(frame_ids: list[int], gap: int = 2) -> list[list[int]]:
    output: list[list[int]] = []
    for frame_id in frame_ids:
        if not output or frame_id > output[-1][-1] + gap:
            output.append([frame_id])
        else:
            output[-1].append(frame_id)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--starter-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if str(args.starter_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.starter_root.resolve()))
    from team_kit.dataset_loader import TripDataset

    episodes: list[dict[str, object]] = []
    for trip_id in TRIPS:
        with (args.output_root / "conservative_union" / f"{trip_id}.csv").open(encoding="utf-8", newline="") as handle:
            predictions = {int(row["frame_id"]): finite(row["predicted_ttc"]) for row in csv.DictReader(handle)}
        with (args.output_root / "evidence" / f"{trip_id}.csv").open(encoding="utf-8", newline="") as handle:
            evidence = {int(row["frame_id"]): row for row in csv.DictReader(handle)}
        truth = {int(frame.frame_id): float(frame.min_ttc) for frame in TripDataset(args.practice_root / trip_id).iter_frames()}
        for kind, condition in {
            "FP": lambda frame_id: predictions[frame_id] < 2.0 and truth[frame_id] >= 2.0,
            "FN": lambda frame_id: predictions[frame_id] >= 2.0 and truth[frame_id] < 2.0,
        }.items():
            for episode in runs([frame_id for frame_id in sorted(predictions) if condition(frame_id)]):
                representative = min(episode, key=lambda frame_id: predictions[frame_id] if kind == "FP" else truth[frame_id])
                sources = Counter(evidence[frame_id].get("union_source", "unknown") for frame_id in episode)
                episodes.append({
                    "trip_id": trip_id,
                    "error_type": kind,
                    "start_frame": episode[0],
                    "end_frame": episode[-1],
                    "error_frames": len(episode),
                    "duration_s": round(len(episode) / 20.0, 2),
                    "representative_frame": representative,
                    "predicted_ttc_s": "inf" if not math.isfinite(predictions[representative]) else round(predictions[representative], 3),
                    "true_ttc_s": "inf" if not math.isfinite(truth[representative]) else round(truth[representative], 3),
                    "dominant_source": sources.most_common(1)[0][0],
                    "selected_track_id": evidence[representative].get("union_selected_track_id", ""),
                })
    episodes.sort(key=lambda row: (-int(row["error_frames"]), str(row["trip_id"]), int(row["start_frame"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(episodes[0])
    with (args.output_dir / "v1_error_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(episodes)
    lines = ["# V1 organizer-style error-event table", "", "Each row is a contiguous practice-set FP/FN episode; `representative_frame` is the frame with lowest predicted TTC (FP) or lowest true TTC (FN).", "", "| Trip | Type | Frames | Duration | Representative | Pred TTC | True TTC | Source |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in episodes:
        lines.append(f"| {row['trip_id']} | {row['error_type']} | {row['start_frame']}–{row['end_frame']} ({row['error_frames']}) | {row['duration_s']} s | {row['representative_frame']} | {row['predicted_ttc_s']} | {row['true_ttc_s']} | {row['dominant_source']} |")
    (args.output_dir / "V1_ERROR_EVENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(episodes)} error episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
