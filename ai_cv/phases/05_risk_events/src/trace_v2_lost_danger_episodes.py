"""Trace pre-registered V2 event-policy loss against trusted TTC labels.

This is diagnostic-only: it never changes EKF, occupancy, or FSM parameters.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


def _number(value: str) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.inf
    except (TypeError, ValueError):
        return math.inf


def _truth(path: Path) -> dict[int, float]:
    with gzip.open(path / f"{path.name}.json.gz", "rt", encoding="utf-8") as handle:
        return {int(item["frame_id"]): float(item["min_ttc"]) for item in json.load(handle)["frames"]}


def _episodes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    active: list[dict[str, object]] = []
    for row in rows + [{}]:
        if row.get("lost"):
            active.append(row)
            continue
        if not active:
            continue
        result.append({
            "start_frame": active[0]["frame_id"],
            "end_frame": active[-1]["frame_id"],
            "lost_frames": len(active),
            "low_occupancy_suppressed_frames": sum(bool(item["low_occupancy_suppressed"]) for item in active),
            "finite_match_frames": sum(float(item["match_iou"]) >= 0.30 for item in active),
            "raw_ttc_min_s": min(float(item["raw_ttc_s"]) for item in active),
            "v2_ttc_min_s": min(float(item["v2_ttc_s"]) for item in active),
        })
        active = []
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "contract": {
            "lost_danger": "trusted TTC < 2s, raw V1 union TTC < 2s, V2 submitted TTC >= 2s",
            "purpose": "diagnose policy loss; do not tune parameters from this report",
        },
        "trips": {},
    }
    for evidence in sorted(args.evidence_root.glob("T*-Sample.csv")):
        truth = _truth(args.practice_root / evidence.stem)
        rows: list[dict[str, object]] = []
        with evidence.open(encoding="utf-8", newline="") as handle:
            for source in csv.DictReader(handle):
                frame_id = int(source["frame_id"])
                raw_ttc, v2_ttc = _number(source["union_predicted_ttc"]), _number(source["v2_event_to_ttc"])
                rows.append({
                    "frame_id": frame_id,
                    "lost": truth[frame_id] < 2.0 and raw_ttc < 2.0 and v2_ttc >= 2.0,
                    "raw_ttc_s": raw_ttc,
                    "v2_ttc_s": v2_ttc,
                    "match_iou": _number(source["v2_event_match_iou"]),
                    "low_occupancy_suppressed": source["v2_event_low_occupancy_suppressed"] == "True",
                })
        episodes = _episodes(rows)
        report["trips"][evidence.stem] = {
            "lost_danger_frames": sum(item["lost_frames"] for item in episodes),
            "episodes": episodes,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
