"""Measure conservative semantic-containment association coverage offline.

An association is proposed only when exactly one YOLO stereo-track centre lies
inside a classical component box.  This has no TTC effect and introduces no
learned score or tuned distance threshold.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path


def _number(value: str) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.inf
    except (TypeError, ValueError):
        return math.inf


def _truth(trip_path: Path) -> dict[int, float]:
    with gzip.open(trip_path / f"{trip_path.name}.json.gz", "rt", encoding="utf-8") as handle:
        return {int(frame["frame_id"]): float(frame["min_ttc"]) for frame in json.load(handle)["frames"]}


def _bbox(value: str | list[float]) -> tuple[float, float, float, float] | None:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
        return tuple(float(item) for item in raw) if len(raw) == 4 else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _contains_centre(container: tuple[float, float, float, float], candidate: tuple[float, float, float, float]) -> bool:
    x, y = (candidate[0] + candidate[2]) / 2.0, (candidate[1] + candidate[3]) / 2.0
    return container[0] <= x <= container[2] and container[1] <= y <= container[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    total, trips = Counter(), {}
    for evidence_path in sorted(args.evidence_root.glob("T*-Sample.csv")):
        truth, counts = _truth(args.practice_root / evidence_path.stem), Counter()
        with evidence_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["union_source"].startswith("classical") or _number(row["union_predicted_ttc"]) >= 2.0:
                    continue
                classical_box = _bbox(row["classical_selected_bbox_xyxy"])
                if classical_box is None:
                    continue
                counts["classical_danger"] += 1
                key = "tp" if truth[int(row["frame_id"])] < 2.0 else "fp"
                updates = json.loads(row.get("v2_shadow_updates_json", "[]"))
                contained = [
                    update for update in updates
                    if update.get("measurement_source") == "yolo_box_median_disparity"
                    and (box := _bbox(update.get("bbox_xyxy", []))) is not None
                    and _contains_centre(classical_box, box)
                ]
                if len(contained) == 1:
                    counts["containment_unique"] += 1
                    counts[f"containment_unique_{key}"] += 1
                elif len(contained) > 1:
                    counts["containment_ambiguous"] += 1
                    counts[f"containment_ambiguous_{key}"] += 1
                else:
                    counts["containment_none"] += 1
                    counts[f"containment_none_{key}"] += 1
        total.update(counts)
        trips[evidence_path.stem] = dict(counts)
    report = {
        "contract": {
            "proposal": "exactly one YOLO stereo-track box centre lies inside classical component box",
            "purpose": "coverage audit only; no risk decision, F1 run, or parameter tuning",
        },
        "overall": dict(total), "per_trip": trips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
