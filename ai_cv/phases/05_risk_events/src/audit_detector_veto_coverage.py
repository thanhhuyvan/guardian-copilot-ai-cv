"""Audit detector TTC as an offline veto signal for classical V1 danger."""

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
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.inf
    except (TypeError, ValueError):
        return math.inf


def _truth(trip_path: Path) -> dict[int, float]:
    with gzip.open(trip_path / f"{trip_path.name}.json.gz", "rt", encoding="utf-8") as handle:
        return {int(frame["frame_id"]): float(frame["min_ttc"]) for frame in json.load(handle)["frames"]}


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
                classical_ttc = _number(row["classical_predicted_ttc"])
                if classical_ttc >= 2.0:
                    continue
                label = "tp" if truth[int(row["frame_id"])] < 2.0 else "fp"
                counts[f"classical_danger_{label}"] += 1
                detector_ttc = _number(row["predicted_ttc"])
                if detector_ttc < 2.0:
                    counts[f"detector_supports_{label}"] += 1
                else:
                    counts[f"detector_vetoes_{label}"] += 1
        total.update(counts)
        trips[evidence_path.stem] = dict(counts)
    report = {
        "contract": {
            "classical_danger": "classical TTC < 2s",
            "detector_support": "detector-owned TTC < 2s",
            "purpose": "offline coverage only; no veto policy or threshold chosen",
        },
        "overall": dict(total), "per_trip": trips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
