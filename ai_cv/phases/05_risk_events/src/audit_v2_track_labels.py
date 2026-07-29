"""Audit human V2 path/CPA labels without choosing a deployment threshold."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


VALID_PATH_RELATIONS = {"on_path", "adjacent", "diverging"}
VALID_OCCLUSION = {"yes", "no", "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.labels.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    complete = [
        row for row in rows
        if row["path_relation"] in VALID_PATH_RELATIONS
        and row["occluded"] in VALID_OCCLUSION
        and row["cpa_distance_m"].strip()
    ]
    invalid = [
        row for row in rows
        if row["path_relation"] and row["path_relation"] not in VALID_PATH_RELATIONS
    ]
    by_relation: dict[str, list[float]] = {}
    for row in complete:
        by_relation.setdefault(row["path_relation"], []).append(
            float(row["occupancy_probability"])
        )
    report = {
        "total_rows": len(rows),
        "complete_rows": len(complete),
        "incomplete_rows": len(rows) - len(complete),
        "invalid_path_relation_rows": len(invalid),
        "trip_counts": dict(Counter(row["trip_id"] for row in complete)),
        "path_relation_counts": dict(Counter(row["path_relation"] for row in complete)),
        "mean_occupancy_by_path_relation": {
            relation: sum(values) / len(values) for relation, values in by_relation.items()
        },
        "decision": (
            "ready_for_track_level_review" if len(complete) >= 30 and not invalid
            else "labels_incomplete_no_risk_gate_decision"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
