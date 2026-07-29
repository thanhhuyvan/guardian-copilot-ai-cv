"""Compare blinded path labels with direct path-only shadow telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


VALID_RELATIONS = {"on_path", "adjacent", "diverging"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.labels.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    with args.shadow.open(encoding="utf-8", newline="") as handle:
        offsets = {
            (row["trip_id"], row["frame_id"], row["track_id"]): row
            for row in csv.DictReader(handle)
        }

    complete: list[tuple[str, float]] = []
    invalid: list[dict[str, str]] = []
    unavailable = 0
    for label in labels:
        relation = label["path_relation"].strip()
        if not relation:
            continue
        if relation not in VALID_RELATIONS:
            invalid.append(label)
            continue
        key = (label["trip_id"], label["frame_id"], label["track_id"])
        offset = offsets.get(key)
        if offset is None or offset["available"] != "true":
            unavailable += 1
            continue
        complete.append((relation, abs(float(offset["path_offset_m"]))))

    by_relation: dict[str, list[float]] = {relation: [] for relation in VALID_RELATIONS}
    for relation, offset in complete:
        by_relation[relation].append(offset)
    means = {
        relation: sum(values) / len(values)
        for relation, values in by_relation.items() if values
    }
    non_path = by_relation["adjacent"] + by_relation["diverging"]
    direction_consistent = bool(by_relation["on_path"] and non_path) and (
        means["on_path"] < sum(non_path) / len(non_path)
    )
    report = {
        "total_review_rows": len(labels),
        "completed_label_rows": sum(bool(row["path_relation"].strip()) for row in labels),
        "valid_geometry_label_rows": len(complete),
        "unavailable_geometry_labeled_rows": unavailable,
        "invalid_label_rows": len(invalid),
        "path_relation_counts": dict(Counter(relation for relation, _ in complete)),
        "mean_absolute_path_offset_m_by_relation": means,
        "direction_consistent_without_tuned_cutoff": direction_consistent,
        "decision": (
            "review_complete_consider_one_risk_gate_experiment"
            if len(complete) >= 24 and not invalid and direction_consistent
            else "labels_or_geometry_insufficient_no_scored_pipeline_change"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
