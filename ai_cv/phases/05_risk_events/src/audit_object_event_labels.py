"""Validate blind object-event labels before any state-model experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


VALID = {
    "event_owner": {"yes", "no", "uncertain"},
    "path_relation": {"on_path", "adjacent", "crossing", "diverging", "uncertain"},
    "relative_motion": {"closing", "steady", "opening", "uncertain"},
    "occluded": {"yes", "no", "unknown"},
    "review_confidence": {"high", "medium", "low"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.labels.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    invalid: dict[str, list[int]] = {field: [] for field in VALID}
    complete = []
    provisional_lines: list[int] = []
    for index, row in enumerate(rows, start=2):
        is_complete = True
        for field, values in VALID.items():
            value = row.get(field, "").strip()
            if value not in values:
                invalid[field].append(index)
                is_complete = False
        cpa = row.get("cpa_distance_m", "").strip()
        if not (cpa == "unknown" or (cpa and _positive_number(cpa))):
            invalid.setdefault("cpa_distance_m", []).append(index)
            is_complete = False
        if is_complete:
            complete.append(row)
        if "PROVISIONAL_AI_ASSUMPTION" in row.get("notes", ""):
            provisional_lines.append(index)
    high_or_medium = sum(row["review_confidence"] in {"high", "medium"} for row in complete)
    has_reviewable_cpa = any(row["cpa_distance_m"] != "unknown" for row in complete)
    ready = (
        len(complete) >= 30
        and not any(invalid.values())
        and not provisional_lines
        and high_or_medium >= 30
        and has_reviewable_cpa
    )
    report = {
        "total_rows": len(rows),
        "complete_rows": len(complete),
        "invalid_rows_by_field": {field: lines for field, lines in invalid.items() if lines},
        "trip_counts": dict(Counter(row["trip_id"] for row in complete)),
        "event_owner_counts": dict(Counter(row["event_owner"] for row in complete)),
        "path_relation_counts": dict(Counter(row["path_relation"] for row in complete)),
        "motion_counts": dict(Counter(row["relative_motion"] for row in complete)),
        "high_or_medium_confidence_rows": high_or_medium,
        "provisional_ai_assumption_lines": provisional_lines,
        "reviewable_cpa_rows": sum(row["cpa_distance_m"] != "unknown" for row in complete),
        "decision": (
            "ready_for_state_validation"
            if ready
            else "labels_incomplete_no_state_or_f1_experiment"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _positive_number(value: str) -> bool:
    try:
        return float(value) > 0.0
    except ValueError:
        return False


if __name__ == "__main__":
    main()
