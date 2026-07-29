"""Score human-verified T05 association labels without touching TTC policy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.labels.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    complete = [row for row in rows if row["same_object"] in {"yes", "no", "uncertain"}]
    labels = Counter(row["same_object"] for row in complete)
    top_one = [row for row in complete if row["rank"] == "1"]
    top_one_labels = Counter(row["same_object"] for row in top_one)
    yes_no = top_one_labels["yes"] + top_one_labels["no"]
    report = {
        "contract": "labels validate association only; never train or tune TTC policy",
        "candidate_pairs": len(rows),
        "complete_pairs": len(complete),
        "label_counts": dict(labels),
        "top_one_label_counts": dict(top_one_labels),
        "top_one_precision_yes_over_yes_no": (
            top_one_labels["yes"] / yes_no if yes_no else None
        ),
        "decision": (
            "incomplete_labels_no_association_decision"
            if len(complete) < len(rows)
            else "review_top_one_precision_and_true_danger_errors"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
