"""Create a label file that hides all model outputs from the path reviewer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["trip_id", "frame_id", "left_image_path", "track_id", "path_relation", "occluded", "notes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.source_labels.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = [{field: row.get(field, "") for field in FIELDS} for row in source]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} blinded rows to {args.output}")


if __name__ == "__main__":
    main()
