"""Generate an offline, preregistered temporal-regression TTC candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def regression_ttc(observations: list[dict]) -> float | None:
    """Return a weighted linear-depth TTC; None means retain the baseline."""
    if len(observations) < 5:
        return None
    items = sorted(observations, key=lambda item: float(item["timestamp"]))
    time = np.asarray([float(item["timestamp"]) for item in items], dtype=float)
    depth = np.asarray([float(item["depth_m"]) for item in items], dtype=float)
    sigma = np.asarray([float(item.get("depth_sigma_m", 1.0)) for item in items], dtype=float)
    if time[-1] <= time[0] or np.any(~np.isfinite(depth)) or np.any(depth <= 0):
        return None
    sigma = np.where((sigma > 0.0) & np.isfinite(sigma), sigma, 1.0)
    design = np.column_stack((np.ones(len(time)), time - time[0]))
    coefficient = np.linalg.lstsq(design * (1.0 / sigma)[:, None], depth / sigma, rcond=None)[0]
    rate = float(coefficient[1])
    return float(depth[-1] / -rate) if rate < 0.0 and math.isfinite(rate) else math.inf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    counts: Counter[str] = Counter()
    for evidence_path in sorted(args.evidence_root.glob("T*-Sample.csv")):
        evidence = {int(row["frame_id"]): row for row in csv.DictReader(evidence_path.open(encoding="utf-8", newline=""))}
        baseline_path = args.baseline_predictions / evidence_path.name
        rows = list(csv.DictReader(baseline_path.open(encoding="utf-8", newline="")))
        for row in rows:
            source = evidence[int(row["frame_id"])]
            if not source["union_source"].startswith("classical"):
                continue
            candidate = regression_ttc(json.loads(source.get("classical_selected_observations_json", "[]")))
            if candidate is None:
                counts["classical_history_unavailable"] += 1
                continue
            row["predicted_ttc"] = "inf" if math.isinf(candidate) else f"{candidate:.6f}"
            counts["classical_ttc_replaced"] += 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / evidence_path.name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    manifest = {"contract": "offline preregistered candidate; no truth used", "counts": dict(counts)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
