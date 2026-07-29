"""Describe raw classical-track depth-rate stability by organizer event truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_containment_association import _number, _truth


def _metrics(observations: list[dict]) -> dict[str, float] | None:
    if len(observations) < 3:
        return None
    ordered = sorted(observations, key=lambda item: float(item["timestamp"]))
    time = np.asarray([float(item["timestamp"]) for item in ordered], dtype=float)
    depth = np.asarray([float(item["depth_m"]) for item in ordered], dtype=float)
    centre = np.asarray([float(item["center_x"]) for item in ordered], dtype=float)
    if time[-1] <= time[0] or np.any(~np.isfinite(depth)) or np.any(depth <= 0):
        return None
    design = np.column_stack((np.ones(len(time)), time - time[0]))
    slope = float(np.linalg.lstsq(design, depth, rcond=None)[0][1])
    residual = depth - design @ np.linalg.lstsq(design, depth, rcond=None)[0]
    return {
        "raw_depth_rate_mps": float((depth[-1] - depth[-2]) / (time[-1] - time[-2])),
        "linear_depth_rate_mps": slope,
        "depth_linear_residual_mad_m": float(np.median(np.abs(residual - np.median(residual)))),
        "tail_centre_step_px": float(abs(centre[-1] - centre[-2])),
        "max_centre_step_px": float(np.max(np.abs(np.diff(centre)))),
        "linear_ttc_s": float(depth[-1] / -slope) if slope < 0 else math.inf,
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([item for item in values if math.isfinite(item)], dtype=float)
    if not len(finite):
        return {"count": 0, "p50": None, "p90": None, "p95": None}
    return {"count": int(len(finite)), **{f"p{p}": float(np.percentile(finite, p)) for p in (50, 90, 95)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(args.evidence_root.glob("T*-Sample.csv")):
        truth = _truth(args.practice_root / path.stem)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["union_source"].startswith("classical") or _number(row["union_predicted_ttc"]) >= 2.0:
                    continue
                result = _metrics(json.loads(row.get("classical_selected_observations_json", "[]")))
                if result is None:
                    continue
                group = "true_danger" if truth[int(row["frame_id"])] < 2.0 else "false_alert"
                for name, value in result.items():
                    groups[group][name].append(value)
    report = {
        "contract": "descriptive only; no new TTC, threshold, filter, or F1 run",
        "groups": {group: {name: _summary(values) for name, values in metrics.items()} for group, metrics in groups.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
