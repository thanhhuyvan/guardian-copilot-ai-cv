"""Measure causal classical-track residuals before selecting V2 EKF noise."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def residual_sigmas(observations: list[dict[str, float]], focal_px: float, cx_px: float) -> tuple[float, float] | None:
    if len(observations) < 4:
        return None
    time = np.asarray([item["timestamp"] for item in observations], dtype=float)
    depth = np.asarray([item["depth_m"] for item in observations], dtype=float)
    center = np.asarray([item["center_x"] for item in observations], dtype=float)
    if time[-1] <= time[0] or np.any(depth <= 0.0):
        return None
    lateral = (center - cx_px) * depth / focal_px
    design = np.column_stack([np.ones(len(time)), time - time[0]])
    depth_residual = depth - design @ np.linalg.lstsq(design, depth, rcond=None)[0]
    lateral_residual = lateral - design @ np.linalg.lstsq(design, lateral, rcond=None)[0]
    # Robust Gaussian sigma from residual MAD; this is a measurement audit,
    # not an F1-optimized filter parameter.
    return tuple(
        float(1.4826 * np.median(np.abs(values - np.median(values))))
        for values in (depth_residual, lateral_residual)
    )


def audit(evidence_path: Path, focal_px: float, cx_px: float) -> dict[str, object]:
    rows = list(csv.DictReader(evidence_path.open(encoding="utf-8", newline="")))
    histories: dict[str, dict[float, dict[str, float]]] = {}
    for row in rows:
        shadow_tracks = json.loads(
            row.get("classical_risk_track_measurements_json", "[]")
        )
        for item in shadow_tracks:
            histories.setdefault(str(item["track_id"]), {})[
                float(item["timestamp"])
            ] = item
        track_id = row.get("classical_selected_track_id", "")
        if shadow_tracks or not track_id:
            continue
        for item in json.loads(row["classical_selected_observations_json"]):
            histories.setdefault(track_id, {})[float(item["timestamp"])] = item
    sigmas = [
        value
        for history in histories.values()
        if (value := residual_sigmas(list(sorted(history.values(), key=lambda x: x["timestamp"])), focal_px, cx_px))
        is not None
    ]
    depth = np.asarray([item[0] for item in sigmas], dtype=float)
    lateral = np.asarray([item[1] for item in sigmas], dtype=float)
    summary = {
        "tracks_with_history": len(histories),
        "tracks_with_residual_estimate": len(sigmas),
        "depth_residual_sigma_m": _summary(depth),
        "lateral_residual_sigma_m": _summary(lateral),
        "decision": "measurements_only_do_not_promote_filter_from_this_report",
    }
    return summary


def _summary(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"p50": None, "p90": None, "p95": None}
    return {key: float(np.percentile(values, percentile)) for key, percentile in (("p50", 50), ("p90", 90), ("p95", 95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--focal-px", type=float, required=True)
    parser.add_argument("--principal-x-px", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.evidence, args.focal_px, args.principal_x_px)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
