"""Compare robust YOLO-box range change for on-path false and true danger."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE02_SRC = REPOSITORY_ROOT / "ai_cv" / "phases" / "02_detection_tracking" / "src"
if str(PHASE02_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE02_SRC))

from evaluate_path_only_gate import (  # noqa: E402
    best_yolo_update,
    direct_path_offset_m,
    load_truth,
    parse_ttc,
)


def theil_sen_depth_slope(history: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Return depth slope and median pairwise residual; no F1-tuned gate."""
    if len(history) < 3 or history[-1][0] - history[0][0] < 0.15:
        return None
    slopes = [
        (later_depth - earlier_depth) / (later_time - earlier_time)
        for index, (earlier_time, earlier_depth) in enumerate(history[:-1])
        for later_time, later_depth in history[index + 1 :]
        if later_time > earlier_time
    ]
    if not slopes:
        return None
    slope = float(statistics.median(slopes))
    intercepts = [depth - slope * timestamp for timestamp, depth in history]
    intercept = float(statistics.median(intercepts))
    residual_mad = float(statistics.median(abs(depth - (slope * timestamp + intercept)) for timestamp, depth in history))
    return slope, residual_mad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.30)
    parser.add_argument("--corridor-half-width-m", type=float, default=1.75)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--focal-px", type=float, default=320.0)
    parser.add_argument("--principal-x-px", type=float, default=320.0)
    args = parser.parse_args()
    if args.history < 3 or args.corridor_half_width_m <= 0.0:
        raise ValueError("history must be >=3 and corridor must be positive")

    with args.evidence.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    truth = load_truth(args.practice_root, args.evidence.stem)
    history: dict[int, list[tuple[float, float]]] = defaultdict(list)
    audited: list[dict[str, object]] = []
    for row, target_ttc in zip(rows, truth, strict=True):
        for update in json.loads(row.get("v2_shadow_updates_json", "[]")):
            if update.get("measurement_source") != "yolo_box_median_disparity":
                continue
            history[int(update["track_id"])].append((float(row["timestamp"]), float(update["depth_m"])))
        raw_ttc = parse_ttc(row["union_predicted_ttc"])
        if not (row["union_source"].startswith("classical") and raw_ttc < 2.0):
            continue
        update, overlap = best_yolo_update(row, minimum_iou=args.minimum_iou)
        if update is None:
            continue
        offset = direct_path_offset_m(
            row, update, focal_px=args.focal_px, principal_x_px=args.principal_x_px
        )
        if offset is None or abs(offset) > args.corridor_half_width_m:
            continue
        track_id = int(update["track_id"])
        estimate = theil_sen_depth_slope(history[track_id][-args.history :])
        if estimate is None:
            continue
        depth_slope, residual_mad = estimate
        audited.append({
            "frame_id": int(row["frame_id"]),
            "truth_group": "true_danger" if target_ttc < 2.0 else "false_danger",
            "truth_ttc": float(target_ttc), "raw_classical_ttc": raw_ttc,
            "track_id": track_id, "matched_iou": overlap,
            "path_offset_m": offset,
            "robust_closing_mps": -depth_slope,
            "depth_fit_residual_mad_m": residual_mad,
            "history_observations": len(history[track_id][-args.history :]),
        })
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in audited:
        grouped[str(item["truth_group"])].append(item)
    summary = {
        group: {
            "rows": len(items),
            "median_robust_closing_mps": float(np.median([float(item["robust_closing_mps"]) for item in items])),
            "median_depth_fit_residual_mad_m": float(np.median([float(item["depth_fit_residual_mad_m"]) for item in items])),
            "nonpositive_closing_rows": sum(float(item["robust_closing_mps"]) <= 0.0 for item in items),
        }
        for group, items in grouped.items() if items
    }
    report = {
        "contract": "diagnostic only; no closing threshold, TTC, association, or risk decision changed",
        "trip_id": args.evidence.stem,
        "history": args.history,
        "rows_by_group": dict(Counter(str(item["truth_group"]) for item in audited)),
        "summary": summary,
        "rows": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
