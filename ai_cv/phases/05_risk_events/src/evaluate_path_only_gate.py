"""Offline, pre-registered path-only event diagnostic over frozen evidence.

This never reruns perception and never changes the V1 TTC.  It reads Phase 17
evidence, emits the unchanged TTC in one column, and emits a separate
event-to-TTC diagnostic column only where direct current-frame geometry is
available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE02_SRC = REPOSITORY_ROOT / "ai_cv" / "phases" / "02_detection_tracking" / "src"
if str(PHASE02_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE02_SRC))

from cross_validate_guarded_ttc import score  # noqa: E402
from path_relative_state import (  # noqa: E402
    camera_measurement_to_planar,
    host_lateral_displacement_m,
    yaw_rate_rps,
)


def official_trip_mean(per_trip: list[dict[str, object]], policy: str) -> dict[str, float | int]:
    """Mirror the challenge report: mean metrics across the six trips."""
    metrics = [item[policy] for item in per_trip]
    assert all(isinstance(item, dict) for item in metrics)
    metric_dicts = [item for item in metrics if isinstance(item, dict)]
    return {
        "f1": float(np.mean([float(item["f1"]) for item in metric_dicts])),
        "composite": float(np.mean([float(item["composite"]) for item in metric_dicts])),
        "mae_critical": float(np.mean([float(item["mae_critical"]) for item in metric_dicts])),
        "precision": float(np.mean([float(item["precision"]) for item in metric_dicts])),
        "recall": float(np.mean([float(item["recall"]) for item in metric_dicts])),
        "tp_sum": int(sum(int(item["tp"]) for item in metric_dicts)),
        "fp_sum": int(sum(int(item["fp"]) for item in metric_dicts)),
        "fn_sum": int(sum(int(item["fn"]) for item in metric_dicts)),
    }


def parse_ttc(value: str) -> float:
    return math.inf if value == "inf" else float(value)


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    if overlap <= 0.0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return overlap / max(1e-9, first_area + second_area - overlap)


def direct_path_offset_m(
    row: dict[str, str], update: dict[str, object], *, focal_px: float, principal_x_px: float
) -> float | None:
    """Return direct current-frame target offset from host's bicycle arc."""
    box = update.get("bbox_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        return None
    depth_m = float(update["depth_m"])
    centre_x_px = (float(box[0]) + float(box[2])) / 2.0
    measurement = camera_measurement_to_planar(
        depth_m=depth_m,
        center_x_px=centre_x_px,
        focal_length_px=focal_px,
        principal_x_px=principal_x_px,
    )
    speed_mps = float(row["ego_speed_mps"])
    yaw_rate = yaw_rate_rps(speed_mps, float(row["lateral_accel_mps2"]))
    if measurement is None or yaw_rate is None or speed_mps <= 1.0:
        return None
    horizon_s = measurement[0] / speed_mps
    if not math.isfinite(horizon_s) or horizon_s < 0.0:
        return None
    host_lateral_m = host_lateral_displacement_m(speed_mps, yaw_rate, horizon_s)
    return float(measurement[1] - host_lateral_m)


def best_yolo_update(
    row: dict[str, str], *, minimum_iou: float
) -> tuple[dict[str, object] | None, float]:
    """Associate the frozen selected classical box to one current YOLO track."""
    raw_box = row.get("classical_selected_bbox_xyxy", "")
    if not raw_box:
        return None, 0.0
    selected_box = json.loads(raw_box)
    candidates = json.loads(row.get("v2_shadow_updates_json", "[]"))
    best: tuple[dict[str, object] | None, float] = (None, 0.0)
    for candidate in candidates:
        if candidate.get("measurement_source") != "yolo_box_median_disparity":
            continue
        candidate_box = candidate.get("bbox_xyxy")
        if not isinstance(candidate_box, list) or len(candidate_box) != 4:
            continue
        overlap = iou(list(map(float, selected_box)), list(map(float, candidate_box)))
        if overlap > best[1]:
            best = (candidate, overlap)
    return best if best[1] >= minimum_iou else (None, best[1])


def gated_ttc(
    row: dict[str, str], *, focal_px: float, principal_x_px: float,
    minimum_iou: float, corridor_half_width_m: float,
) -> tuple[float, dict[str, object]]:
    """Return raw TTC or a finite non-danger event encoding with telemetry."""
    raw_ttc = parse_ttc(row["union_predicted_ttc"])
    audit: dict[str, object] = {
        "frame_id": int(row["frame_id"]),
        "raw_ttc": raw_ttc,
        "eligible": False,
        "matched_iou": 0.0,
        "path_offset_m": None,
        "suppressed": False,
    }
    if not (row["union_source"].startswith("classical") and raw_ttc < 2.0):
        return raw_ttc, audit
    update, overlap = best_yolo_update(row, minimum_iou=minimum_iou)
    audit["matched_iou"] = overlap
    if update is None:
        return raw_ttc, audit
    offset_m = direct_path_offset_m(
        row, update, focal_px=focal_px, principal_x_px=principal_x_px
    )
    if offset_m is None:
        return raw_ttc, audit
    audit["eligible"] = True
    audit["path_offset_m"] = offset_m
    if abs(offset_m) > corridor_half_width_m:
        audit["suppressed"] = True
        return 2.0, audit
    return raw_ttc, audit


def load_truth(practice_root: Path, trip_id: str) -> np.ndarray:
    starter_root = REPOSITORY_ROOT / "Package_starterkit" / "package_starterkit"
    if str(starter_root) not in sys.path:
        sys.path.insert(0, str(starter_root))
    from team_kit.dataset_loader import TripDataset

    dataset = TripDataset(practice_root / trip_id)
    return np.asarray([float(frame.min_ttc) for frame in dataset.iter_frames()], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.30)
    parser.add_argument("--corridor-half-width-m", type=float, default=1.75)
    parser.add_argument("--focal-px", type=float, default=320.0)
    parser.add_argument("--principal-x-px", type=float, default=320.0)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_iou <= 1.0 or args.corridor_half_width_m <= 0.0:
        raise ValueError("invalid fixed association or physical corridor parameter")

    report: dict[str, object] = {"policy": {
        "source": "frozen_phase17_conservative_union",
        "minimum_iou": args.minimum_iou,
        "corridor_half_width_m": args.corridor_half_width_m,
        "event_ttc_when_suppressed": 2.0,
        "raw_ttc_preserved_in_parallel": True,
    }, "per_trip": [], "raw_score": {}, "path_only_event_score": {},
        "pooled_frame_score_diagnostic": {}}
    raw_all: list[float] = []
    gated_all: list[float] = []
    truth_all: list[float] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for evidence_path in sorted(args.evidence_root.glob("T*-Sample.csv")):
        trip_id = evidence_path.stem
        with evidence_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        truth = load_truth(args.practice_root, trip_id)
        if len(rows) != len(truth):
            raise ValueError(f"{trip_id}: frozen evidence/truth frame-count mismatch")
        raw = np.asarray([parse_ttc(row["union_predicted_ttc"]) for row in rows], dtype=float)
        gated_and_audit = [
            gated_ttc(
                row, focal_px=args.focal_px, principal_x_px=args.principal_x_px,
                minimum_iou=args.minimum_iou,
                corridor_half_width_m=args.corridor_half_width_m,
            ) for row in rows
        ]
        gated = np.asarray([item[0] for item in gated_and_audit], dtype=float)
        audit = [item[1] for item in gated_and_audit]
        with (args.output_dir / f"{trip_id}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(audit)
        trip_result = {
            "trip_id": trip_id,
            "raw": asdict(score(raw, truth)),
            "path_only_event": asdict(score(gated, truth)),
            "eligible_frames": sum(bool(item["eligible"]) for item in audit),
            "suppressed_frames": sum(bool(item["suppressed"]) for item in audit),
        }
        report["per_trip"].append(trip_result)
        raw_all.extend(raw)
        gated_all.extend(gated)
        truth_all.extend(truth)
    per_trip = report["per_trip"]
    assert isinstance(per_trip, list)
    report["raw_score"] = official_trip_mean(per_trip, "raw")
    report["path_only_event_score"] = official_trip_mean(per_trip, "path_only_event")
    report["pooled_frame_score_diagnostic"] = {
        "raw": asdict(score(np.asarray(raw_all), np.asarray(truth_all))),
        "path_only_event": asdict(score(np.asarray(gated_all), np.asarray(truth_all))),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
