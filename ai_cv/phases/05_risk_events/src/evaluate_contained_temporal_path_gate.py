"""Score a fail-open contained-continuous extension of the path-only cue."""

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
from evaluate_path_only_gate import (  # noqa: E402
    direct_path_offset_m,
    load_truth,
    official_trip_mean,
    parse_ttc,
)


def contained_candidates(row: dict[str, str]) -> list[dict[str, object]]:
    """Return YOLO tracks whose centre lies in the selected classical box."""
    raw_box = row.get("classical_selected_bbox_xyxy", "")
    if not raw_box:
        return []
    x0, y0, x1, y1 = map(float, json.loads(raw_box))
    candidates = []
    for candidate in json.loads(row.get("v2_shadow_updates_json", "[]")):
        if candidate.get("measurement_source") != "yolo_box_median_disparity":
            continue
        box = candidate.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        centre_x = (float(box[0]) + float(box[2])) / 2.0
        centre_y = (float(box[1]) + float(box[3])) / 2.0
        if x0 <= centre_x <= x1 and y0 <= centre_y <= y1:
            candidates.append(candidate)
    return candidates


def gate_trip(
    rows: list[dict[str, str]], *, focal_px: float, principal_x_px: float,
    corridor_half_width_m: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Apply the extension; every unsafe condition preserves the frozen TTC."""
    predictions: list[float] = []
    audit: list[dict[str, object]] = []
    previous_frame: int | None = None
    previous_track: int | None = None
    for row in rows:
        raw_ttc = parse_ttc(row["union_predicted_ttc"])
        frame_id = int(row["frame_id"])
        record: dict[str, object] = {
            "frame_id": frame_id, "raw_ttc": raw_ttc, "event_ttc": raw_ttc,
            "association_status": "not_classical_danger", "path_offset_m": None,
            "suppressed": False,
        }
        danger = row["union_source"].startswith("classical") and raw_ttc < 2.0
        if not danger:
            previous_frame, previous_track = frame_id, None
            predictions.append(raw_ttc); audit.append(record)
            continue
        candidates = contained_candidates(row)
        if len(candidates) != 1:
            record["association_status"] = "ambiguous_or_missing_containment"
            previous_frame, previous_track = frame_id, None
            predictions.append(raw_ttc); audit.append(record)
            continue
        candidate = candidates[0]
        candidate_id = int(candidate["track_id"])
        contiguous = previous_frame == frame_id - 1 and previous_track == candidate_id
        previous_frame, previous_track = frame_id, candidate_id
        if not contiguous:
            record["association_status"] = "episode_start_or_no_continuity"
            predictions.append(raw_ttc); audit.append(record)
            continue
        offset = direct_path_offset_m(
            row, candidate, focal_px=focal_px, principal_x_px=principal_x_px
        )
        if offset is None:
            record["association_status"] = "continuous_geometry_unavailable"
            predictions.append(raw_ttc); audit.append(record)
            continue
        record["path_offset_m"] = offset
        if abs(offset) > corridor_half_width_m:
            record["association_status"] = "continuous_off_path"
            record["event_ttc"] = 2.0
            record["suppressed"] = True
            predictions.append(2.0)
        else:
            record["association_status"] = "continuous_on_path"
            predictions.append(raw_ttc)
        audit.append(record)
    return np.asarray(predictions, dtype=float), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corridor-half-width-m", type=float, default=1.75)
    parser.add_argument("--focal-px", type=float, default=320.0)
    parser.add_argument("--principal-x-px", type=float, default=320.0)
    args = parser.parse_args()
    if args.corridor_half_width_m <= 0.0:
        raise ValueError("corridor half-width must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "policy": {
            "association": "unique_containment_and_prior_frame_continuity",
            "corridor_half_width_m": args.corridor_half_width_m,
            "event_ttc_when_off_path": 2.0,
            "fail_open": True,
        },
        "per_trip": [],
    }
    for evidence_path in sorted(args.evidence_root.glob("T*-Sample.csv")):
        with evidence_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        truth = load_truth(args.practice_root, evidence_path.stem)
        if len(rows) != len(truth):
            raise ValueError(f"{evidence_path.stem}: frame count mismatch")
        raw = np.asarray([parse_ttc(row["union_predicted_ttc"]) for row in rows])
        gated, audit = gate_trip(
            rows, focal_px=args.focal_px, principal_x_px=args.principal_x_px,
            corridor_half_width_m=args.corridor_half_width_m,
        )
        with (args.output_dir / f"{evidence_path.stem}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
            writer.writeheader(); writer.writerows(audit)
        report["per_trip"].append({
            "trip_id": evidence_path.stem,
            "raw": asdict(score(raw, truth)),
            "contained_continuous_path": asdict(score(gated, truth)),
            "suppressed_frames": sum(bool(item["suppressed"]) for item in audit),
            "status_counts": {
                status: sum(item["association_status"] == status for item in audit)
                for status in sorted({str(item["association_status"]) for item in audit})
            },
        })
    per_trip = report["per_trip"]
    assert isinstance(per_trip, list)
    report["raw_score"] = official_trip_mean(per_trip, "raw")
    report["contained_continuous_path_score"] = official_trip_mean(
        per_trip, "contained_continuous_path"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
