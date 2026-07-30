"""Shadow-audit a fixed image-expansion TTC cue against frozen V1 evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
for path in (Path(__file__).resolve().parent, ROOT / "Package_starterkit" / "package_starterkit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v2_risk_cues import looming_tau, ttc_cues_agree  # noqa: E402


TRIPS = tuple(f"T{index:02d}-Sample" for index in range(1, 7))
RATIO = 2.0
MAX_GAP_S = 0.11


def ttc(value: str) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.inf
    except ValueError:
        return math.inf


def bbox_area(value: str) -> float | None:
    try:
        x1, y1, x2, y2 = json.loads(value)
        area = (float(x2) - float(x1)) * (float(y2) - float(y1))
        return area if area > 0.0 else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from team_kit.dataset_loader import TripDataset

    categories: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, object]] = []
    total_danger = 0
    eligible = 0
    histories: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for trip_id in TRIPS:
        truth = {
            int(frame.frame_id): float(frame.min_ttc)
            for frame in TripDataset(args.practice_root / trip_id).iter_frames()
        }
        with (args.evidence_root / f"{trip_id}.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            v1_ttc = ttc(row["union_predicted_ttc"])
            if not v1_ttc < 2.0:
                continue
            total_danger += 1
            track_id = row.get("union_selected_track_id", "").strip()
            area = bbox_area(row.get("union_selected_bbox_xyxy", ""))
            timestamp = float(row["timestamp"])
            if not track_id or area is None:
                categories["unavailable"]["all"] += 1
                continue
            key = (trip_id, track_id)
            history = histories.get(key, [])
            if history and timestamp - history[-1][0] > MAX_GAP_S:
                history = []
            history = (history + [(timestamp, area)])[-5:]
            histories[key] = history
            tau = looming_tau([item[1] for item in history], [item[0] for item in history])
            if not math.isfinite(tau):
                categories["unavailable"]["all"] += 1
                continue
            eligible += 1
            truth_class = "true_danger" if truth[int(row["frame_id"])] < 2.0 else "false_danger"
            outcome = "agree" if ttc_cues_agree(v1_ttc, tau, ratio=RATIO) else "disagree"
            categories[truth_class][outcome] += 1
            if len(examples) < 30:
                examples.append({
                    "trip_id": trip_id,
                    "frame_id": int(row["frame_id"]),
                    "truth_class": truth_class,
                    "v1_ttc_s": round(v1_ttc, 3),
                    "looming_tau_s": round(tau, 3),
                    "outcome": outcome,
                })
    report = {
        "schema": "guardian.looming-shadow-audit.v1",
        "policy": "shadow only; no predictions are changed",
        "fixed_ratio": RATIO,
        "track_history": "selected V1 union box, 3-5 consecutive samples, reset after >0.11 s gap",
        "v1_danger_frames": total_danger,
        "eligible_finite_looming_frames": eligible,
        "coverage": eligible / total_danger if total_danger else 0.0,
        "outcomes": {key: dict(value) for key, value in sorted(categories.items())},
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
