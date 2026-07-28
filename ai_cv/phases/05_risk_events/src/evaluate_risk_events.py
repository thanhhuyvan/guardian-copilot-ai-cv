"""Evaluate fixed Phase 05 hysteresis policies on conservative-union TTC."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE02_SRC = (
    REPOSITORY_ROOT / "ai_cv" / "phases" / "02_detection_tracking" / "src"
)
for path in (Path(__file__).resolve().parent, PHASE02_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cross_validate_confidence_router import (  # noqa: E402
    TRIPS,
    PairedTrip,
    conservative_union_predictions,
    load_paired_trip,
)
from cross_validate_guarded_ttc import score  # noqa: E402
from risk_events import (  # noqa: E402
    RiskFrame,
    RiskState,
    RiskStateConfig,
    RiskStateMachine,
    build_risk_events,
)


CONFIGURATIONS = {
    "raw": RiskStateConfig(
        danger_enter_frames=1,
        critical_enter_frames=1,
        danger_exit_frames=1,
        attentive_enter_frames=1,
        attentive_exit_frames=1,
        event_merge_gap_frames=0,
    ),
    "debounce_2": RiskStateConfig(
        danger_enter_frames=2,
        critical_enter_frames=2,
        danger_exit_frames=2,
        attentive_enter_frames=2,
        attentive_exit_frames=3,
        event_merge_gap_frames=2,
    ),
    # Frozen before evaluation from the Phase 05 research plan.
    "recommended": RiskStateConfig(
        danger_enter_frames=3,
        critical_enter_frames=2,
        danger_exit_frames=3,
        attentive_enter_frames=2,
        attentive_exit_frames=4,
        event_merge_gap_frames=4,
    ),
    "strict_4": RiskStateConfig(
        danger_enter_frames=4,
        critical_enter_frames=2,
        danger_exit_frames=3,
        attentive_enter_frames=2,
        attentive_exit_frames=4,
        event_merge_gap_frames=4,
    ),
}


def run_state_machine(
    data: PairedTrip,
    predictions: np.ndarray,
    config: RiskStateConfig,
) -> list[RiskFrame]:
    machine = RiskStateMachine(config)
    return [
        machine.update(
            int(frame_id),
            index * 0.05,
            float(ttc),
            reliable=True,
        )
        for index, (frame_id, ttc) in enumerate(
            zip(data.frame_ids, predictions, strict=True)
        )
    ]


def _mask_runs(mask: np.ndarray, merge_gap_frames: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, mask.size - 1))
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= merge_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _event_overlap(
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]


def event_metrics(
    data: PairedTrip,
    frames: Sequence[RiskFrame],
    events: Sequence[dict[str, object]],
) -> dict[str, float | int]:
    truth_danger = data.truth < 2.0
    predicted_high = np.asarray(
        [frame.state == RiskState.HIGH_RISK for frame in frames], dtype=bool
    )
    truth_runs = _mask_runs(truth_danger, merge_gap_frames=2)
    predicted_runs = [
        (
            int(np.searchsorted(data.frame_ids, int(event["start_frame"]))),
            int(np.searchsorted(data.frame_ids, int(event["end_frame"]))),
        )
        for event in events
    ]
    matched_truth = [
        any(_event_overlap(truth_run, predicted) for predicted in predicted_runs)
        for truth_run in truth_runs
    ]
    matched_predictions = [
        any(_event_overlap(predicted, truth_run) for truth_run in truth_runs)
        for predicted in predicted_runs
    ]
    onset_delays = []
    fragmentation = 0
    for truth_run in truth_runs:
        overlaps = [
            predicted
            for predicted in predicted_runs
            if _event_overlap(predicted, truth_run)
        ]
        if overlaps:
            onset_delays.append(max(0, min(item[0] for item in overlaps) - truth_run[0]))
            fragmentation += max(0, len(overlaps) - 1)
    true_events = len(truth_runs)
    predicted_events = len(predicted_runs)
    event_tp = int(sum(matched_truth))
    event_fn = true_events - event_tp
    event_fp = predicted_events - int(sum(matched_predictions))
    event_recall = event_tp / true_events if true_events else 1.0
    event_precision = (
        int(sum(matched_predictions)) / predicted_events
        if predicted_events
        else (1.0 if not true_events else 0.0)
    )
    alert_ttc = np.where(predicted_high, 1.0, math.inf)
    alert_frame_metric = score(alert_ttc, data.truth)
    return {
        "truth_events": true_events,
        "predicted_events": predicted_events,
        "event_tp": event_tp,
        "event_fp": event_fp,
        "event_fn": event_fn,
        "event_recall": float(event_recall),
        "event_precision": float(event_precision),
        "fragmentation": fragmentation,
        "mean_onset_delay_frames": (
            float(np.mean(onset_delays)) if onset_delays else 0.0
        ),
        "false_high_frames": int(
            np.count_nonzero(predicted_high & ~truth_danger)
        ),
        "false_event_duration_sec": float(
            np.count_nonzero(predicted_high & ~truth_danger) * 0.05
        ),
        "high_risk_frames": int(np.count_nonzero(predicted_high)),
        "alert_state_f1": float(alert_frame_metric.f1),
        "alert_state_precision": float(alert_frame_metric.precision),
        "alert_state_recall": float(alert_frame_metric.recall),
    }


def evaluate(
    trips: Sequence[PairedTrip],
    output_root: Path,
    schema_path: Path,
) -> dict[str, object]:
    from jsonschema import Draft202012Validator

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    report: dict[str, object] = {
        "schema": "guardian.phase05.risk-event-evaluation.v1",
        "input_policy": "conservative_union",
        "truth_event_definition": "TTC < 2 s, merge gaps <= 2 frames",
        "selected_configuration": "recommended",
        "configurations": {},
    }
    selected_events: dict[str, list[dict[str, object]]] = {}
    for config_name, config in CONFIGURATIONS.items():
        trip_reports = {}
        processing_times = []
        for data in trips:
            predictions = conservative_union_predictions(data)
            started = time.perf_counter()
            frames = run_state_machine(data, predictions, config)
            events = build_risk_events(
                frames,
                trip_id=data.trip_id,
                run_id=f"phase05-{data.trip_id}",
                merge_gap_frames=config.event_merge_gap_frames,
            )
            processing_times.append(
                (time.perf_counter() - started) * 1000.0 / len(frames)
            )
            for event in events:
                errors = sorted(
                    validator.iter_errors(event), key=lambda item: list(item.path)
                )
                if errors:
                    raise ValueError(
                        f"invalid event {event['event_id']}: {errors[0].message}"
                    )
            trip_reports[data.trip_id] = event_metrics(data, frames, events)
            if config_name == "recommended":
                selected_events[data.trip_id] = events

        values = list(trip_reports.values())
        report["configurations"][config_name] = {
            "config": asdict(config),
            "macro_event_recall": float(
                np.mean([item["event_recall"] for item in values])
            ),
            "macro_event_precision": float(
                np.mean([item["event_precision"] for item in values])
            ),
            "macro_alert_state_f1": float(
                np.mean([item["alert_state_f1"] for item in values])
            ),
            "total_false_event_duration_sec": float(
                sum(item["false_event_duration_sec"] for item in values)
            ),
            "total_predicted_events": int(
                sum(item["predicted_events"] for item in values)
            ),
            "total_fragmentation": int(
                sum(item["fragmentation"] for item in values)
            ),
            "processing_ms_per_frame": {
                "mean": float(np.mean(processing_times)),
                "maximum_trip_mean": float(np.max(processing_times)),
            },
            "trips": trip_reports,
        }

    events_root = output_root / "risk_events"
    events_root.mkdir(parents=True, exist_ok=True)
    for trip_id, events in selected_events.items():
        (events_root / f"{trip_id}.json").write_text(
            json.dumps(events, indent=2, sort_keys=True), encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classical-diagnostics",
        type=Path,
        default=Path("ai_cv/outputs/benchmarks/phase03_guarded/diagnostics"),
    )
    parser.add_argument(
        "--detector-evidence",
        type=Path,
        default=Path("ai_cv/outputs/phase05_router_evidence/evidence"),
    )
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("ai_cv/shared/contracts/risk_event.v1.schema.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ai_cv/phases/05_risk_events/artifacts/event_evaluation"),
    )
    args = parser.parse_args()
    trips = [
        load_paired_trip(
            trip_id,
            args.classical_diagnostics,
            args.detector_evidence,
            args.practice_root,
            args.starter_root,
        )
        for trip_id in TRIPS
    ]
    report = evaluate(trips, args.output_root, args.schema_path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "risk_event_evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
