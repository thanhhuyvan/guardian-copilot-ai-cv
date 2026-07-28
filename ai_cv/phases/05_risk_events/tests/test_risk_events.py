from __future__ import annotations

import math
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_events import (
    RiskState,
    RiskStateConfig,
    RiskStateMachine,
    build_risk_events,
    nominal_band,
)


def test_nominal_threshold_boundaries() -> None:
    assert nominal_band(math.inf) == "SAFE"
    assert nominal_band(3.0) == "SAFE"
    assert nominal_band(2.0) == "WARNING"
    assert nominal_band(1.5) == "DANGER"
    assert nominal_band(1.499) == "CRITICAL"


def test_danger_debounce_and_exit_hysteresis() -> None:
    machine = RiskStateMachine(
        RiskStateConfig(
            danger_enter_frames=3,
            critical_enter_frames=2,
            danger_exit_frames=3,
        )
    )
    values = [1.8, 1.8, 1.8, math.inf, math.inf, math.inf]
    states = [
        machine.update(index, index * 0.05, value).state
        for index, value in enumerate(values)
    ]
    assert states == [
        RiskState.NORMAL,
        RiskState.ATTENTIVE,
        RiskState.HIGH_RISK,
        RiskState.HIGH_RISK,
        RiskState.HIGH_RISK,
        RiskState.NORMAL,
    ]


def test_critical_uses_shorter_entry_debounce() -> None:
    machine = RiskStateMachine()
    assert machine.update(0, 0.0, 1.0).state == RiskState.NORMAL
    assert machine.update(1, 0.05, 1.0).state == RiskState.HIGH_RISK


def test_unknown_and_trip_reset_clear_state() -> None:
    machine = RiskStateMachine(RiskStateConfig(danger_enter_frames=1))
    assert machine.update(0, 0.0, 1.8).state == RiskState.HIGH_RISK
    assert (
        machine.update(1, 0.05, math.inf, reliable=False).state
        == RiskState.UNKNOWN
    )
    machine.reset()
    assert machine.update(0, 0.0, math.inf).state == RiskState.NORMAL


def test_event_merge_and_minimum_ttc() -> None:
    machine = RiskStateMachine(
        RiskStateConfig(
            danger_enter_frames=1,
            critical_enter_frames=1,
            danger_exit_frames=1,
        )
    )
    values = [1.8, 1.2, math.inf, math.inf, 1.7]
    frames = [
        machine.update(index, index * 0.05, value)
        for index, value in enumerate(values)
    ]

    events = build_risk_events(
        frames,
        trip_id="T00-Sample",
        run_id="test-run",
        merge_gap_frames=2,
    )

    assert len(events) == 1
    assert events[0]["start_frame"] == 0
    assert events[0]["end_frame"] == 4
    assert events[0]["min_ttc_sec"] == 1.2
    assert events[0]["severity"] == "CRITICAL"
