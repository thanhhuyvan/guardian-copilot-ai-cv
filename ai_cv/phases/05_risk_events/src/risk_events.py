"""Deterministic TTC risk-state hysteresis and event aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    ATTENTIVE = "ATTENTIVE"
    HIGH_RISK = "HIGH_RISK"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RiskStateConfig:
    danger_enter_frames: int = 3
    critical_enter_frames: int = 2
    danger_exit_frames: int = 3
    attentive_enter_frames: int = 2
    attentive_exit_frames: int = 4
    event_merge_gap_frames: int = 4

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.danger_enter_frames < 1 or self.critical_enter_frames < 1:
            raise ValueError("danger entry debounce must be positive")
        if self.danger_exit_frames < 1:
            raise ValueError("danger exit debounce must be positive")


@dataclass(frozen=True)
class RiskFrame:
    frame_id: int
    timestamp: float
    ttc_sec: float
    raw_band: str
    state: RiskState
    reason: str


def nominal_band(ttc_sec: float) -> str:
    if not math.isfinite(ttc_sec) or ttc_sec >= 3.0:
        return "SAFE"
    if ttc_sec >= 2.0:
        return "WARNING"
    if ttc_sec >= 1.5:
        return "DANGER"
    return "CRITICAL"


class RiskStateMachine:
    def __init__(self, config: RiskStateConfig | None = None) -> None:
        self.config = config or RiskStateConfig()
        self.reset()

    def reset(self) -> None:
        self.state = RiskState.NORMAL
        self._danger_streak = 0
        self._attentive_streak = 0
        self._danger_clear_streak = 0
        self._attentive_clear_streak = 0

    def update(
        self,
        frame_id: int,
        timestamp: float,
        ttc_sec: float,
        *,
        reliable: bool = True,
    ) -> RiskFrame:
        if not reliable:
            self.state = RiskState.UNKNOWN
            self._danger_streak = 0
            self._attentive_streak = 0
            self._danger_clear_streak = 0
            self._attentive_clear_streak = 0
            return RiskFrame(
                frame_id, timestamp, ttc_sec, "UNKNOWN", self.state, "unreliable"
            )

        band = nominal_band(ttc_sec)
        danger = band in {"DANGER", "CRITICAL"}
        attentive = band in {"WARNING", "DANGER", "CRITICAL"}
        if self.state == RiskState.UNKNOWN:
            self.state = RiskState.NORMAL

        self._danger_streak = self._danger_streak + 1 if danger else 0
        self._attentive_streak = (
            self._attentive_streak + 1 if attentive else 0
        )

        if self.state == RiskState.HIGH_RISK:
            self._danger_clear_streak = (
                0 if danger else self._danger_clear_streak + 1
            )
            if self._danger_clear_streak >= self.config.danger_exit_frames:
                self.state = (
                    RiskState.ATTENTIVE if attentive else RiskState.NORMAL
                )
                self._danger_clear_streak = 0
                return RiskFrame(
                    frame_id,
                    timestamp,
                    ttc_sec,
                    band,
                    self.state,
                    "danger_exit_confirmed",
                )
            return RiskFrame(
                frame_id,
                timestamp,
                ttc_sec,
                band,
                self.state,
                "danger_held",
            )

        required_danger_frames = (
            self.config.critical_enter_frames
            if band == "CRITICAL"
            else self.config.danger_enter_frames
        )
        if danger and self._danger_streak >= required_danger_frames:
            self.state = RiskState.HIGH_RISK
            self._danger_clear_streak = 0
            return RiskFrame(
                frame_id,
                timestamp,
                ttc_sec,
                band,
                self.state,
                "danger_enter_confirmed",
            )

        if self.state == RiskState.ATTENTIVE:
            self._attentive_clear_streak = (
                0 if attentive else self._attentive_clear_streak + 1
            )
            if (
                self._attentive_clear_streak
                >= self.config.attentive_exit_frames
            ):
                self.state = RiskState.NORMAL
                self._attentive_clear_streak = 0
                reason = "attentive_exit_confirmed"
            else:
                reason = "attentive_held"
        elif (
            attentive
            and self._attentive_streak >= self.config.attentive_enter_frames
        ):
            self.state = RiskState.ATTENTIVE
            self._attentive_clear_streak = 0
            reason = "attentive_enter_confirmed"
        else:
            reason = "normal"
        return RiskFrame(
            frame_id, timestamp, ttc_sec, band, self.state, reason
        )


def _state_runs(frames: Sequence[RiskFrame]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, frame in enumerate(frames):
        if frame.state == RiskState.HIGH_RISK and start is None:
            start = index
        elif frame.state != RiskState.HIGH_RISK and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(frames) - 1))
    return runs


def build_risk_events(
    frames: Sequence[RiskFrame],
    *,
    trip_id: str,
    run_id: str,
    merge_gap_frames: int,
) -> list[dict[str, object]]:
    runs = _state_runs(frames)
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= merge_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    events = []
    for event_index, (start, end) in enumerate(merged, start=1):
        selected = frames[start : end + 1]
        finite_ttc = [
            frame.ttc_sec for frame in selected if math.isfinite(frame.ttc_sec)
        ]
        min_ttc = min(finite_ttc) if finite_ttc else 3.0
        duration_frames = end - start + 1
        event_quality = min(1.0, duration_frames / 7.0)
        confidence_level = (
            "HIGH"
            if duration_frames >= 7
            else "MEDIUM"
            if duration_frames >= 3
            else "LOW"
        )
        events.append(
            {
                "schema_version": "risk_event.v1",
                "run_id": run_id,
                "event_id": f"{run_id}-E{event_index}",
                "trip_id": trip_id,
                "start_frame": int(selected[0].frame_id),
                "end_frame": int(selected[-1].frame_id),
                "start_time": float(selected[0].timestamp),
                "end_time": float(selected[-1].timestamp),
                "min_ttc_sec": float(max(0.0, min_ttc)),
                "object_type": "unknown",
                "track_id": 0,
                "severity": "CRITICAL" if min_ttc < 1.5 else "DANGER",
                "event_quality": float(event_quality),
                "confidence_level": confidence_level,
                "clip_path": None,
            }
        )
    return events
