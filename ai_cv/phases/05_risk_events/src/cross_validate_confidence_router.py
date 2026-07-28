"""Leakage-controlled LOTO evaluation of a classical/detector TTC router."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE02_SRC = (
    REPOSITORY_ROOT / "ai_cv" / "phases" / "02_detection_tracking" / "src"
)
if str(PHASE02_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE02_SRC))

from cross_validate_guarded_ttc import Metrics, score  # noqa: E402


TRIPS = tuple(f"T0{index}-Sample" for index in range(1, 7))
BANNED_FEATURE_TOKENS = ("trip", "frame", "timestamp", "truth", "target")
FEATURE_NAMES = (
    "detector_is_danger",
    "detector_minus_classical_confidence",
    "classical_minus_detector_residual",
    "detector_minus_classical_quality",
    "detector_minus_classical_lr_support",
    "detector_minus_classical_history",
    "detector_depth_confidence",
    "detector_minus_classical_proposal_count",
    "detector_risk_track_count",
    "detector_minus_classical_ground_confidence",
    "ego_speed_normalized",
    "inverse_ttc_disagreement",
)


def parse_ttc(value: str | float | None) -> float:
    if value is None:
        return math.inf
    text = str(value).strip().lower()
    if text in {"", "inf", "+inf", "infinity", "nan"}:
        return math.inf
    try:
        result = float(text)
    except ValueError:
        return math.inf
    return result if math.isfinite(result) else math.inf


def numeric(row: dict[str, str], name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def inverse_ttc(value: float) -> float:
    return 1.0 / max(0.1, value) if math.isfinite(value) else 0.0


def router_features(
    classical: dict[str, str],
    detector: dict[str, str],
) -> np.ndarray:
    """Create causal features; identifiers are used only outside this function."""
    classical_ttc = parse_ttc(classical.get("predicted_ttc"))
    detector_ttc = parse_ttc(detector.get("predicted_ttc"))
    classical_residual = numeric(
        classical, "selected_motion_residual_m", default=5.0
    )
    detector_residual = numeric(
        detector, "selected_motion_residual_m", default=5.0
    )
    return np.asarray(
        [
            float(detector_ttc < 2.0),
            numeric(detector, "selection_confidence")
            - numeric(classical, "prediction_confidence"),
            np.clip(classical_residual - detector_residual, -5.0, 5.0) / 5.0,
            numeric(detector, "selected_observation_quality")
            - numeric(classical, "selected_observation_quality"),
            numeric(detector, "selected_lr_support")
            - numeric(classical, "selected_lr_support"),
            (
                numeric(detector, "selected_history_length")
                - numeric(classical, "selected_history_length")
            )
            / 11.0,
            numeric(detector, "selected_depth_confidence"),
            np.clip(
                numeric(detector, "depth_valid_component_count")
                - numeric(classical, "relevant_component_count"),
                -5.0,
                5.0,
            )
            / 5.0,
            min(5.0, numeric(detector, "risk_track_count")) / 5.0,
            numeric(detector, "ground_confidence")
            - numeric(classical, "ground_confidence"),
            min(40.0, numeric(detector, "ego_speed_mps")) / 40.0,
            abs(inverse_ttc(detector_ttc) - inverse_ttc(classical_ttc)),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PairedTrip:
    trip_id: str
    frame_ids: np.ndarray
    truth: np.ndarray
    classical_ttc: np.ndarray
    detector_ttc: np.ndarray
    features: np.ndarray

    @property
    def disagreement(self) -> np.ndarray:
        return (self.classical_ttc < 2.0) != (self.detector_ttc < 2.0)


@dataclass(frozen=True)
class LogisticRouter:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def detector_probability(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        design = np.column_stack(
            [np.ones(normalized.shape[0], dtype=np.float64), normalized]
        )
        logits = np.clip(design @ self.weights, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_router(
    features: np.ndarray,
    choose_detector: np.ndarray,
    *,
    l2: float = 2.0,
    iterations: int = 40,
) -> LogisticRouter:
    """Fit a small class-balanced logistic router with deterministic IRLS."""
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("unexpected router feature shape")
    if features.shape[0] != choose_detector.size or not features.shape[0]:
        raise ValueError("router training rows must be non-empty and aligned")
    labels = choose_detector.astype(np.float64)
    if np.unique(labels).size < 2:
        raise ValueError("router training requires both source-choice classes")

    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = (features - mean) / scale
    design = np.column_stack(
        [np.ones(normalized.shape[0], dtype=np.float64), normalized]
    )
    positives = max(1, int(np.count_nonzero(labels == 1.0)))
    negatives = max(1, int(np.count_nonzero(labels == 0.0)))
    sample_weight = np.where(
        labels == 1.0,
        labels.size / (2.0 * positives),
        labels.size / (2.0 * negatives),
    )
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        gradient = (
            design.T @ (sample_weight * (probability - labels))
            + penalty @ weights
        )
        step = np.linalg.solve(hessian + 1e-8 * np.eye(hessian.shape[0]), gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return LogisticRouter(mean=mean, scale=scale, weights=weights)


def route_predictions(
    data: PairedTrip,
    router: LogisticRouter,
) -> tuple[np.ndarray, np.ndarray]:
    """Use classical on agreement; route only danger-decision disagreements."""
    predictions = data.classical_ttc.copy()
    choose_detector = np.zeros(data.frame_ids.size, dtype=bool)
    disagreement = data.disagreement
    if np.any(disagreement):
        probability = router.detector_probability(data.features[disagreement])
        selected = probability >= 0.5
        disagreement_indices = np.flatnonzero(disagreement)
        choose_detector[disagreement_indices[selected]] = True
        predictions[choose_detector] = data.detector_ttc[choose_detector]
    return predictions, choose_detector


def conservative_union_predictions(data: PairedTrip) -> np.ndarray:
    """Use a detector-only danger decision; otherwise preserve the fallback."""
    predictions = data.classical_ttc.copy()
    detector_only_danger = (
        (data.detector_ttc < 2.0) & ~(data.classical_ttc < 2.0)
    )
    predictions[detector_only_danger] = data.detector_ttc[detector_only_danger]
    return predictions


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_paired_trip(
    trip_id: str,
    classical_diagnostics: Path,
    detector_evidence: Path,
    practice_root: Path,
    starter_root: Path,
) -> PairedTrip:
    classical_rows = [
        row
        for row in _load_rows(classical_diagnostics / f"{trip_id}.csv")
        if row.get("variant") == "track_p35_guarded"
    ]
    detector_rows = _load_rows(detector_evidence / f"{trip_id}.csv")
    classical_by_frame = {int(row["frame_id"]): row for row in classical_rows}
    detector_by_frame = {int(row["frame_id"]): row for row in detector_rows}
    if set(classical_by_frame) != set(detector_by_frame):
        raise ValueError(f"{trip_id} classical/detector frame IDs differ")

    resolved_starter = str(starter_root.resolve())
    if resolved_starter not in sys.path:
        sys.path.insert(0, resolved_starter)
    from team_kit.dataset_loader import TripDataset

    dataset = TripDataset(practice_root / trip_id)
    truth_by_frame = {
        int(frame.frame_id): float(frame.min_ttc)
        for frame in dataset.iter_frames()
    }
    frame_ids = np.asarray(sorted(classical_by_frame), dtype=np.int64)
    if set(frame_ids.tolist()) != set(truth_by_frame):
        raise ValueError(f"{trip_id} paired evidence does not cover source frames")
    return PairedTrip(
        trip_id=trip_id,
        frame_ids=frame_ids,
        truth=np.asarray([truth_by_frame[int(item)] for item in frame_ids]),
        classical_ttc=np.asarray(
            [
                parse_ttc(classical_by_frame[int(item)].get("predicted_ttc"))
                for item in frame_ids
            ]
        ),
        detector_ttc=np.asarray(
            [
                parse_ttc(detector_by_frame[int(item)].get("predicted_ttc"))
                for item in frame_ids
            ]
        ),
        features=np.vstack(
            [
                router_features(
                    classical_by_frame[int(item)],
                    detector_by_frame[int(item)],
                )
                for item in frame_ids
            ]
        ),
    )


def _training_data(
    trips: Sequence[PairedTrip],
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = []
    labels = []
    for data in trips:
        disagreement = data.disagreement
        feature_rows.append(data.features[disagreement])
        detector_correct = (
            (data.detector_ttc[disagreement] < 2.0)
            == (data.truth[disagreement] < 2.0)
        )
        labels.append(detector_correct)
    return np.vstack(feature_rows), np.concatenate(labels)


def _metric_document(metric: Metrics) -> dict[str, float | int]:
    return asdict(metric)


def cross_validate(
    trips: Sequence[PairedTrip],
) -> dict[str, object]:
    folds = []
    router_metrics = []
    classical_metrics = []
    detector_metrics = []
    conservative_union_metrics = []
    for held_out in trips:
        training = [trip for trip in trips if trip.trip_id != held_out.trip_id]
        features, labels = _training_data(training)
        router = fit_router(features, labels)
        predictions, selected_detector = route_predictions(held_out, router)
        router_metric = score(predictions, held_out.truth)
        classical_metric = score(held_out.classical_ttc, held_out.truth)
        detector_metric = score(held_out.detector_ttc, held_out.truth)
        conservative_union_metric = score(
            conservative_union_predictions(held_out), held_out.truth
        )
        router_metrics.append(router_metric)
        classical_metrics.append(classical_metric)
        detector_metrics.append(detector_metric)
        conservative_union_metrics.append(conservative_union_metric)
        folds.append(
            {
                "held_out_trip": held_out.trip_id,
                "training_disagreements": int(features.shape[0]),
                "training_detector_choice_fraction": float(np.mean(labels)),
                "validation_disagreements": int(
                    np.count_nonzero(held_out.disagreement)
                ),
                "validation_detector_choices": int(
                    np.count_nonzero(selected_detector)
                ),
                "router": _metric_document(router_metric),
                "classical": _metric_document(classical_metric),
                "detector": _metric_document(detector_metric),
                "conservative_union": _metric_document(
                    conservative_union_metric
                ),
            }
        )

    def macro(metrics: Sequence[Metrics], name: str) -> float:
        return float(np.mean([getattr(metric, name) for metric in metrics]))

    return {
        "schema": "guardian.phase05.confidence-router-loto.v1",
        "feature_names": list(FEATURE_NAMES),
        "banned_feature_tokens": list(BANNED_FEATURE_TOKENS),
        "selection_scope": "danger-decision disagreements only",
        "agreement_policy": "classical fallback",
        "model": {
            "type": "class-balanced logistic regression",
            "l2": 2.0,
            "threshold": 0.5,
        },
        "macro": {
            "router_f1": macro(router_metrics, "f1"),
            "classical_f1": macro(classical_metrics, "f1"),
            "detector_f1": macro(detector_metrics, "f1"),
            "router_composite": macro(router_metrics, "composite"),
            "classical_composite": macro(classical_metrics, "composite"),
            "detector_composite": macro(detector_metrics, "composite"),
            "router_mae_critical": macro(router_metrics, "mae_critical"),
            "classical_mae_critical": macro(classical_metrics, "mae_critical"),
            "detector_mae_critical": macro(detector_metrics, "mae_critical"),
            "conservative_union_f1": macro(
                conservative_union_metrics, "f1"
            ),
            "conservative_union_composite": macro(
                conservative_union_metrics, "composite"
            ),
            "conservative_union_mae_critical": macro(
                conservative_union_metrics, "mae_critical"
            ),
        },
        "decision": {
            "learned_router_passed": (
                macro(router_metrics, "f1")
                >= max(
                    macro(classical_metrics, "f1"),
                    macro(detector_metrics, "f1"),
                )
                and macro(router_metrics, "composite")
                >= macro(classical_metrics, "composite")
            ),
            "recommended_runtime_policy": "conservative_union_candidate",
            "promotion_blocker": (
                "T01 per-trip regression remains; external validation and "
                "event-level false-alert analysis are required."
            ),
        },
        "folds": folds,
    }


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
        "--output",
        type=Path,
        default=Path(
            "ai_cv/phases/05_risk_events/artifacts/"
            "confidence_router_loto.json"
        ),
    )
    args = parser.parse_args()
    if any(
        token in feature.lower()
        for feature in FEATURE_NAMES
        for token in BANNED_FEATURE_TOKENS
    ):
        raise RuntimeError("router feature list contains a banned leakage token")
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
    report = cross_validate(trips)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
