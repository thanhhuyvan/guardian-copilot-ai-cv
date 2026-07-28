"""Run the Phase 06 visual robustness screening matrix without altering data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE05_SRC = REPOSITORY_ROOT / "ai_cv" / "phases" / "05_risk_events" / "src"
PHASE06_SRC = Path(__file__).resolve().parent
for path in (PHASE05_SRC, PHASE06_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_detector_owned_ttc import run  # noqa: E402
from run_integrated_deployment import ALL_TRIPS, parse_args as deployment_args  # noqa: E402
from robustness import (  # noqa: E402
    Perturbation,
    VISUAL_PERTURBATIONS,
    apply_perturbation,
    screening_selector,
)


def _macro(report: dict[str, Any]) -> dict[str, float]:
    metrics = [
        trip["metrics"]["conservative_union"]
        for trip in report["trips"].values()
    ]
    return {
        key: float(np.mean([item[key] for item in metrics]))
        for key in ("f1", "composite", "mae_critical", "precision", "recall")
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic in-memory Phase 06 visual screening."
    )
    parser.add_argument("--safe-stride", type=int, default=8)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--perturbations", nargs="+", default=list(VISUAL_PERTURBATIONS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "ai_cv" / "outputs" / "phase06_robustness",
    )
    parser.add_argument("--model-path", type=Path, default=REPOSITORY_ROOT / "yolo26n.pt")
    parser.add_argument("--trips", nargs="+", default=ALL_TRIPS)
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--stereo-workers", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--depth-confidence-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--minimum-depth-confidence", type=float, default=0.25)
    return parser.parse_args(argv)


def _base_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    deployment = deployment_args([])
    deployment.output_dir = output_dir
    deployment.model_path = args.model_path
    deployment.trips = args.trips
    deployment.repeats = 1
    deployment.warmup_frames = 100
    deployment.opencv_threads = args.opencv_threads
    deployment.stereo_workers = args.stereo_workers
    deployment.progress_every = args.progress_every
    deployment.depth_confidence_gate = args.depth_confidence_gate
    deployment.minimum_depth_confidence = args.minimum_depth_confidence
    return deployment


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if args.safe_stride < 1:
        raise ValueError("safe stride must be positive")
    invalid = set(args.perturbations) - set(VISUAL_PERTURBATIONS)
    if invalid:
        raise ValueError(f"unsupported perturbations: {sorted(invalid)}")
    if set(args.severities) - {1, 2, 3}:
        raise ValueError("severities must be drawn from 1, 2, 3")

    selector = lambda frame: screening_selector(frame, safe_stride=args.safe_stride)
    rows: list[dict[str, Any]] = []
    baseline = run(_base_args(args, args.output_dir / "clean"), frame_selector=selector)
    baseline_metrics = _macro(baseline)
    rows.append({"condition": "clean", "severity": 0, **baseline_metrics})

    for kind in args.perturbations:
        for severity in args.severities:
            perturbation = Perturbation(kind, severity)

            def transform(
                left: np.ndarray,
                right: np.ndarray,
                trip_id: str,
                frame_id: int,
            ) -> tuple[np.ndarray, np.ndarray]:
                return apply_perturbation(
                    left,
                    right,
                    trip_id=trip_id,
                    frame_id=frame_id,
                    perturbation=perturbation,
                )

            report = run(
                _base_args(args, args.output_dir / f"{kind}_s{severity}"),
                image_transform=transform,
                frame_selector=selector,
            )
            metrics = _macro(report)
            rows.append(
                {
                    "condition": kind,
                    "severity": severity,
                    **metrics,
                    "f1_delta_from_clean": metrics["f1"] - baseline_metrics["f1"],
                    "composite_delta_from_clean": (
                        metrics["composite"] - baseline_metrics["composite"]
                    ),
                }
            )

    summary = {
        "protocol": {
            "screening_population": "all danger frames plus every Nth safe frame",
            "safe_stride": args.safe_stride,
            "trips": args.trips,
            "source_dataset_modified": False,
            "repeats": 1,
        },
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "robustness_screening.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run_matrix(parse_args())
