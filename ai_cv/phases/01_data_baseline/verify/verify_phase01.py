"""Verify strict predictions and the frozen Stage 1 metric regression."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from validate_predictions import validate_prediction_set


def verify_metrics(report: dict, regression: dict) -> None:
    metrics = regression["metrics"]
    tolerance = regression["tolerance"]
    for key in (
        "overall_mae_critical",
        "overall_inv_ttc_mae",
        "overall_f1",
        "overall_composite_score",
    ):
        actual = float(report[key])
        expected = float(metrics[key])
        allowed = float(tolerance[key])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=allowed):
            raise ValueError(
                f"{key}: actual {actual}, expected {expected} ± {allowed}"
            )

    per_trip = {item["trip_id"]: item for item in report["per_trip"]}
    worst_trip = min(per_trip.values(), key=lambda item: item["composite_score"])
    if worst_trip["trip_id"] != metrics["worst_trip"]:
        raise ValueError(
            f"worst trip {worst_trip['trip_id']}, expected {metrics['worst_trip']}"
        )
    expected_worst = float(metrics["worst_trip_composite_score"])
    if not math.isclose(
        float(worst_trip["composite_score"]),
        expected_worst,
        rel_tol=0.0,
        abs_tol=float(tolerance["overall_composite_score"]),
    ):
        raise ValueError(
            f"worst-trip composite {worst_trip['composite_score']}, "
            f"expected {expected_worst}"
        )


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("ai_cv/outputs/predictions/baseline_official"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("ai_cv/outputs/reports/baseline_official/evaluation.json"),
    )
    parser.add_argument(
        "--regression",
        type=Path,
        default=phase_root / "artifacts" / "baseline_regression.json",
    )
    args = parser.parse_args()

    try:
        summaries = validate_prediction_set(args.predictions_root, args.data_root)
        report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
        regression = json.loads(args.regression.read_text(encoding="utf-8"))
        verify_metrics(report, regression)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")

    print(
        f"PASS Stage 1: {len(summaries)} trips, "
        f"composite={report['overall_composite_score']}, "
        f"worst={regression['metrics']['worst_trip']} "
        f"({regression['metrics']['worst_trip_composite_score']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
