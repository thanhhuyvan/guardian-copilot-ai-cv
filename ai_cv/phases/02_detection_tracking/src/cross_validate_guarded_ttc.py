"""Leave-one-trip-out validation for the Guardian track-p35 TTC guard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


TRIPS = tuple(f"T0{index}-Sample" for index in range(1, 7))
CURRENT_GUARD = {
    "corridor_top_width": 0.10,
    "corridor_bottom_width": 0.50,
    "minimum_bottom": 0.50,
    "minimum_height": 0.05,
    "minimum_confidence": 0.75,
    "maximum_closing_speed_mps": 20.0,
    "maximum_depth_m": 20.0,
    "maximum_motion_residual_m": 0.8,
}
MINIMUM_COMPOSITE = 28.7


@dataclass(frozen=True)
class GuardConfig:
    corridor_top_width: float
    corridor_bottom_width: float
    minimum_bottom: float
    minimum_height: float
    minimum_confidence: float
    maximum_closing_speed_mps: float
    maximum_depth_m: float
    maximum_motion_residual_m: float


@dataclass(frozen=True)
class Metrics:
    f1: float
    composite: float
    mae_critical: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


@dataclass
class TripData:
    trip_id: str
    frame_ids: np.ndarray
    ground_truth: np.ndarray
    candidate_frame_index: np.ndarray
    center_x: np.ndarray
    bottom_y: np.ndarray
    height: np.ndarray
    confidence: np.ndarray
    closing_speed: np.ndarray
    depth: np.ndarray
    residual: np.ndarray
    ttc: np.ndarray


def parse_ttc(value: str | None) -> float:
    if value is None or value.strip().lower() in {"", "inf", "+inf", "infinity"}:
        return math.inf
    try:
        result = float(value)
    except ValueError:
        return math.inf
    return result if math.isfinite(result) else math.inf


def load_trip(source_root: Path, trip_id: str) -> TripData:
    truth_path = source_root / "predictions" / "track_p35" / f"{trip_id}.csv"
    candidate_path = source_root / "track_candidates" / f"{trip_id}.csv"
    if not truth_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError(
            f"missing frozen predictions/candidates for {trip_id} under {source_root}"
        )

    with truth_path.open(encoding="utf-8", newline="") as handle:
        truth_rows = list(csv.DictReader(handle))
    frame_ids = np.asarray([int(row["frame_id"]) for row in truth_rows], dtype=int)
    ground_truth = np.asarray(
        [parse_ttc(row.get("ground_truth_ttc")) for row in truth_rows],
        dtype=float,
    )
    frame_positions = {frame_id: index for index, frame_id in enumerate(frame_ids)}

    with candidate_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    return TripData(
        trip_id=trip_id,
        frame_ids=frame_ids,
        ground_truth=ground_truth,
        candidate_frame_index=np.asarray(
            [frame_positions[int(row["frame_id"])] for row in rows],
            dtype=int,
        ),
        center_x=values("selected_center_x_norm"),
        bottom_y=values("selected_bottom_y_norm"),
        height=values("selected_height_norm"),
        confidence=values("confidence"),
        closing_speed=values("closing_speed_mps"),
        depth=values("depth_m"),
        residual=values("motion_residual_m"),
        ttc=np.asarray([parse_ttc(row["candidate_ttc"]) for row in rows]),
    )


def corridor_membership(
    center_x: np.ndarray,
    bottom_y: np.ndarray,
    top_width: float,
    bottom_width: float,
) -> np.ndarray:
    progress = np.clip((bottom_y - 0.36) / (1.0 - 0.36), 0.0, 1.0)
    half_width = (top_width + progress * (bottom_width - top_width)) / 2.0
    return (bottom_y >= 0.36) & (np.abs(center_x - 0.5) <= half_width)


def predict(data: TripData, config: GuardConfig) -> np.ndarray:
    accepted = (
        corridor_membership(
            data.center_x,
            data.bottom_y,
            config.corridor_top_width,
            config.corridor_bottom_width,
        )
        & (data.bottom_y >= config.minimum_bottom)
        & (data.height >= config.minimum_height)
        & (data.confidence >= config.minimum_confidence)
        & (data.closing_speed <= config.maximum_closing_speed_mps)
        & (data.depth <= config.maximum_depth_m)
        & (data.residual <= config.maximum_motion_residual_m)
        & np.isfinite(data.ttc)
    )
    predictions = np.full(data.frame_ids.size, math.inf, dtype=float)
    np.minimum.at(
        predictions,
        data.candidate_frame_index[accepted],
        data.ttc[accepted],
    )
    return predictions


def score(predictions: np.ndarray, truth: np.ndarray) -> Metrics:
    pred_danger = predictions < 2.0
    true_danger = truth < 2.0
    tp = int(np.count_nonzero(pred_danger & true_danger))
    fp = int(np.count_nonzero(pred_danger & ~true_danger))
    fn = int(np.count_nonzero(~pred_danger & true_danger))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    critical = truth < 3.0
    clipped_predictions = np.where(np.isfinite(predictions), predictions, 99.0)
    mae_critical = float(
        np.mean(np.abs(clipped_predictions[critical] - truth[critical]))
    )
    inv_prediction = np.where(
        np.isfinite(predictions),
        1.0 / np.maximum(predictions, 0.1),
        0.0,
    )
    inv_truth = np.where(
        np.isfinite(truth),
        1.0 / np.maximum(truth, 0.1),
        0.0,
    )
    inv_mae = float(np.mean(np.abs(inv_prediction - inv_truth)))
    composite = (
        0.40 * max(0.0, 100.0 - 20.0 * mae_critical)
        + 0.30 * 100.0 * f1
        + 0.30 * max(0.0, 100.0 - 200.0 * inv_mae)
    )
    return Metrics(
        f1=f1,
        composite=composite,
        mae_critical=mae_critical,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def guard_grid() -> list[GuardConfig]:
    corridor_pairs = ((0.10, 0.50), (0.12, 0.50), (0.12, 0.55), (0.16, 0.55))
    return [
        GuardConfig(top, bottom, min_bottom, min_height, confidence, speed, depth, residual)
        for (top, bottom), min_bottom, min_height, confidence, speed, depth, residual
        in product(
            corridor_pairs,
            (0.45, 0.50, 0.55),
            (0.03, 0.05, 0.08),
            (0.70, 0.75, 0.80),
            (15.0, 20.0, 25.0),
            (15.0, 20.0, 25.0),
            (0.5, 0.8, 1.1),
        )
    ]


def current_guard_index(configs: Sequence[GuardConfig]) -> int:
    target = GuardConfig(**CURRENT_GUARD)
    return configs.index(target)


def configuration_distance(config: GuardConfig) -> float:
    scales = {
        "corridor_top_width": 0.02,
        "corridor_bottom_width": 0.05,
        "minimum_bottom": 0.05,
        "minimum_height": 0.02,
        "minimum_confidence": 0.05,
        "maximum_closing_speed_mps": 5.0,
        "maximum_depth_m": 5.0,
        "maximum_motion_residual_m": 0.3,
    }
    return sum(
        abs(getattr(config, name) - CURRENT_GUARD[name]) / scale
        for name, scale in scales.items()
    )


def write_predictions(
    path: Path,
    data: TripData,
    predictions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_id", "predicted_ttc"),
        )
        writer.writeheader()
        for frame_id, value in zip(data.frame_ids, predictions):
            writer.writerow(
                {
                    "frame_id": int(frame_id),
                    "predicted_ttc": (
                        "inf" if not math.isfinite(value) else round(float(value), 6)
                    ),
                }
            )


def write_chart(folds: list[dict], path: Path) -> None:
    labels = [fold["held_out_trip"].replace("-Sample", "") for fold in folds]
    train = [fold["training"]["f1"] for fold in folds]
    held = [fold["held_out"]["f1"] for fold in folds]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.bar(positions - 0.18, train, 0.36, label="five-trip training")
    axis.bar(positions + 0.18, held, 0.36, label="held-out trip")
    axis.axhline(0.60, color="black", linestyle="--", linewidth=1, label="F1 target")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Danger F1")
    axis.set_title("Leave-one-trip-out generalization")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_markdown(summary: dict, path: Path) -> None:
    gap = summary["mean_train_minus_heldout_f1_gap"]
    if gap <= 0.03:
        interpretation = "low evidence of guard overfitting"
    elif gap <= 0.08:
        interpretation = "moderate evidence of guard overfitting"
    else:
        interpretation = "strong evidence of guard overfitting"
    lines = [
        "# Leave-one-trip-out TTC validation",
        "",
        "Each fold selected guard thresholds on five trips and evaluated once on",
        "the untouched sixth trip. Selection maximized danger-F1 while requiring",
        "training composite >= 28.7 and critical-TTC MAE no worse than the current",
        "guard on those same five training trips.",
        "",
        "## Result",
        "",
        f"- Held-out macro F1: `{summary['heldout_macro_f1']:.3f}`",
        f"- Held-out composite: `{summary['heldout_composite']:.1f}`",
        f"- Held-out critical-TTC MAE: `{summary['heldout_critical_mae']:.3f} s`",
        f"- Mean train-minus-held-out F1 gap: `{gap:.3f}` ({interpretation})",
        "",
        "![Training versus held-out F1](loto_generalization.png)",
        "",
        "## Fold results",
        "",
        "| Held-out trip | Training F1 | Held-out F1 | F1 gap | Held-out composite | Held-out critical MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fold in summary["folds"]:
        held = fold["held_out"]
        train = fold["training"]
        lines.append(
            f"| {fold['held_out_trip']} | {train['f1']:.3f} | "
            f"{held['f1']:.3f} | {train['f1'] - held['f1']:.3f} | "
            f"{held['composite']:.1f} | {held['mae_critical']:.3f} s |"
        )
    lines.extend(
        [
            "",
            "## Final all-trip configuration",
            "",
            "This configuration is trained on all six practice trips and is intended",
            "for the next external evaluation. Its all-trip score is training evidence,",
            "not an unbiased validation score.",
            "",
            "```json",
            json.dumps(summary["final_all_trip_config"], indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=Path("Package_starterkit/package_starterkit"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    print("Loading frozen candidate traces...", flush=True)
    trips = [load_trip(args.source_root, trip_id) for trip_id in TRIPS]
    configs = guard_grid()
    print(f"Scoring {len(configs)} physical guard configurations...", flush=True)
    metrics: list[list[Metrics]] = [
        [score(predict(data, config), data.ground_truth) for data in trips]
        for config in configs
    ]
    current_index = current_guard_index(configs)
    folds: list[dict] = []
    heldout_predictions: dict[str, np.ndarray] = {}

    for heldout_index, heldout in enumerate(trips):
        training_indices = [index for index in range(len(trips)) if index != heldout_index]
        current_training_mae = float(
            np.mean(
                [metrics[current_index][index].mae_critical for index in training_indices]
            )
        )
        feasible = []
        for config_index, config in enumerate(configs):
            training_metrics = [metrics[config_index][index] for index in training_indices]
            mean_f1 = float(np.mean([item.f1 for item in training_metrics]))
            mean_composite = float(
                np.mean([item.composite for item in training_metrics])
            )
            mean_mae = float(
                np.mean([item.mae_critical for item in training_metrics])
            )
            if (
                mean_composite >= MINIMUM_COMPOSITE
                and mean_mae <= current_training_mae + 1e-9
            ):
                feasible.append(
                    (
                        mean_f1,
                        mean_composite,
                        -mean_mae,
                        -configuration_distance(config),
                        -config_index,
                    )
                )
        if not feasible:
            raise RuntimeError(f"no safeguard-valid configuration for {heldout.trip_id}")
        selected_index = -max(feasible)[-1]
        selected = configs[selected_index]
        training_metrics = [
            metrics[selected_index][index] for index in training_indices
        ]
        heldout_metric = metrics[selected_index][heldout_index]
        heldout_predictions[heldout.trip_id] = predict(heldout, selected)
        fold = {
            "held_out_trip": heldout.trip_id,
            "selected_config": asdict(selected),
            "training": {
                "f1": float(np.mean([item.f1 for item in training_metrics])),
                "composite": float(
                    np.mean([item.composite for item in training_metrics])
                ),
                "mae_critical": float(
                    np.mean([item.mae_critical for item in training_metrics])
                ),
            },
            "held_out": asdict(heldout_metric),
        }
        folds.append(fold)
        print(
            f"{heldout.trip_id}: train F1={fold['training']['f1']:.3f}, "
            f"held-out F1={heldout_metric.f1:.3f}",
            flush=True,
        )

    all_current_mae = float(
        np.mean([item.mae_critical for item in metrics[current_index]])
    )
    full_feasible = []
    for config_index, config in enumerate(configs):
        items = metrics[config_index]
        mean_f1 = float(np.mean([item.f1 for item in items]))
        mean_composite = float(np.mean([item.composite for item in items]))
        mean_mae = float(np.mean([item.mae_critical for item in items]))
        if mean_composite >= MINIMUM_COMPOSITE and mean_mae <= all_current_mae + 1e-9:
            full_feasible.append(
                (
                    mean_f1,
                    mean_composite,
                    -mean_mae,
                    -configuration_distance(config),
                    -config_index,
                )
            )
    final_index = -max(full_feasible)[-1]
    final_config = configs[final_index]

    predictions_root = args.output_root / "heldout_predictions"
    for data in trips:
        write_predictions(
            predictions_root / f"{data.trip_id}.csv",
            data,
            heldout_predictions[data.trip_id],
        )

    sys.path.insert(0, str(args.starter_root.resolve()))
    from team_kit.evaluation import evaluate

    official_report = evaluate(
        predictions_root,
        args.practice_root,
        args.output_root / "heldout_evaluation.json",
    )
    gaps = [
        fold["training"]["f1"] - fold["held_out"]["f1"]
        for fold in folds
    ]
    summary = {
        "protocol": "leave-one-trip-out-v1",
        "selection_objective": "maximum training macro danger-F1",
        "safeguards": {
            "minimum_training_composite": MINIMUM_COMPOSITE,
            "maximum_training_critical_mae": (
                "current guard on the same five training trips"
            ),
        },
        "candidate_count": len(configs),
        "folds": folds,
        "heldout_macro_f1": official_report.overall_f1,
        "heldout_composite": official_report.overall_composite_score,
        "heldout_critical_mae": official_report.overall_mae_critical,
        "mean_train_minus_heldout_f1_gap": float(np.mean(gaps)),
        "final_all_trip_config": asdict(final_config),
        "final_all_trip_training_metrics": {
            "f1": float(np.mean([item.f1 for item in metrics[final_index]])),
            "composite": float(
                np.mean([item.composite for item in metrics[final_index]])
            ),
            "mae_critical": float(
                np.mean([item.mae_critical for item in metrics[final_index]])
            ),
        },
    }
    (args.output_root / "loto_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_root / "loto_folds.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "held_out_trip",
                "training_f1",
                "heldout_f1",
                "training_composite",
                "heldout_composite",
                "training_mae_critical",
                "heldout_mae_critical",
                "f1_gap",
            ),
        )
        writer.writeheader()
        for fold in folds:
            writer.writerow(
                {
                    "held_out_trip": fold["held_out_trip"],
                    "training_f1": fold["training"]["f1"],
                    "heldout_f1": fold["held_out"]["f1"],
                    "training_composite": fold["training"]["composite"],
                    "heldout_composite": fold["held_out"]["composite"],
                    "training_mae_critical": fold["training"]["mae_critical"],
                    "heldout_mae_critical": fold["held_out"]["mae_critical"],
                    "f1_gap": fold["training"]["f1"] - fold["held_out"]["f1"],
                }
            )
    write_chart(folds, args.output_root / "loto_generalization.png")
    write_markdown(summary, args.output_root / "LOTO_REPORT.md")
    print(
        f"RESULT held-out macro F1={official_report.overall_f1:.3f}, "
        f"composite={official_report.overall_composite_score:.1f}, "
        f"critical MAE={official_report.overall_mae_critical:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
