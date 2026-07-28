"""Diagnose whether existing causal features can explain T03/T05 errors.

This is intentionally an overfit/capacity experiment, not a promotion
benchmark.  It reports:

1. deliberate in-sample fit with a high-capacity small decision tree; and
2. four-fold blocked validation with a fixed shallow tree.

Frame ID, timestamp, trip identity, and ground truth are never model inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


WINDOWS = {
    "T03-Sample": (280, 360),
    "T05-Sample": (430, 580),
}

FEATURE_NAMES = (
    "candidate_count",
    "finite_ttc_count",
    "min_candidate_ttc",
    "min_depth_m",
    "max_closing_speed_mps",
    "min_motion_residual_m",
    "max_confidence",
    "max_bottom_y_norm",
    "max_width_norm",
    "max_height_norm",
    "max_track_hits",
    "max_history_length",
    "max_observation_quality",
    "min_depth_mad_ratio",
    "max_lr_support",
    "max_corridor_overlap",
    "urgent_ttc",
    "urgent_depth_m",
    "urgent_closing_speed_mps",
    "urgent_motion_residual_m",
    "urgent_confidence",
    "urgent_center_offset",
    "urgent_bottom_y_norm",
    "urgent_width_norm",
    "urgent_height_norm",
    "urgent_track_hits",
    "urgent_history_length",
    "urgent_observation_quality",
    "urgent_depth_mad_ratio",
    "urgent_lr_support",
    "urgent_corridor_overlap",
)


@dataclass(frozen=True)
class ClassificationMetrics:
    f1: float
    precision: float
    recall: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass
class TreeNode:
    prediction: bool
    positive_count: int
    sample_count: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


def _parse_float(value: str | None, *, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _parse_ttc(value: str | None) -> float:
    return _parse_float(value, fallback=99.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def load_ground_truth(
    practice_root: Path,
    trip_id: str,
) -> dict[int, float]:
    path = practice_root / trip_id / f"{trip_id}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    return {
        int(frame["frame_id"]): float(frame["min_ttc"])
        for frame in document["frames"]
    }


def load_baseline(
    predictions_root: Path,
    trip_id: str,
) -> dict[int, float]:
    with (predictions_root / f"{trip_id}.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return {
            int(row["frame_id"]): _parse_ttc(row["predicted_ttc"])
            for row in csv.DictReader(handle)
        }


def load_candidates(
    candidate_root: Path,
    trip_id: str,
) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with (candidate_root / f"{trip_id}.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["frame_id"])].append(row)
    return grouped


def _row_value(row: dict[str, str], name: str, fallback: float = 0.0) -> float:
    return _parse_float(row.get(name), fallback=fallback)


def aggregate_frame(rows: Sequence[dict[str, str]]) -> np.ndarray:
    """Create causal frame features without identifiers or label information."""
    if not rows:
        return np.asarray(
            [
                0,
                0,
                99,
                99,
                0,
                20,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
                99,
                99,
                0,
                20,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                0,
                0,
            ],
            dtype=float,
        )

    ttc = np.asarray([_parse_ttc(row.get("candidate_ttc")) for row in rows])
    depth = np.asarray(
        [_clamp(_row_value(row, "depth_m", 99.0), 0.0, 99.0) for row in rows]
    )
    closing = np.asarray(
        [
            _clamp(_row_value(row, "closing_speed_mps"), -200.0, 200.0)
            for row in rows
        ]
    )
    residual = np.asarray(
        [
            _clamp(_row_value(row, "motion_residual_m", 20.0), 0.0, 20.0)
            for row in rows
        ]
    )
    confidence = np.asarray([_row_value(row, "confidence") for row in rows])
    bottom = np.asarray(
        [_row_value(row, "selected_bottom_y_norm") for row in rows]
    )
    width = np.asarray(
        [_row_value(row, "selected_width_norm") for row in rows]
    )
    height = np.asarray(
        [_row_value(row, "selected_height_norm") for row in rows]
    )
    hits = np.asarray([_row_value(row, "selected_track_hits") for row in rows])
    history = np.asarray(
        [_row_value(row, "selected_history_length") for row in rows]
    )
    quality = np.asarray(
        [_row_value(row, "selected_observation_quality") for row in rows]
    )
    mad_ratio = np.asarray(
        [
            _clamp(
                _row_value(row, "selected_depth_mad_ratio", 10.0),
                0.0,
                10.0,
            )
            for row in rows
        ]
    )
    lr_support = np.asarray(
        [_row_value(row, "selected_lr_support") for row in rows]
    )
    corridor = np.asarray(
        [_row_value(row, "selected_corridor_overlap") for row in rows]
    )

    urgent_index = int(np.argmin(ttc))
    urgent = rows[urgent_index]
    urgent_values = (
        ttc[urgent_index],
        depth[urgent_index],
        closing[urgent_index],
        residual[urgent_index],
        confidence[urgent_index],
        abs(_row_value(urgent, "selected_center_x_norm", 0.5) - 0.5),
        bottom[urgent_index],
        width[urgent_index],
        height[urgent_index],
        hits[urgent_index],
        history[urgent_index],
        quality[urgent_index],
        mad_ratio[urgent_index],
        lr_support[urgent_index],
        corridor[urgent_index],
    )
    values = (
        len(rows),
        int(np.count_nonzero(ttc < 99.0)),
        float(np.min(ttc)),
        float(np.min(depth)),
        float(np.max(closing)),
        float(np.min(residual)),
        float(np.max(confidence)),
        float(np.max(bottom)),
        float(np.max(width)),
        float(np.max(height)),
        float(np.max(hits)),
        float(np.max(history)),
        float(np.max(quality)),
        float(np.min(mad_ratio)),
        float(np.max(lr_support)),
        float(np.max(corridor)),
        *urgent_values,
    )
    return np.asarray(values, dtype=float)


def load_window(
    practice_root: Path,
    candidate_root: Path,
    baseline_root: Path,
    trip_id: str,
    first_frame: int,
    last_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ground_truth = load_ground_truth(practice_root, trip_id)
    baseline = load_baseline(baseline_root, trip_id)
    candidates = load_candidates(candidate_root, trip_id)
    frame_ids = np.arange(first_frame, last_frame + 1, dtype=int)
    features = np.vstack(
        [aggregate_frame(candidates.get(int(frame_id), ())) for frame_id in frame_ids]
    )
    labels = np.asarray(
        [ground_truth[int(frame_id)] < 2.0 for frame_id in frame_ids],
        dtype=bool,
    )
    baseline_labels = np.asarray(
        [baseline[int(frame_id)] < 2.0 for frame_id in frame_ids],
        dtype=bool,
    )
    return frame_ids, features, labels, baseline_labels


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> ClassificationMetrics:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    tp = int(np.count_nonzero(labels & predictions))
    fp = int(np.count_nonzero(~labels & predictions))
    fn = int(np.count_nonzero(labels & ~predictions))
    tn = int(np.count_nonzero(~labels & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / labels.size if labels.size else 0.0
    return ClassificationMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def _gini(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    positive = float(np.mean(labels))
    return 1.0 - positive * positive - (1.0 - positive) ** 2


def _thresholds(values: np.ndarray, maximum: int = 64) -> np.ndarray:
    unique = np.unique(values)
    if unique.size < 2:
        return np.empty(0, dtype=float)
    thresholds = (unique[:-1] + unique[1:]) / 2.0
    if thresholds.size <= maximum:
        return thresholds
    indices = np.unique(
        np.linspace(0, thresholds.size - 1, maximum, dtype=int)
    )
    return thresholds[indices]


def fit_tree(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    maximum_depth: int,
    minimum_leaf: int,
    depth: int = 0,
) -> TreeNode:
    positive_count = int(np.count_nonzero(labels))
    node = TreeNode(
        prediction=positive_count * 2 >= labels.size,
        positive_count=positive_count,
        sample_count=int(labels.size),
    )
    if (
        depth >= maximum_depth
        or labels.size < minimum_leaf * 2
        or positive_count in {0, labels.size}
    ):
        return node

    parent_cost = labels.size * _gini(labels)
    best: tuple[float, int, float, np.ndarray] | None = None
    for feature_index in range(features.shape[1]):
        values = features[:, feature_index]
        for threshold in _thresholds(values):
            left_mask = values <= threshold
            left_size = int(np.count_nonzero(left_mask))
            right_size = labels.size - left_size
            if left_size < minimum_leaf or right_size < minimum_leaf:
                continue
            cost = (
                left_size * _gini(labels[left_mask])
                + right_size * _gini(labels[~left_mask])
            )
            candidate = (float(cost), feature_index, float(threshold), left_mask)
            if best is None or candidate[:3] < best[:3]:
                best = candidate

    if best is None or best[0] >= parent_cost - 1e-12:
        return node
    _, feature_index, threshold, left_mask = best
    node.feature_index = feature_index
    node.threshold = threshold
    node.left = fit_tree(
        features[left_mask],
        labels[left_mask],
        maximum_depth=maximum_depth,
        minimum_leaf=minimum_leaf,
        depth=depth + 1,
    )
    node.right = fit_tree(
        features[~left_mask],
        labels[~left_mask],
        maximum_depth=maximum_depth,
        minimum_leaf=minimum_leaf,
        depth=depth + 1,
    )
    return node


def predict_tree(tree: TreeNode, features: np.ndarray) -> np.ndarray:
    predictions = np.empty(features.shape[0], dtype=bool)
    for row_index, row in enumerate(features):
        node = tree
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            assert node.left is not None
            assert node.right is not None
            node = (
                node.left
                if row[node.feature_index] <= node.threshold
                else node.right
            )
        predictions[row_index] = node.prediction
    return predictions


def split_usage(tree: TreeNode) -> Counter[str]:
    usage: Counter[str] = Counter()

    def visit(node: TreeNode) -> None:
        if node.is_leaf:
            return
        assert node.feature_index is not None
        assert node.left is not None
        assert node.right is not None
        usage[FEATURE_NAMES[node.feature_index]] += 1
        visit(node.left)
        visit(node.right)

    visit(tree)
    return usage


def blocked_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int = 4,
) -> tuple[np.ndarray, list[dict[str, object]], Counter[str]]:
    predictions = np.empty(labels.size, dtype=bool)
    fold_reports: list[dict[str, object]] = []
    usage: Counter[str] = Counter()
    for fold_index, test_indices in enumerate(
        np.array_split(np.arange(labels.size), folds),
        start=1,
    ):
        train_mask = np.ones(labels.size, dtype=bool)
        train_mask[test_indices] = False
        tree = fit_tree(
            features[train_mask],
            labels[train_mask],
            maximum_depth=3,
            minimum_leaf=5,
        )
        fold_prediction = predict_tree(tree, features[test_indices])
        predictions[test_indices] = fold_prediction
        usage.update(split_usage(tree))
        fold_reports.append(
            {
                "fold": fold_index,
                "first_index": int(test_indices[0]),
                "last_index": int(test_indices[-1]),
                "metrics": asdict(
                    classification_metrics(labels[test_indices], fold_prediction)
                ),
            }
        )
    return predictions, fold_reports, usage


def _write_frame_predictions(
    output: Path,
    frame_ids: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    in_sample: np.ndarray,
    blocked: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "ground_truth_danger",
                "baseline_danger",
                "in_sample_tree_danger",
                "blocked_tree_danger",
            ]
        )
        for values in zip(
            frame_ids,
            labels,
            baseline,
            in_sample,
            blocked,
            strict=True,
        ):
            writer.writerow([int(values[0]), *(int(value) for value in values[1:])])


def _format_metric(metric: dict[str, object]) -> str:
    return (
        f"F1={float(metric['f1']):.3f}, "
        f"P={float(metric['precision']):.3f}, "
        f"R={float(metric['recall']):.3f}, "
        f"TP/FP/FN={metric['tp']}/{metric['fp']}/{metric['fn']}"
    )


def _write_markdown(output: Path, report: dict[str, object]) -> None:
    lines = [
        "# Mini-fold overfit capacity diagnostic",
        "",
        "This is an intentional capacity test, not promotion evidence. Frame ID,",
        "timestamp, trip identity, and ground truth were excluded from model inputs.",
        "Blocked validation holds out four contiguous frame blocks.",
        "",
        "| Window | Baseline F1 | In-sample F1 | Blocked F1 | Diagnosis |",
        "|---|---:|---:|---:|---|",
    ]
    for trip_id, result in report["windows"].items():
        baseline = result["baseline_metrics"]
        in_sample = result["in_sample_metrics"]
        blocked = result["blocked_metrics"]
        lines.append(
            f"| {trip_id} `{result['first_frame']}-{result['last_frame']}` "
            f"| {baseline['f1']:.3f} | {in_sample['f1']:.3f} "
            f"| {blocked['f1']:.3f} | {result['diagnosis']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["conclusion"],
            "",
            "A high in-sample score with weak blocked score means the recorded",
            "signals contain local discriminatory information but do not generalize",
            "across the episode. A weak in-sample score means the present features",
            "lack enough information and the measurement method must change.",
            "",
            "## Details",
            "",
        ]
    )
    for trip_id, result in report["windows"].items():
        lines.extend(
            [
                f"### {trip_id}",
                "",
                f"- Baseline: {_format_metric(result['baseline_metrics'])}",
                f"- In-sample tree: {_format_metric(result['in_sample_metrics'])}",
                f"- Blocked tree: {_format_metric(result['blocked_metrics'])}",
                "- In-sample split features: "
                + ", ".join(
                    f"`{name}` ({count})"
                    for name, count in result["in_sample_split_usage"].items()
                ),
                "- Blocked split features: "
                + ", ".join(
                    f"`{name}` ({count})"
                    for name, count in result["blocked_split_usage"].items()
                ),
                "",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(
    practice_root: Path,
    candidate_root: Path,
    baseline_root: Path,
    output_root: Path,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    windows: dict[str, object] = {}
    for trip_id, (first_frame, last_frame) in WINDOWS.items():
        frame_ids, features, labels, baseline = load_window(
            practice_root,
            candidate_root,
            baseline_root,
            trip_id,
            first_frame,
            last_frame,
        )
        overfit_tree = fit_tree(
            features,
            labels,
            maximum_depth=8,
            minimum_leaf=1,
        )
        in_sample = predict_tree(overfit_tree, features)
        blocked, folds, blocked_usage = blocked_predictions(features, labels)
        baseline_metrics = classification_metrics(labels, baseline)
        in_sample_metrics = classification_metrics(labels, in_sample)
        blocked_metrics = classification_metrics(labels, blocked)

        if in_sample_metrics.f1 < 0.80:
            diagnosis = "feature-capacity failure"
        elif blocked_metrics.f1 < 0.60:
            diagnosis = "local signal, poor blocked generalization"
        else:
            diagnosis = "promising signal; proceed to causal ablation"

        _write_frame_predictions(
            output_root / f"{trip_id}_frame_predictions.csv",
            frame_ids,
            labels,
            baseline,
            in_sample,
            blocked,
        )
        windows[trip_id] = {
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frame_count": int(frame_ids.size),
            "danger_frames": int(np.count_nonzero(labels)),
            "feature_names": list(FEATURE_NAMES),
            "baseline_metrics": asdict(baseline_metrics),
            "in_sample_metrics": asdict(in_sample_metrics),
            "blocked_metrics": asdict(blocked_metrics),
            "in_sample_split_usage": dict(
                sorted(
                    split_usage(overfit_tree).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "blocked_split_usage": dict(
                sorted(
                    blocked_usage.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "blocked_folds": folds,
            "diagnosis": diagnosis,
        }

    capacity_pass = all(
        result["in_sample_metrics"]["f1"] >= 0.80
        for result in windows.values()
    )
    blocked_pass = all(
        result["blocked_metrics"]["f1"] >= 0.60
        for result in windows.values()
    )
    if not capacity_pass:
        conclusion = (
            "At least one error window cannot be fit to F1 0.80 from the current "
            "causal trace features. Stop threshold/model tuning and add new "
            "object-centric depth or motion measurements."
        )
    elif not blocked_pass:
        conclusion = (
            "The current signals can fit the local windows but do not generalize "
            "across contiguous held-out blocks. A larger classifier would likely "
            "overfit; implement physics-based object depth/motion features before "
            "another selection sweep."
        )
    else:
        conclusion = (
            "Both windows show usable blocked signal. Implement the dominant "
            "features causally, then test them with full six-trip LOTO."
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "purpose": "intentional mini-fold capacity diagnostic; not promotion evidence",
        "leakage_controls": [
            "no frame_id feature",
            "no timestamp feature",
            "no trip identity feature",
            "no ground-truth feature",
            "four contiguous held-out blocks",
        ],
        "models": {
            "in_sample": {"maximum_depth": 8, "minimum_leaf": 1},
            "blocked": {
                "folds": 4,
                "maximum_depth": 3,
                "minimum_leaf": 5,
            },
        },
        "windows": windows,
        "capacity_pass": capacity_pass,
        "blocked_pass": blocked_pass,
        "conclusion": conclusion,
    }
    (output_root / "minifold_capacity_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "MINIFOLD_CAPACITY_REPORT.md", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--practice-root",
        type=Path,
        default=Path("Practice_Dataset"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(
            "ai_cv/outputs/benchmarks/phase04_loto/source/track_candidates"
        ),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(
            "ai_cv/outputs/benchmarks/"
            "phase05_original_baseline_rerun/predictions"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "ai_cv/phases/05_risk_events/artifacts/minifold_overfit"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_diagnostic(
        args.practice_root,
        args.candidate_root,
        args.baseline_root,
        args.output_root,
    )
    for trip_id, result in report["windows"].items():
        print(
            f"{trip_id}: baseline F1="
            f"{result['baseline_metrics']['f1']:.3f}, "
            f"in-sample F1={result['in_sample_metrics']['f1']:.3f}, "
            f"blocked F1={result['blocked_metrics']['f1']:.3f}; "
            f"{result['diagnosis']}"
        )
    print(report["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
