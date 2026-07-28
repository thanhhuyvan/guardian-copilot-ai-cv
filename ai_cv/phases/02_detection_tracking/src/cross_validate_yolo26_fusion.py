"""Leave-one-trip-out 6-fold cross-validation for Phase 04B YOLO26 Semantic Fusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from cross_validate_guarded_ttc import (
    CURRENT_GUARD,
    GuardConfig,
    Metrics,
    TripData,
    corridor_membership,
    load_trip,
    score,
)
from detector_interfaces import Detection
from semantic_fusion import (
    SemanticAssociation,
    TemporalSemanticState,
    associate_component_with_detections,
)


TRIPS = tuple(f"T0{index}-Sample" for index in range(1, 7))

# A fold is infeasible when none of the 27 policies satisfies the composite
# and MAE constraints on the 5 training trips.  We record this explicitly
# rather than silently inserting a default config.
_INFEASIBLE_SENTINEL = "INFEASIBLE"


@dataclass(frozen=True)
class SemanticConfig:
    semantic_score_threshold: float  # [0.20, 0.25, 0.30]
    consecutive_misses: int          # [2, 3, 4]
    close_fallback_depth_m: float    # [4.0, 5.0, 6.0]


def load_detections_csv(csv_path: Path) -> dict[int, list[Detection]]:
    """Load pre-computed YOLO detections per frame_id."""
    detections_by_frame: dict[int, list[Detection]] = {}
    if not csv_path.is_file():
        return detections_by_frame

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row["frame_id"])
            det = Detection(
                bbox_xyxy=(
                    float(row["x0"]),
                    float(row["y0"]),
                    float(row["x1"]),
                    float(row["y1"]),
                ),
                class_id=int(row["class_id"]),
                class_name=row["class_name"],
                confidence=float(row["confidence"]),
            )
            detections_by_frame.setdefault(frame_id, []).append(det)

    return detections_by_frame


def load_candidate_extras(candidate_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load track_id and selected_width_norm arrays from a candidate CSV.

    These columns exist in the candidate CSV but are not exposed by the shared
    TripData / load_trip in cross_validate_guarded_ttc.py.  We load them here
    so the frozen shared module is not modified.

    Returns:
        track_ids  – int array, one entry per candidate row.
        widths     – float array of selected_width_norm, one entry per candidate row.
    """
    track_ids = []
    widths = []
    with candidate_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            track_ids.append(int(row["track_id"]))
            widths.append(float(row["selected_width_norm"]))
    return np.asarray(track_ids, dtype=int), np.asarray(widths, dtype=float)


def predict_with_semantic_fusion(
    data: TripData,
    guard_config: GuardConfig,
    semantic_config: SemanticConfig | None,
    detections_by_frame: dict[int, list[Detection]],
    track_ids: np.ndarray | None = None,
    widths: np.ndarray | None = None,
    image_shape: tuple[int, int] = (360, 640),
) -> tuple[np.ndarray, list[dict]]:
    """
    Run candidate track filtering with temporal semantic fusion.

    Temporal state is keyed by ``track_id`` so consecutive-miss counts
    accumulate correctly across frames.  The component bounding box is
    reconstructed using both ``selected_width_norm`` and
    ``selected_height_norm`` from the candidate row.

    Returns predicted per-frame TTC array and per-candidate diagnostics.
    """
    accepted_candidates = []
    diagnostics = []

    # Sort candidates by frame_id so each track's observations arrive in order.
    sorted_candidate_indices = np.argsort(data.candidate_frame_index)

    # Per-track semantic state keyed by track_id (persistent across frames).
    # Bug fix: was previously keyed by row index `idx` which resets each frame,
    # preventing consecutive-miss accumulation entirely.
    track_states: dict[int, TemporalSemanticState] = {}

    for idx in sorted_candidate_indices:
        frame_idx = data.candidate_frame_index[idx]
        frame_id = int(data.frame_ids[frame_idx])

        # --- Candidate physical attributes ---
        # Use pre-loaded track_id array; fall back to row index only as last resort.
        track_id = int(track_ids[idx]) if track_ids is not None else int(idx)
        cx = data.center_x[idx]
        by = data.bottom_y[idx]
        # Bug fix: use actual width norm, not height norm for both dimensions.
        w = float(widths[idx]) if widths is not None else data.height[idx]
        h = data.height[idx]
        conf = data.confidence[idx]
        speed = data.closing_speed[idx]
        depth = data.depth[idx]
        residual = data.residual[idx]
        candidate_ttc = data.ttc[idx]
        conf = data.confidence[idx]
        speed = data.closing_speed[idx]
        depth = data.depth[idx]
        residual = data.residual[idx]
        candidate_ttc = data.ttc[idx]

        # --- Physical corridor & threshold check ---
        in_corridor = corridor_membership(
            np.array([cx]),
            np.array([by]),
            guard_config.corridor_top_width,
            guard_config.corridor_bottom_width,
        )[0]

        physically_accepted = (
            in_corridor
            and (by >= guard_config.minimum_bottom)
            and (h >= guard_config.minimum_height)
            and (conf >= guard_config.minimum_confidence)
            and (speed <= guard_config.maximum_closing_speed_mps)
            and (depth <= guard_config.maximum_depth_m)
            and (residual <= guard_config.maximum_motion_residual_m)
            and np.isfinite(candidate_ttc)
        )

        suppressed_by_semantics = False

        if semantic_config is not None:
            img_h, img_w = image_shape

            # Reconstruct component bbox in pixels using correct width and height.
            comp_w_px = max(5, int(w * img_w))
            comp_h_px = max(5, int(h * img_h))
            comp_cx_px = int(cx * img_w)
            comp_by_px = int(by * img_h)
            comp_x0 = max(0, comp_cx_px - comp_w_px // 2)
            comp_y0 = max(0, comp_by_px - comp_h_px)
            comp_x1 = min(img_w, comp_cx_px + comp_w_px // 2)
            comp_y1 = min(img_h, comp_by_px)
            comp_bbox = (comp_x0, comp_y0, comp_x1, comp_y1)

            frame_dets = detections_by_frame.get(frame_id, [])
            assoc = associate_component_with_detections(
                comp_bbox, frame_dets, image_shape
            )

            # Key state by track_id — persists across frames for the same track.
            if track_id not in track_states:
                track_states[track_id] = TemporalSemanticState()

            sem_state = track_states[track_id]
            sem_state.update(assoc)

            suppressed_by_semantics = sem_state.is_suppressed(
                latest_depth_m=depth,
                score_threshold=semantic_config.semantic_score_threshold,
                max_misses=semantic_config.consecutive_misses,
                fallback_depth_m=semantic_config.close_fallback_depth_m,
            )

            diagnostics.append(
                {
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "candidate_idx": int(idx),
                    "depth_m": float(depth),
                    "matched": assoc.matched,
                    "matched_class": assoc.class_name,
                    "semantic_score": float(sem_state.score),
                    "misses": sem_state.consecutive_misses,
                    "suppressed": suppressed_by_semantics,
                }
            )

        if physically_accepted and not suppressed_by_semantics:
            accepted_candidates.append((frame_idx, candidate_ttc))

    predictions = np.full(data.frame_ids.size, math.inf, dtype=float)
    for frame_idx, candidate_ttc in accepted_candidates:
        if candidate_ttc < predictions[frame_idx]:
            predictions[frame_idx] = candidate_ttc

    return predictions, diagnostics


def run_6fold_cross_validation(
    source_root: Path,
    detections_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = GuardConfig(**CURRENT_GUARD)

    print("Loading trip data and detector CSVs...")
    trips_data: dict[str, TripData] = {}
    trip_detections: dict[str, dict[int, list[Detection]]] = {}
    trip_track_ids: dict[str, np.ndarray] = {}
    trip_widths: dict[str, np.ndarray] = {}

    for trip_id in TRIPS:
        trips_data[trip_id] = load_trip(source_root, trip_id)
        det_path = detections_dir / f"{trip_id}.csv"
        trip_detections[trip_id] = load_detections_csv(det_path)
        cand_path = source_root / "track_candidates" / f"{trip_id}.csv"
        t_ids, t_widths = load_candidate_extras(cand_path)
        trip_track_ids[trip_id] = t_ids
        trip_widths[trip_id] = t_widths

    # 27 search candidates
    grid = [
        SemanticConfig(score_thresh, misses, depth)
        for score_thresh in (0.20, 0.25, 0.30)
        for misses in (2, 3, 4)
        for depth in (4.0, 5.0, 6.0)
    ]

    print(f"Running 6-fold leave-one-trip-out search over {len(grid)} hyperparameter policies...")

    # Physical guard baseline on every trip (no semantics).
    baseline_metrics: dict[str, Metrics] = {}
    for trip_id, data in trips_data.items():
        preds, _ = predict_with_semantic_fusion(
            data, guard, semantic_config=None, detections_by_frame={},
            track_ids=trip_track_ids[trip_id], widths=trip_widths[trip_id],
        )
        baseline_metrics[trip_id] = score(preds, data.ground_truth)

    fold_results = []

    for fold_idx, val_trip_id in enumerate(TRIPS):
        train_trip_ids = [t for t in TRIPS if t != val_trip_id]
        print(f"\n--- Fold {fold_idx + 1}/6: Holdout = {val_trip_id} ---")

        best_train_f1 = -1.0
        best_config: SemanticConfig | None = None
        best_train_metrics: dict | None = None

        for sem_cfg in grid:
            train_f1s, train_composites, train_maes, train_recalls = [], [], [], []

            for t_id in train_trip_ids:
                data = trips_data[t_id]
                dets = trip_detections[t_id]
                preds, _ = predict_with_semantic_fusion(
                    data, guard, sem_cfg, dets,
                    track_ids=trip_track_ids[t_id], widths=trip_widths[t_id],
                )
                m = score(preds, data.ground_truth)
                train_f1s.append(m.f1)
                train_composites.append(m.composite)
                train_maes.append(m.mae_critical)
                train_recalls.append(m.recall)

            mean_train_f1   = float(np.mean(train_f1s))
            mean_train_comp = float(np.mean(train_composites))
            mean_train_mae  = float(np.mean(train_maes))
            mean_train_recall = float(np.mean(train_recalls))

            base_train_mae = float(
                np.mean([baseline_metrics[t_id].mae_critical for t_id in train_trip_ids])
            )

            valid = (mean_train_comp >= 38.4) and (mean_train_mae <= base_train_mae + 1e-4)

            if valid and mean_train_f1 > best_train_f1:
                best_train_f1 = mean_train_f1
                best_config = sem_cfg
                best_train_metrics = {
                    "f1": mean_train_f1,
                    "composite": mean_train_comp,
                    "mae_critical": mean_train_mae,
                    "recall": mean_train_recall,
                }

        # Fix 6: infeasible fold — no config met both constraints.
        # Record explicitly; do NOT silently fall back to a default.
        if best_config is None:
            print(f"  WARNING: Fold {fold_idx + 1} is INFEASIBLE — no policy met composite>=38.4 and MAE constraints on train trips.")
            fold_results.append(
                {
                    "fold": fold_idx + 1,
                    "validation_trip": val_trip_id,
                    "selected_config": _INFEASIBLE_SENTINEL,
                    "train_metrics": None,
                    "val_metrics": None,
                    "infeasible": True,
                }
            )
            continue

        # Evaluate selected config on the untouched held-out validation trip.
        val_data = trips_data[val_trip_id]
        val_dets = trip_detections[val_trip_id]
        val_preds, val_diag = predict_with_semantic_fusion(
            val_data, guard, best_config, val_dets,
            track_ids=trip_track_ids[val_trip_id], widths=trip_widths[val_trip_id],
        )
        val_metric = score(val_preds, val_data.ground_truth)

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "validation_trip": val_trip_id,
                "selected_config": asdict(best_config),
                "train_metrics": best_train_metrics,
                "val_metrics": {
                    "f1": val_metric.f1,
                    "composite": val_metric.composite,
                    "mae_critical": val_metric.mae_critical,
                    "precision": val_metric.precision,
                    "recall": val_metric.recall,
                    "tp": val_metric.tp,
                    "fp": val_metric.fp,
                    "fn": val_metric.fn,
                },
                "infeasible": False,
            }
        )

        print(
            f"  Config={asdict(best_config)}  "
            f"Val F1={val_metric.f1:.4f}, Composite={val_metric.composite:.2f}, "
            f"MAE={val_metric.mae_critical:.3f}s, FP={val_metric.fp}, Recall={val_metric.recall:.3f}"
        )

    # Only include feasible folds in macro averages.
    feasible = [f for f in fold_results if not f.get("infeasible", False)]
    infeasible_count = len(fold_results) - len(feasible)

    if not feasible:
        print("\nAll folds infeasible — cannot compute macro summary.")
        report = {"macro_summary": None, "fold_details": fold_results, "infeasible_folds": infeasible_count}
        with (output_dir / "loto_yolo26_fusion_report.json").open("w") as f:
            json.dump(report, f, indent=2)
        return

    val_f1s        = [f["val_metrics"]["f1"]        for f in feasible]
    val_composites = [f["val_metrics"]["composite"]  for f in feasible]
    val_maes       = [f["val_metrics"]["mae_critical"] for f in feasible]

    t05_fp_entry = next((f for f in feasible if f["validation_trip"] == "T05-Sample"), None)
    t03_entry    = next((f for f in feasible if f["validation_trip"] == "T03-Sample"), None)
    val_t05_fp   = t05_fp_entry["val_metrics"]["fp"]     if t05_fp_entry else None
    val_t03_recall = t03_entry["val_metrics"]["recall"]  if t03_entry else None

    macro_summary = {
        "feasible_folds": len(feasible),
        "infeasible_folds": infeasible_count,
        "macro_danger_f1":    float(np.mean(val_f1s)),
        "macro_composite":    float(np.mean(val_composites)),
        "macro_critical_mae": float(np.mean(val_maes)),
        "t05_false_positives": val_t05_fp,
        "t03_recall":          val_t03_recall,
        "gates_passed": {
            "macro_f1_ge_0_60":        bool(np.mean(val_f1s) >= 0.60),
            "macro_composite_ge_38_4": bool(np.mean(val_composites) >= 38.4),
            "macro_mae_le_46_638":     bool(np.mean(val_maes) <= 46.638),
            "t05_fp_le_20":            bool(val_t05_fp <= 20) if val_t05_fp is not None else None,
            "t03_recall_ge_0_276":     bool(val_t03_recall >= 0.276) if val_t03_recall is not None else None,
        },
    }

    report = {"macro_summary": macro_summary, "fold_details": fold_results}
    report_path = output_dir / "loto_yolo26_fusion_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n================ Leave-One-Trip-Out Summary ================")
    print(f"Feasible folds:         {len(feasible)}/6  (infeasible: {infeasible_count})")
    print(f"Macro Danger F1:        {macro_summary['macro_danger_f1']:.4f} (Target: >= 0.60)")
    print(f"Macro Composite:        {macro_summary['macro_composite']:.2f}  (Target: >= 38.4)")
    print(f"Macro Critical MAE:     {macro_summary['macro_critical_mae']:.3f} s (Target: <= 46.638 s)")
    if val_t05_fp is not None:
        print(f"T05 False Positives:    {val_t05_fp}      (Target: <= 20)")
    else:
        print(f"T05 False Positives:    N/A (fold infeasible)")
    if val_t03_recall is not None:
        print(f"T03 Recall:             {val_t03_recall:.3f}   (Target: >= 0.276)")
    else:
        print(f"T03 Recall:             N/A (fold infeasible)")
    print(f"Report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 6-fold LOTO for YOLO26 Semantic Fusion.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("ai_cv/phases/02_detection_tracking/artifacts"),
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference/detections"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_loto"),
    )
    args = parser.parse_args()

    run_6fold_cross_validation(
        source_root=args.source_root,
        detections_dir=args.detections_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
