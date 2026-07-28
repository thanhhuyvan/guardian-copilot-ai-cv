"""Reproducible per-config-per-trip sweep for Phase 04B YOLO26 Semantic Fusion.

This script runs the frozen 27-policy semantic grid on each of the six
practice trips and writes a single CSV with one row per (configuration, trip):

    configuration, trip, tp, fp, fn, precision, recall, f1,
    composite, mae_critical, suppressed_candidates

It then computes four selection modes from those rows so the accuracy ceiling
is reproducible before any fine-tuning decision:

    baseline       -- physical guard only, no semantics (regression reference)
    global_best    -- one config applied to all trips, chosen by macro F1
    loto           -- proper 6-fold leave-one-trip-out selection (config chosen
                      on 5 train trips, evaluated on the held-out trip)
    oracle         -- per-trip best choice among baseline and all semantic
                      configs (upper bound; not selectable)

The macro F1 upper bound is reported correctly: it is the mean of per-trip F1
under the oracle selection, including the option to disable semantics on trips
where every semantic config regresses. The physical baseline macro is reported
on the same trips for a fair comparison.

Usage:
    python ai_cv/phases/02_detection_tracking/src/sweep_yolo26_fusion.py \
        --source-root ai_cv/outputs/benchmarks/phase04_loto/source \
        --detections-dir ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference/detections \
        --output-dir ai_cv/phases/02_detection_tracking/artifacts/yolo26_loto
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cross_validate_guarded_ttc import CURRENT_GUARD, GuardConfig, TripData, load_trip, score
from cross_validate_yolo26_fusion import (
    TRIPS,
    SemanticConfig,
    load_candidate_extras,
    load_detections_csv,
    predict_with_semantic_fusion,
)


# Frozen 27-policy grid defined by the plan (kept identical to LOTO).
SEMANTIC_GRID: tuple[SemanticConfig, ...] = tuple(
    SemanticConfig(score_thresh, misses, depth)
    for score_thresh in (0.20, 0.25, 0.30)
    for misses in (2, 3, 4)
    for depth in (4.0, 5.0, 6.0)
)


def config_label(cfg: SemanticConfig) -> str:
    """Compact stable label for a SemanticConfig, e.g. 's0.25_m3_d5.0'."""
    return f"s{cfg.semantic_score_threshold}_m{cfg.consecutive_misses}_d{cfg.close_fallback_depth_m}"


def run_one(
    data: TripData,
    guard: GuardConfig,
    semantic_cfg: SemanticConfig | None,
    dets: dict[int, list],
    track_ids: np.ndarray,
    widths: np.ndarray,
) -> tuple[float, float, float, int, int, int, float, int]:
    """Run fusion for one (config, trip) and return the requested metric tuple.

    Returns (f1, precision, recall, tp, fp, fn, composite, mae_critical,
    suppressed). When semantic_cfg is None the physical guard is run with no
    detections; suppression count is 0 by definition.
    """
    preds, diag = predict_with_semantic_fusion(
        data, guard, semantic_cfg, dets, track_ids=track_ids, widths=widths
    )
    m = score(preds, data.ground_truth)
    suppressed = sum(1 for d in diag if d["suppressed"]) if semantic_cfg is not None else 0
    return (
        m.f1, m.precision, m.recall, m.tp, m.fp, m.fn,
        m.composite, m.mae_critical, suppressed,
    )


def run_full_sweep(
    source_root: Path,
    detections_dir: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, dict[Path, dict[Path, np.ndarray]]]]:
    """Run baseline + 27 configs x 6 trips; write the sweep CSV.

    Returns (csv_path, results) where results is nested dicts:
        results[config_label][trip_id] = np.array(
            [f1, prec, rec, tp, fp, fn, composite, mae, suppressed]
        )
    Only config rows are stored; baseline is returned separately.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = GuardConfig(**CURRENT_GUARD)

    trips_data: dict[str, TripData] = {}
    trip_dets: dict[str, dict[int, list]] = {}
    trip_track_ids: dict[str, np.ndarray] = {}
    trip_widths: dict[str, np.ndarray] = {}

    for trip_id in TRIPS:
        trips_data[trip_id] = load_trip(source_root, trip_id)
        trip_dets[trip_id] = load_detections_csv(detections_dir / f"{trip_id}.csv")
        t_ids, t_widths = load_candidate_extras(source_root / "track_candidates" / f"{trip_id}.csv")
        trip_track_ids[trip_id] = t_ids
        trip_widths[trip_id] = t_widths

    csv_path = output_dir / "sweep_per_config_per_trip.csv"
    print(f"Writing sweep CSV to {csv_path} ...")
    print(
        f"  Configurations: 1 baseline + {len(SEMANTIC_GRID)} semantic "
        f"= {1 + len(SEMANTIC_GRID)} rows/trip, "
        f"{(1 + len(SEMANTIC_GRID)) * len(TRIPS)} total rows"
    )

    results: dict[str, dict[str, np.ndarray]] = {}
    baseline: dict[str, np.ndarray] = {}

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "configuration", "trip",
            "tp", "fp", "fn",
            "precision", "recall", "f1",
            "composite", "mae_critical",
            "suppressed_candidates",
        ])

        # Baseline row per trip (config label "baseline"): physical guard only.
        for trip_id in TRIPS:
            data = trips_data[trip_id]
            f1, prec, rec, tp, fp, fn, comp, mae, sup = run_one(
                data, guard, None, {}, trip_track_ids[trip_id], trip_widths[trip_id]
            )
            baseline[trip_id] = np.array([f1, prec, rec, tp, fp, fn, comp, mae, sup])
            writer.writerow([
                "baseline", trip_id, tp, fp, fn,
                f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}",
                f"{comp:.6f}", f"{mae:.6f}", sup,
            ])

        # 27 semantic configs per trip.
        for cfg in SEMANTIC_GRID:
            label = config_label(cfg)
            results[label] = {}
            for trip_id in TRIPS:
                data = trips_data[trip_id]
                f1, prec, rec, tp, fp, fn, comp, mae, sup = run_one(
                    data, guard, cfg, trip_dets[trip_id],
                    trip_track_ids[trip_id], trip_widths[trip_id],
                )
                row = np.array([f1, prec, rec, tp, fp, fn, comp, mae, sup])
                results[label][trip_id] = row
                writer.writerow([
                    label, trip_id, tp, fp, fn,
                    f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}",
                    f"{comp:.6f}", f"{mae:.6f}", sup,
                ])

    print(f"  Done. {csv_path.stat().st_size} bytes.")
    return csv_path, {"results": results, "baseline": baseline}


# --------------------------------------------------------------------------- helpers


def _macro(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _row_get(row: np.ndarray, name: str) -> float:
    """Map a metric name to a column index in the 9-element result row."""
    cols = ["f1", "precision", "recall", "tp", "fp", "fn", "composite", "mae", "suppressed"]
    return float(row[cols.index(name)])


def compute_selection_modes(
    sweep: dict,
) -> dict:
    """Compute baseline, global_best, loto, oracle from the sweep results.

    The macro F1 upper bound is the mean of per-trip F1 under oracle selection.
    """
    results: dict[str, dict[str, np.ndarray]] = sweep["results"]
    baseline: dict[str, np.ndarray] = sweep["baseline"]

    # -------- baseline macro (physical guard only) ----------------------------
    base_f1 = [baseline[t][0] for t in TRIPS]
    base_comp = [baseline[t][6] for t in TRIPS]
    base_mae = [baseline[t][7] for t in TRIPS]
    base_fp = [int(baseline[t][4]) for t in TRIPS]
    base_recall = [baseline[t][2] for t in TRIPS]

    # -------- global: one config across all trips ------------------------------
    # The plan's selection objective ranks by training F1 with composite>=38.4
    # and MAE constraint. For the global view (no held-out trip) we apply the
    # same rule to the union of all six trips.
    # The MAE constraint reference is the physical baseline macro MAE.
    base_macro_mae = _macro(base_mae)
    global_candidates = []
    for label, per_trip in results.items():
        macro_f1 = _macro([per_trip[t][0] for t in TRIPS])
        macro_comp = _macro([per_trip[t][6] for t in TRIPS])
        macro_mae = _macro([per_trip[t][7] for t in TRIPS])
        macro_fp = sum(int(per_trip[t][4]) for t in TRIPS)
        macro_recall = _macro([per_trip[t][2] for t in TRIPS])
        passes_mae = macro_mae <= base_macro_mae + 1e-4
        passes_comp = macro_comp >= 38.4
        global_candidates.append((
            label, macro_f1, macro_comp, macro_mae, passes_comp, passes_mae,
            macro_fp, macro_recall,
        ))
    # Select: among configs passing BOTH gates, pick highest macro F1;
    # if none passes, record the best-by-F1 overall and flag it infeasible.
    feasible_global = [c for c in global_candidates if c[4] and c[5]]
    if feasible_global:
        best_global = max(feasible_global, key=lambda c: c[1])
        global_feasible = True
    else:
        best_global = max(global_candidates, key=lambda c: c[1])
        global_feasible = False

    # -------- LOTO: proper 6-fold leave-one-trip-out --------------------------
    # For each fold: train on the OTHER 5 trips, select the config that maximizes
    # mean train F1 subject to (a) composite>=38.4 and (b) MAE <= base_train_mae.
    # Then evaluate that config on the held-out trip. Macro is the mean of the
    # 6 held-out F1 values -- the true out-of-sample generalization estimate.
    loto_folds = []
    for val_trip in TRIPS:
        train_trips = [t for t in TRIPS if t != val_trip]
        base_train_mae = _macro([baseline[t][7] for t in train_trips])
        best_label, best_train_f1 = None, -1.0
        for label, per_trip in results.items():
            train_f1s = [per_trip[t][0] for t in train_trips]
            train_comps = [per_trip[t][6] for t in train_trips]
            train_maes = [per_trip[t][7] for t in train_trips]
            m_f1 = _macro(train_f1s)
            m_comp = _macro(train_comps)
            m_mae = _macro(train_maes)
            valid = (m_comp >= 38.4) and (m_mae <= base_train_mae + 1e-4)
            if valid and m_f1 > best_train_f1:
                best_train_f1 = m_f1
                best_label = label
        if best_label is None:
            loto_folds.append({
                "validation_trip": val_trip,
                "selected_config": None,
                "infeasible": True,
                "val_f1": float("nan"), "val_composite": float("nan"),
                "val_mae": float("nan"), "val_fp": None, "val_recall": float("nan"),
            })
            continue
        val_row = results[best_label][val_trip]
        loto_folds.append({
            "validation_trip": val_trip,
            "selected_config": best_label,
            "infeasible": False,
            "val_f1": float(val_row[0]),
            "val_composite": float(val_row[6]),
            "val_mae": float(val_row[7]),
            "val_fp": int(val_row[4]),
            "val_recall": float(val_row[2]),
        })

    feasible_loto = [f for f in loto_folds if not f.get("infeasible", False)]
    if feasible_loto:
        loto_macro_f1 = _macro([f["val_f1"] for f in feasible_loto])
        loto_macro_comp = _macro([f["val_composite"] for f in feasible_loto])
        loto_macro_mae = _macro([f["val_mae"] for f in feasible_loto])
        t05_fp = next((f["val_fp"] for f in feasible_loto if f["validation_trip"] == "T05-Sample"), None)
        t03_recall = next((f["val_recall"] for f in feasible_loto if f["validation_trip"] == "T03-Sample"), None)
    else:
        loto_macro_f1 = loto_macro_comp = loto_macro_mae = float("nan")
        t05_fp = t03_recall = None

    # -------- oracle: per-trip best available choice (not selectable) --------
    # A real upper bound must include the option to disable semantics. Excluding
    # the physical baseline can make the reported "upper bound" lower than an
    # observed system, which is mathematically invalid. Per-trip gates are not
    # applied here because acceptance gates are macro gates; aggregate gate
    # status is reported on the resulting cherry-picked predictions.
    oracle_per_trip: dict[str, dict] = {}
    oracle_f1s: list[float] = []
    oracle_composites: list[float] = []
    oracle_maes: list[float] = []
    for trip_id in TRIPS:
        base_row = baseline[trip_id]
        candidates = [(
            "baseline",
            float(base_row[0]),
            float(base_row[6]),
            float(base_row[7]),
            int(base_row[4]),
            float(base_row[2]),
        )]
        for label, per_trip in results.items():
            row = per_trip[trip_id]
            candidates.append((
                label, float(row[0]), float(row[6]), float(row[7]),
                int(row[4]), float(row[2]),
            ))
        best = max(candidates, key=lambda candidate: candidate[1])
        oracle_per_trip[trip_id] = {
            "config": best[0],
            "f1": best[1],
            "composite": best[2],
            "mae_critical": best[3],
            "fp": best[4],
            "recall": best[5],
        }
        oracle_f1s.append(best[1])
        oracle_composites.append(best[2])
        oracle_maes.append(best[3])

    oracle_macro_f1 = _macro(oracle_f1s)
    oracle_macro_composite = _macro(oracle_composites)
    oracle_macro_mae = _macro(oracle_maes)
    oracle_passes_composite = oracle_macro_composite >= 38.4
    oracle_passes_mae = oracle_macro_mae <= base_macro_mae + 1e-4
    summary = {
        "baseline": {
            "macro_f1": _macro(base_f1),
            "macro_composite": _macro(base_comp),
            "macro_mae_critical": _macro(base_mae),
            "total_fp": sum(base_fp),
            "macro_recall": _macro(base_recall),
            "per_trip": {
                t: {
                    "f1": float(baseline[t][0]),
                    "composite": float(baseline[t][6]),
                    "mae_critical": float(baseline[t][7]),
                    "fp": int(baseline[t][4]),
                    "recall": float(baseline[t][2]),
                } for t in TRIPS
            },
        },
        "global_best": {
            "config": best_global[0],
            "macro_f1": best_global[1],
            "macro_composite": best_global[2],
            "macro_mae_critical": best_global[3],
            "passes_comp_gate": best_global[4],
            "passes_mae_gate": best_global[5],
            "feasible": global_feasible,
            "total_fp": best_global[6],
            "macro_recall": best_global[7],
            "selection_rule": "macro F1 with composite>=38.4 and macro MAE<=baseline+1e-4",
        },
        "loto": {
            "macro_f1": loto_macro_f1,
            "macro_composite": loto_macro_comp,
            "macro_mae_critical": loto_macro_mae,
            "feasible_folds": len(feasible_loto),
            "infeasible_folds": len(loto_folds) - len(feasible_loto),
            "t05_false_positives": t05_fp,
            "t03_recall": t03_recall,
            "folds": loto_folds,
            "selection_rule": (
                "per fold: train on 5 trips, pick config maximizing train macro "
                "F1 subject to composite>=38.4 and MAE<=train baseline+1e-4; "
                "evaluate once on held-out trip"
            ),
        },
        "oracle_per_trip": {
            "macro_f1": oracle_macro_f1,
            "macro_composite": oracle_macro_composite,
            "macro_mae_critical": oracle_macro_mae,
            "passes_composite_gate": oracle_passes_composite,
            "passes_mae_gate": oracle_passes_mae,
            "selection_space": "baseline (semantics off) + 27 semantic configs",
            "description": "UPPER BOUND ONLY -- not selectable (uses test-trip "
                           "labels to choose baseline or a semantic config per "
                           "trip); reported to bound the achievable F1 ceiling.",
            "per_trip": oracle_per_trip,
        },
        "gates_vs_target": {
            "target_macro_f1_ge_0_60": {
                "baseline": _macro(base_f1) >= 0.60,
                "global_best": best_global[1] >= 0.60,
                "loto": loto_macro_f1 >= 0.60 if feasible_loto else False,
                "oracle": oracle_macro_f1 >= 0.60,
            },
            "target_mae_no_regression_vs_baseline": {
                "global_best": best_global[5],
                "loto_all_feasible": len(feasible_loto) == len(TRIPS),
                "oracle": oracle_passes_mae,
            },
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible per-config-per-trip YOLO26 fusion sweep."
    )
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("ai_cv/outputs/benchmarks/phase04_loto/source"),
    )
    parser.add_argument(
        "--detections-dir", type=Path,
        default=Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference/detections"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_loto"),
    )
    args = parser.parse_args()

    csv_path, sweep = run_full_sweep(
        args.source_root, args.detections_dir, args.output_dir,
    )
    summary = compute_selection_modes(sweep)

    summary_path = args.output_dir / "sweep_selection_modes.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Concise human-readable summary.
    print("\n================ Selection Modes Summary ================")
    b = summary["baseline"]
    g = summary["global_best"]
    lo = summary["loto"]
    orc = summary["oracle_per_trip"]
    print(f"Sweep CSV:           {csv_path}")
    print(f"Selection modes JSON: {summary_path}")
    print()
    print(f"baseline (physical guard):   macro F1={b['macro_f1']:.4f}  "
          f"composite={b['macro_composite']:.2f}  MAE={b['macro_mae_critical']:.2f}s  "
          f"total FP={b['total_fp']}")
    feas_str = "FEASIBLE" if g["feasible"] else "INFEASIBLE"
    print(f"global_best ({g['config']}): macro F1={g['macro_f1']:.4f}  "
          f"composite={g['macro_composite']:.2f}  MAE={g['macro_mae_critical']:.2f}s  "
          f"FP={g['total_fp']}  [{feas_str}]")
    if lo["feasible_folds"] > 0:
        print(f"loto ({lo['feasible_folds']}/6 feasible):    macro F1={lo['macro_f1']:.4f}  "
              f"composite={lo['macro_composite']:.2f}  MAE={lo['macro_mae_critical']:.2f}s  "
              f"T05 FP={lo['t05_false_positives']}  T03 recall={lo['t03_recall']:.3f}")
    else:
        print(f"loto (all 6 folds infeasible):  no config met both gates on any fold")
    print(f"oracle (UPPER BOUND):        macro F1={orc['macro_f1']:.4f}  "
          f"composite={orc['macro_composite']:.2f}  "
          f"MAE={orc['macro_mae_critical']:.2f}s")
    print()
    print(f"Target macro F1 >= 0.60:")
    for k, v in summary["gates_vs_target"]["target_macro_f1_ge_0_60"].items():
        print(f"  {k:20s} {v}")
    print()
    if orc["macro_f1"] < 0.60:
        print("CONCLUSION: oracle (upper bound) macro F1 < 0.60 -> no config or "
              "selection mode can reach the gate. Per plan: proceed to stratified "
              "T03 annotation to classify whether suppressed candidates are genuine "
              "YOLO misses, stereo noise, or association failures.")


if __name__ == "__main__":
    main()
