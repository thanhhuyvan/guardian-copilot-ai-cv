"""Create diagnostic charts and stability tables for the vertical slice."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRIPS = [f"T0{index}-Sample" for index in range(1, 7)]
TRACK_VARIANTS = ("track_p20", "track_p35", "track_median")


def confusion_rows(output_root: Path) -> list[dict]:
    rows = []
    for variant_dir in sorted((output_root / "predictions").iterdir()):
        if not variant_dir.is_dir():
            continue
        total_tp = total_fp = total_fn = finite = 0
        for path in sorted(variant_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            prediction = pd.to_numeric(
                frame.predicted_ttc, errors="coerce"
            ).to_numpy(float)
            ground_truth = pd.to_numeric(
                frame.ground_truth_ttc, errors="coerce"
            ).to_numpy(float)
            predicted_danger = prediction < 2.0
            actual_danger = ground_truth < 2.0
            tp = int(np.count_nonzero(predicted_danger & actual_danger))
            fp = int(np.count_nonzero(predicted_danger & ~actual_danger))
            fn = int(np.count_nonzero(~predicted_danger & actual_danger))
            total_tp += tp
            total_fp += fp
            total_fn += fn
            finite += int(np.count_nonzero(np.isfinite(prediction)))
            rows.append(
                {
                    "variant": variant_dir.name,
                    "trip_id": path.stem,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "finite_predictions": int(
                        np.count_nonzero(np.isfinite(prediction))
                    ),
                }
            )
        rows.append(
            {
                "variant": variant_dir.name,
                "trip_id": "OVERALL",
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "finite_predictions": finite,
            }
        )
    return rows


def stability_rows(output_root: Path) -> list[dict]:
    rows = []
    for path in sorted((output_root / "diagnostics").glob("*.csv")):
        frame = pd.read_csv(path)
        frame.predicted_ttc = pd.to_numeric(
            frame.predicted_ttc, errors="coerce"
        )
        for variant in TRACK_VARIANTS:
            subset = frame[
                frame.variant.eq(variant) & np.isfinite(frame.predicted_ttc)
            ]
            track_ids = subset.selected_track_id.dropna().to_numpy()
            switches = (
                int(np.count_nonzero(track_ids[1:] != track_ids[:-1]))
                if len(track_ids) > 1
                else 0
            )
            rows.append(
                {
                    "variant": variant,
                    "trip_id": path.stem,
                    "finite_predictions": len(subset),
                    "selected_track_switches": switches,
                    "switches_per_100_finite": (
                        100.0 * switches / len(subset) if len(subset) else 0.0
                    ),
                    "median_ground_confidence": float(
                        subset.ground_confidence.median()
                    ),
                    "median_prediction_confidence": float(
                        subset.prediction_confidence.median()
                    ),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def display_ttc(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(numeric), np.minimum(numeric, 10.0), 10.0)


def plot_ttc_timelines(output_root: Path) -> None:
    figure, axes = plt.subplots(
        3, 2, figsize=(17, 12), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, trip_id in zip(axes.flat, TRIPS, strict=True):
        p35 = pd.read_csv(
            output_root / "predictions" / "track_p35" / f"{trip_id}.csv"
        )
        median = pd.read_csv(
            output_root / "predictions" / "track_median" / f"{trip_id}.csv"
        )
        axis.axhspan(0, 2, color="#ffdddd", alpha=0.8)
        axis.axhspan(2, 3, color="#fff0c2", alpha=0.7)
        axis.plot(
            p35.timestamp,
            display_ttc(p35.ground_truth_ttc),
            color="#111111",
            linewidth=2,
            label="GT",
        )
        axis.plot(
            p35.timestamp,
            display_ttc(p35.predicted_ttc),
            color="#1f77b4",
            linewidth=1,
            alpha=0.8,
            label="track p35",
        )
        axis.plot(
            median.timestamp,
            display_ttc(median.predicted_ttc),
            color="#d62728",
            linewidth=1,
            alpha=0.75,
            label="track median",
        )
        axis.set_title(trip_id)
        axis.set_ylim(0, 10.2)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncols=3, fontsize=8)
    figure.supxlabel("Trip time (seconds)")
    figure.supylabel("TTC capped at 10 s; inf shown at 10 s")
    figure.suptitle("Classical vertical-slice TTC timelines", fontsize=16)
    figure.savefig(output_root / "ttc_timelines.png", dpi=165)
    plt.close(figure)


def plot_tracking_stability(output_root: Path) -> None:
    trips = ("T03-Sample", "T06-Sample")
    figure, axes = plt.subplots(
        len(trips), 2, figsize=(17, 8), constrained_layout=True
    )
    for row, trip_id in enumerate(trips):
        frame = pd.read_csv(output_root / "diagnostics" / f"{trip_id}.csv")
        subset = frame[frame.variant.eq("track_p35")].copy()
        selected = subset.selected_track_id.notna()
        axes[row, 0].scatter(
            subset.loc[selected, "timestamp"],
            subset.loc[selected, "selected_track_id"],
            c=subset.loc[selected, "predicted_ttc"].apply(
                lambda value: 10.0 if str(value) == "inf" else min(float(value), 10.0)
            ),
            cmap="turbo_r",
            s=12,
        )
        axes[row, 0].set_ylabel("selected track ID")
        axes[row, 0].set_title(f"{trip_id}: selected ID fragmentation")
        axes[row, 0].grid(alpha=0.2)

        axes[row, 1].plot(
            subset.timestamp,
            subset.component_count,
            color="#1f77b4",
            label="component count",
        )
        confidence_axis = axes[row, 1].twinx()
        confidence_axis.plot(
            subset.timestamp,
            subset.ground_confidence,
            color="#d62728",
            alpha=0.7,
            label="ground confidence",
        )
        axes[row, 1].set_ylabel("components")
        confidence_axis.set_ylabel("ground confidence")
        confidence_axis.set_ylim(0, 1)
        axes[row, 1].set_title(f"{trip_id}: unstable observations")
        axes[row, 1].grid(alpha=0.2)
    figure.supxlabel("Trip time (seconds)")
    figure.suptitle(
        "Why the classical vertical slice plateaus: component/track instability",
        fontsize=16,
    )
    figure.savefig(output_root / "tracking_stability.png", dpi=165)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ai_cv/outputs/benchmarks/phase02a_vertical_slice"),
    )
    args = parser.parse_args()
    confusion = confusion_rows(args.output_root)
    stability = stability_rows(args.output_root)
    write_rows(args.output_root / "confusion_summary.csv", confusion)
    write_rows(args.output_root / "tracking_stability.csv", stability)
    plot_ttc_timelines(args.output_root)
    plot_tracking_stability(args.output_root)
    print(
        pd.DataFrame(confusion)
        .query("trip_id == 'OVERALL'")
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
