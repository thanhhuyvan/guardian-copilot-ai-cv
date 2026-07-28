"""Compare frozen SGBM and zero-shot LightStereo geometry on 72 pair manifest.

This diagnostic is descriptive only: it does not fit a calibration, alter
thresholds, or claim an accuracy improvement.  Its purpose is to locate
whether a learned-stereo failure originates in disparity scale, valid support,
ground fitting, or obstacle-component inflation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_stereo_latency import _load_dataset_class
from classical_geometry import estimate_ground_model, ground_and_obstacle_masks
from lightstereo_deployment import load_pair_manifest
from stereo_backends import create_backend


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _geometry(disparity: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = valid_mask & np.isfinite(disparity) & (disparity > 0.5)
    values = disparity[valid]
    model, _ = estimate_ground_model(disparity)
    obstacle_fraction = 0.0
    if model is not None:
        _, obstacle, _ = ground_and_obstacle_masks(disparity, model)
        obstacle_fraction = float(np.count_nonzero(obstacle) / disparity.size)
    return {
        "valid_fraction": float(np.count_nonzero(valid) / disparity.size),
        "disparity_median_px": float(np.median(values)) if values.size else 0.0,
        "disparity_p10_px": float(np.percentile(values, 10)) if values.size else 0.0,
        "disparity_p90_px": float(np.percentile(values, 90)) if values.size else 0.0,
        "ground_confidence": float(model.confidence) if model is not None else 0.0,
        "ground_residual_px": float(model.median_residual_px) if model is not None else 0.0,
        "ground_slope": float(model.disparity_per_row) if model is not None else 0.0,
        "obstacle_fraction": obstacle_fraction,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    _, pairs = load_pair_manifest(
        args.manifest,
        expected_kind="lightstereo-onnx-parity",
        expected_count=72,
    )
    data_root = args.data_root.expanduser().resolve()
    starter_root = args.starter_root.expanduser().resolve()
    trip_dataset = _load_dataset_class(starter_root)
    calibration = trip_dataset(data_root / "T01-Sample").load_calibration()
    focal = float(calibration["K_left"][0][0])
    baseline = float(calibration["baseline_m"])
    sgbm = create_backend("sgbm", precision="fp32", opencv_threads=args.opencv_threads)
    learned = create_backend(
        "lightstereo-pytorch", precision="fp32", model_path=args.checkpoint,
        openstereo_root=args.openstereo_root, device_id=args.device_id,
    )
    rows: list[dict[str, Any]] = []
    try:
        for index, pair in enumerate(pairs, start=1):
            left = cv2.imread(str(data_root / pair.left), cv2.IMREAD_COLOR)
            right = cv2.imread(str(data_root / pair.right), cv2.IMREAD_COLOR)
            if left is None or right is None:
                raise RuntimeError(f"cannot decode {pair.trip_id} #{pair.frame_id}")
            reference = sgbm.infer(left, right)
            candidate = learned.infer(left, right)
            ref = _geometry(reference.disparity_px, reference.valid_mask)
            cand = _geometry(candidate.disparity_px, candidate.valid_mask)
            overlap = reference.valid_mask & candidate.valid_mask
            ratio = 0.0
            if np.any(overlap):
                denominator = np.median(reference.disparity_px[overlap])
                if denominator > 0:
                    ratio = float(np.median(candidate.disparity_px[overlap]) / denominator)
            row = {"trip_id": pair.trip_id, "frame_id": int(pair.frame_id), "shared_disparity_ratio": ratio}
            row.update({f"sgbm_{name}": value for name, value in ref.items()})
            row.update({f"lightstereo_{name}": value for name, value in cand.items()})
            rows.append(row)
            print(f"diagnostic {index}/72 {pair.trip_id} #{pair.frame_id}", flush=True)
    finally:
        sgbm.close()
        learned.close()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ratios = [row["shared_disparity_ratio"] for row in rows if row["shared_disparity_ratio"] > 0]
    report = {
        "schema": "guardian.phase02b.lightstereo-calibration-diagnosis.v1",
        "manifest": str(args.manifest.expanduser().resolve()),
        "pair_count": len(rows),
        "calibration": {"focal_length_px": focal, "baseline_m": baseline},
        "disparity_ratio_lightstereo_over_sgbm": _summary(ratios),
        "sgbm": {name: _summary([row[f"sgbm_{name}"] for row in rows]) for name in ("valid_fraction", "ground_confidence", "ground_residual_px", "obstacle_fraction")},
        "lightstereo": {name: _summary([row[f"lightstereo_{name}"] for row in rows]) for name in ("valid_fraction", "ground_confidence", "ground_residual_px", "obstacle_fraction")},
        "largest_obstacle_inflation": sorted(rows, key=lambda row: row["lightstereo_obstacle_fraction"] - row["sgbm_obstacle_fraction"], reverse=True)[:12],
        "per_frame_csv": str(csv_path),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--starter-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openstereo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--opencv-threads", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(run(parse_args()), indent=2, allow_nan=False))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
