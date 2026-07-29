"""Profile the existing V-disparity ground estimator without changing output."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from classical_geometry import (
    fit_ground_line,
    fit_ground_line_vectorized,
    ground_and_obstacle_masks,
    row_disparity_modes,
    v_disparity_histogram,
)
from stereo_backends import create_backend


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(np.mean(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--trip", default="T05-Sample")
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--stereo-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image_root = args.practice_root / args.trip / "kitti"
    left_paths = sorted((image_root / "image_2").glob("*.jpg"))[: args.max_frames]
    if not left_paths:
        raise FileNotFoundError(image_root / "image_2")
    backend = create_backend("sgbm", precision="fp32", opencv_threads=args.opencv_threads, stereo_workers=args.stereo_workers)
    timings: dict[str, list[float]] = {name: [] for name in ("histogram", "row_modes", "line_fit", "line_fit_vectorized", "masks", "ground_total", "ground_total_vectorized")}
    parity_mismatches = 0
    try:
        for index, left_path in enumerate(left_paths):
            right_path = image_root / "image_3" / left_path.name
            left, right = cv2.imread(str(left_path)), cv2.imread(str(right_path))
            if left is None or right is None:
                raise FileNotFoundError(left_path if left is None else right_path)
            disparity = backend.infer(left, right).disparity_px
            start = time.perf_counter(); histogram = v_disparity_histogram(disparity); histogram_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter(); rows, modes, weights = row_disparity_modes(histogram); modes_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter(); model = fit_ground_line(rows, modes, weights); line_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter(); vectorized_model = fit_ground_line_vectorized(rows, modes, weights); vectorized_ms = (time.perf_counter() - start) * 1000.0
            if model != vectorized_model:
                parity_mismatches += 1
            start = time.perf_counter()
            if model is not None:
                ground_and_obstacle_masks(disparity, model)
            masks_ms = (time.perf_counter() - start) * 1000.0
            if index >= args.warmup_frames:
                timings["histogram"].append(histogram_ms)
                timings["row_modes"].append(modes_ms)
                timings["line_fit"].append(line_ms)
                timings["line_fit_vectorized"].append(vectorized_ms)
                timings["masks"].append(masks_ms)
                timings["ground_total"].append(histogram_ms + modes_ms + line_ms)
                timings["ground_total_vectorized"].append(histogram_ms + modes_ms + vectorized_ms)
    finally:
        backend.close()
    report = {
        "contract": "existing V-disparity implementation only; no output/policy change",
        "trip": args.trip,
        "measured_frames": len(timings["ground_total"]),
        "line_fit_parity_mismatches": parity_mismatches,
        "configuration": {"opencv_threads": args.opencv_threads, "stereo_workers": args.stereo_workers},
        "timing_ms": {name: _summary(values) for name, values in timings.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
