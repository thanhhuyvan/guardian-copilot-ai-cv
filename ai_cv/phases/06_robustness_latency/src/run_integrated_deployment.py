"""Run the frozen Phase 06 integrated deployment candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE05_SRC = REPOSITORY_ROOT / "ai_cv" / "phases" / "05_risk_events" / "src"
if str(PHASE05_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE05_SRC))

from evaluate_detector_owned_ttc import run  # noqa: E402


ALL_TRIPS = [f"T{index:02d}-Sample" for index in range(1, 7)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the frozen live Guardian deployment candidate."
    )
    parser.add_argument(
        "--practice-root",
        type=Path,
        default=REPOSITORY_ROOT / "Practice_Dataset",
    )
    parser.add_argument(
        "--starter-root",
        type=Path,
        default=REPOSITORY_ROOT / "Package_starterkit" / "package_starterkit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "ai_cv" / "outputs" / "phase06_integrated",
    )
    parser.add_argument("--trips", nargs="+", default=ALL_TRIPS)
    parser.add_argument(
        "--detector-backend",
        choices=["yolo26-pytorch", "yolo26-onnx"],
        default="yolo26-pytorch",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPOSITORY_ROOT / "yolo26n.pt",
    )
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--start-frame-index", type=int, default=0)
    parser.add_argument("--max-frames-per-trip", type=int)
    parser.add_argument("--latency-target-ms", type=float, default=75.0)
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--stereo-workers", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=300)
    args = parser.parse_args(argv)

    args.integrated_union_events = True
    args.parallel_inference = True
    args.detections_dir = (
        REPOSITORY_ROOT
        / "ai_cv"
        / "phases"
        / "02_detection_tracking"
        / "artifacts"
        / "yolo26_reference"
        / "detections"
    )
    return args


if __name__ == "__main__":
    run(parse_args())
