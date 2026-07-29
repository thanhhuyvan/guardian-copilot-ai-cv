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
    parser.add_argument(
        "--confidence-temporal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the experimental confidence-gated temporal TTC lane.",
    )
    parser.add_argument(
        "--depth-confidence-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--minimum-depth-confidence", type=float, default=0.25)
    args = parser.parse_args(argv)

    args.integrated_union_events = True
    args.parallel_inference = True
    # The Phase 05 evaluator owns experimental gates.  Pin every one off for
    # the deployment candidate so later research switches cannot leak into a
    # certification run.
    args.experimental_classical_minimum_closing_speed_mps = 0.3
    args.experimental_low_ego_speed_max_mps = 0.0
    args.experimental_low_ego_suppressed_ttc = None
    args.experimental_turn_lateral_accel_mps2 = 0.0
    args.experimental_turn_minimum_yolo_iou = 0.0
    args.experimental_path_intersection = False
    args.experimental_path_corridor_half_width_m = 1.75
    args.v2_shadow_state = False
    args.experimental_v2_ekf_ttc_gate = False
    args.experimental_v2_event_to_ttc = False
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
