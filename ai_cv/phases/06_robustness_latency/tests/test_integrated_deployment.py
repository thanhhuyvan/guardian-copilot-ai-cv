from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_integrated_deployment import ALL_TRIPS, parse_args


def test_deployment_defaults_freeze_validated_protocol() -> None:
    args = parse_args([])

    assert args.trips == ALL_TRIPS
    assert args.detector_backend == "yolo26-pytorch"
    assert args.integrated_union_events is True
    assert args.parallel_inference is True
    assert args.warmup_frames == 100
    assert args.repeats == 5
    assert args.latency_target_ms == 75.0
    assert args.opencv_threads == 2
    assert args.stereo_workers == 2


def test_deployment_entry_point_rejects_cached_detections() -> None:
    try:
        parse_args(["--detector-backend", "cached"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("cached detections must not be deployable")
