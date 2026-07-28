from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from path_relative_state import (
    PlanarNoise,
    PlanarRelativeKalmanFilter,
    camera_measurement_to_planar,
    compensate_ego_motion,
    yaw_rate_rps,
)


def test_camera_measurement_projects_stereo_depth_to_lateral_position() -> None:
    assert camera_measurement_to_planar(
        depth_m=10.0, center_x_px=420.0, focal_length_px=500.0, principal_x_px=320.0
    ) == (10.0, 2.0)


def test_ego_motion_turns_prior_relative_position() -> None:
    forward, lateral = compensate_ego_motion(
        longitudinal_m=20.0, lateral_m=0.0, speed_mps=10.0, yaw_rate=0.1, dt_s=0.1
    )
    assert forward < 19.0
    assert lateral < 0.0
    assert yaw_rate_rps(10.0, 1.0) == 0.1
    assert yaw_rate_rps(0.5, 1.0) is None


def test_planar_filter_rejects_large_statistical_innovation() -> None:
    filter_ = PlanarRelativeKalmanFilter(
        PlanarNoise(0.2, 0.2, 1.0, 1.0)
    )
    assert filter_.update(timestamp=0.0, longitudinal_m=10.0, lateral_m=0.0).accepted
    assert filter_.update(timestamp=0.1, longitudinal_m=9.9, lateral_m=0.0).accepted
    outlier = filter_.update(timestamp=0.2, longitudinal_m=2.0, lateral_m=8.0)
    assert not outlier.accepted
    assert math.isfinite(outlier.mahalanobis_squared)
