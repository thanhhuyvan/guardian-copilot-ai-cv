from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from audit_planar_track_noise import residual_sigmas


def test_residual_sigmas_reports_zero_for_perfect_linear_track() -> None:
    observations = [
        {"timestamp": index * 0.1, "depth_m": 10.0 - index * 0.2, "center_x": 320.0}
        for index in range(5)
    ]
    depth, lateral = residual_sigmas(observations, 500.0, 320.0) or (None, None)
    assert depth is not None and depth < 1e-12
    assert lateral is not None and lateral < 1e-12
