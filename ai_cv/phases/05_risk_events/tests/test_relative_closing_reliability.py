from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from audit_relative_closing_reliability import theil_sen_depth_slope


def test_theil_sen_depth_slope_recovers_stable_closing() -> None:
    estimate = theil_sen_depth_slope([(0.0, 10.0), (0.1, 9.7), (0.2, 9.4)])
    assert estimate is not None
    slope, residual_mad = estimate
    assert abs(slope + 3.0) < 1e-9
    assert residual_mad < 1e-9
