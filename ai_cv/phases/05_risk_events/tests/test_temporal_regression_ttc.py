from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment_temporal_regression_ttc import regression_ttc


def test_regression_ttc_uses_five_observations_and_weighted_rate() -> None:
    observations = [
        {"timestamp": i * 0.1, "depth_m": 10.0 - i, "depth_sigma_m": 0.2}
        for i in range(5)
    ]
    assert regression_ttc(observations) == pytest.approx(0.6)
    assert regression_ttc(observations[:4]) is None
