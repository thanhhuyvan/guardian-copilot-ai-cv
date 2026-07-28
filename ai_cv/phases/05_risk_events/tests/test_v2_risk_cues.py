from __future__ import annotations
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2_risk_cues import cpa_path_risk, looming_tau, ttc_cues_agree

def test_path_and_looming_cues() -> None:
    assert cpa_path_risk(longitudinal_m=10, lateral_m=3, longitudinal_velocity_mps=-10, lateral_velocity_mps=0)[0] is False
    assert cpa_path_risk(longitudinal_m=10, lateral_m=0, longitudinal_velocity_mps=-10, lateral_velocity_mps=0)[0] is True
    assert looming_tau([100,121,144],[0,.1,.2]) < math.inf
    assert ttc_cues_agree(1.0,1.5)
