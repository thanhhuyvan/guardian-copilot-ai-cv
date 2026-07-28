"""Independent V2 risk cues; experimental and not connected to V1."""
from __future__ import annotations

import math


def cpa_path_risk(
    *,
    longitudinal_m: float, lateral_m: float,
    longitudinal_velocity_mps: float, lateral_velocity_mps: float,
    horizon_s: float = 2.0, path_half_width_m: float = 1.75,
) -> tuple[bool, float]:
    """Return whether a constant-velocity target enters the ego path."""
    if longitudinal_velocity_mps >= -0.3:
        return False, math.inf
    time_to_host = -longitudinal_m / longitudinal_velocity_mps
    if not 0.0 < time_to_host <= horizon_s:
        return False, time_to_host
    lateral_at_cpa = lateral_m + lateral_velocity_mps * time_to_host
    return abs(lateral_at_cpa) <= path_half_width_m, time_to_host


def looming_tau(
    areas_px2: list[float], timestamps: list[float]) -> float:
    """Causal image-scale expansion TTC (tau); inf when growth is unreliable."""
    if len(areas_px2) < 3 or len(areas_px2) != len(timestamps):
        return math.inf
    if any(area <= 0.0 for area in areas_px2):
        return math.inf
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0.0:
        return math.inf
    slope = (math.log(areas_px2[-1]) - math.log(areas_px2[0])) / duration
    return math.inf if slope <= 0.0 else float(2.0 / slope)


def ttc_cues_agree(stereo_ttc_s: float, looming_ttc_s: float, *, ratio: float = 2.0) -> bool:
    if not math.isfinite(stereo_ttc_s) or not math.isfinite(looming_ttc_s):
        return False
    return max(stereo_ttc_s, looming_ttc_s) / max(0.1, min(stereo_ttc_s, looming_ttc_s)) <= ratio
