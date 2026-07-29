"""Causal ego-compensated planar track state for Guardian V2 experiments.

This module is deliberately not wired into V1 risk selection.  Its noise
values must be measured from stereo and detector residuals before promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PlanarNoise:
    """Measured-noise contract for a planar constant-velocity filter."""

    longitudinal_sigma_m: float
    lateral_sigma_m: float
    longitudinal_accel_sigma_mps2: float
    lateral_accel_sigma_mps2: float


@dataclass(frozen=True)
class PlanarUpdate:
    accepted: bool
    mahalanobis_squared: float
    longitudinal_m: float
    lateral_m: float
    longitudinal_velocity_mps: float
    lateral_velocity_mps: float


def yaw_rate_rps(speed_mps: float, lateral_accel_mps2: float) -> float | None:
    """Estimate yaw rate from telemetry; undefined near standstill."""
    if not math.isfinite(speed_mps) or abs(speed_mps) < 1.0:
        return None
    if not math.isfinite(lateral_accel_mps2):
        return None
    return float(lateral_accel_mps2 / speed_mps)


def camera_measurement_to_planar(
    *,
    depth_m: float,
    center_x_px: float,
    focal_length_px: float,
    principal_x_px: float,
) -> tuple[float, float] | None:
    """Project a box-centre stereo measurement into camera forward/lateral."""
    if (
        not math.isfinite(depth_m)
        or depth_m <= 0.0
        or not math.isfinite(center_x_px)
        or not math.isfinite(focal_length_px)
        or focal_length_px <= 0.0
    ):
        return None
    return (
        float(depth_m),
        float((center_x_px - principal_x_px) * depth_m / focal_length_px),
    )


def compensate_ego_motion(
    *,
    longitudinal_m: float,
    lateral_m: float,
    speed_mps: float,
    yaw_rate: float | None,
    dt_s: float,
) -> tuple[float, float]:
    """Map a prior relative position into the current ego frame.

    The local frame is forward/lateral.  Translation is the bicycle-model
    first-order step; rotation is omitted safely when yaw is unavailable.
    """
    forward = longitudinal_m - max(0.0, speed_mps) * dt_s
    lateral = lateral_m
    angle = 0.0 if yaw_rate is None else -yaw_rate * dt_s
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        float(cosine * forward - sine * lateral),
        float(sine * forward + cosine * lateral),
    )


def host_lateral_displacement_m(
    speed_mps: float, yaw_rate: float | None, horizon_s: float
) -> float:
    """Bicycle-model host lateral displacement over a future horizon."""
    if yaw_rate is None or abs(yaw_rate) < 1e-4:
        return 0.0
    return float(speed_mps / yaw_rate * (1.0 - math.cos(yaw_rate * horizon_s)))


def corridor_occupancy_probability(
    *, lateral_mean_m: float, lateral_variance_m2: float, corridor_half_width_m: float
) -> float:
    """Gaussian probability that a predicted target lies in the ego corridor."""
    if corridor_half_width_m <= 0.0 or lateral_variance_m2 < 0.0:
        raise ValueError("invalid corridor or lateral variance")
    sigma = math.sqrt(max(lateral_variance_m2, 1e-9))
    normal_cdf = lambda value: 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
    upper = (corridor_half_width_m - lateral_mean_m) / sigma
    lower = (-corridor_half_width_m - lateral_mean_m) / sigma
    return float(np.clip(normal_cdf(upper) - normal_cdf(lower), 0.0, 1.0))


class PlanarRelativeKalmanFilter:
    """Four-state causal filter: forward, lateral, and their velocities."""

    def __init__(self, noise: PlanarNoise, *, gate_chi2: float = 9.21) -> None:
        if min(
            noise.longitudinal_sigma_m,
            noise.lateral_sigma_m,
            noise.longitudinal_accel_sigma_mps2,
            noise.lateral_accel_sigma_mps2,
        ) <= 0.0:
            raise ValueError("all filter noise terms must be positive")
        if gate_chi2 <= 0.0:
            raise ValueError("gate_chi2 must be positive")
        self.noise = noise
        self.gate_chi2 = gate_chi2
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.timestamp: float | None = None

    def _predict(
        self, timestamp: float, *, ego_speed_mps: float = 0.0, yaw_rate: float | None = None
    ) -> None:
        if self.state is None or self.covariance is None or self.timestamp is None:
            return
        dt = float(timestamp - self.timestamp)
        if not 0.0 < dt <= 0.5:
            return
        transition = np.asarray(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        def process(axis_sigma: float) -> np.ndarray:
            return axis_sigma**2 * np.asarray(
                [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]],
                dtype=np.float64,
            )
        process_noise = np.zeros((4, 4), dtype=np.float64)
        process_noise[np.ix_([0, 2], [0, 2])] = process(
            self.noise.longitudinal_accel_sigma_mps2
        )
        process_noise[np.ix_([1, 3], [1, 3])] = process(
            self.noise.lateral_accel_sigma_mps2
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        # The state is target motion in the prior ego frame.  Express it in
        # the current ego frame after host translation and yaw rotation.
        position = compensate_ego_motion(
            longitudinal_m=float(self.state[0]), lateral_m=float(self.state[1]),
            speed_mps=ego_speed_mps, yaw_rate=yaw_rate, dt_s=dt,
        )
        angle = 0.0 if yaw_rate is None else -yaw_rate * dt
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        self.state[:2] = position
        self.state[2:] = rotation @ self.state[2:]
        transform = np.zeros((4, 4)); transform[np.ix_([0, 1], [0, 1])] = rotation; transform[np.ix_([2, 3], [2, 3])] = rotation
        self.covariance = transform @ self.covariance @ transform.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update(
        self,
        *,
        timestamp: float,
        longitudinal_m: float,
        lateral_m: float,
        ego_speed_mps: float = 0.0,
        yaw_rate: float | None = None,
    ) -> PlanarUpdate:
        measurement = np.asarray([longitudinal_m, lateral_m], dtype=np.float64)
        if not np.all(np.isfinite(measurement)) or not math.isfinite(timestamp):
            raise ValueError("timestamp and measurement must be finite")
        if self.timestamp is None or timestamp <= self.timestamp or timestamp - self.timestamp > 0.5:
            self.state = np.asarray([*measurement, 0.0, 0.0], dtype=np.float64)
            self.covariance = np.diag(
                [self.noise.longitudinal_sigma_m**2, self.noise.lateral_sigma_m**2, 25.0, 25.0]
            )
            self.timestamp = timestamp
            return PlanarUpdate(True, 0.0, *self.state)
        self._predict(timestamp, ego_speed_mps=ego_speed_mps, yaw_rate=yaw_rate)
        assert self.state is not None and self.covariance is not None
        observation = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        measurement_noise = np.diag(
            [self.noise.longitudinal_sigma_m**2, self.noise.lateral_sigma_m**2]
        )
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + measurement_noise
        mahalanobis_squared = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        accepted = mahalanobis_squared <= self.gate_chi2
        if accepted:
            gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
            self.state = self.state + gain @ innovation
            self.covariance = self.covariance - gain @ observation @ self.covariance
            self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.timestamp = timestamp
        return PlanarUpdate(accepted, mahalanobis_squared, *self.state)

    def lateral_distribution_at(self, horizon_s: float) -> tuple[float, float] | None:
        """Return future lateral mean/variance without mutating filter state."""
        if self.state is None or self.covariance is None or horizon_s < 0.0:
            return None
        row = np.asarray([0.0, 1.0, 0.0, horizon_s])
        return float(row @ self.state), float(row @ self.covariance @ row)
