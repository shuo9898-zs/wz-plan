"""Vehicle-agnostic low-level controller for PPO V2.

The policy emits only two normalized intentions::

    [target_speed, target_yaw_rate]

This module maps them to physical targets and uses two independent feedback
controllers:

* speed PID: target speed - measured speed -> throttle or brake;
* yaw-rate PID: target yaw rate - heading-derived yaw rate -> steering.

No CARLA waypoint, lane heading, base heading, actor map, ACC rule, or CARLA
IMU is used.  The measured yaw rate is calculated from consecutive headings
at the configured control period, which keeps the interface portable to a
real AV localization source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple


CONTROLLER_CONTRACT_VERSION_V2 = "target_speed_target_yaw_rate_dual_pid_v2"


def wrap_heading_delta_deg_v2(delta_deg: float) -> float:
    """Wrap a heading difference to ``[-180, 180)`` degrees."""
    delta = _finite_v2(delta_deg, "delta_deg")
    return (delta + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class PIDGainsV2:
    """PID gains plus an explicit integral-state bound.

    The defaults below are conservative simulation starting points, not
    vehicle calibration.  A real AV must tune them against its own actuator
    response while preserving this controller interface.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 10.0

    def __post_init__(self) -> None:
        for name, value in (
            ("kp", self.kp),
            ("ki", self.ki),
            ("kd", self.kd),
            ("integral_limit", self.integral_limit),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.kp < 0.0 or self.ki < 0.0 or self.kd < 0.0:
            raise ValueError("PID gains must be non-negative")
        if self.integral_limit <= 0.0:
            raise ValueError("integral_limit must be positive")


@dataclass(frozen=True)
class ControllerConfigV2:
    """Physical bounds and two independent controller calibrations."""

    dt_s: float = 0.1
    max_speed_mps: float = 13.89
    max_yaw_rate_deg_s: float = 30.0
    speed_pid: PIDGainsV2 = field(
        default_factory=lambda: PIDGainsV2(
            kp=0.35,
            ki=0.08,
            kd=0.02,
            integral_limit=5.0,
        )
    )
    yaw_rate_pid: PIDGainsV2 = field(
        default_factory=lambda: PIDGainsV2(
            kp=0.04,
            ki=0.005,
            kd=0.001,
            integral_limit=30.0,
        )
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("dt_s", self.dt_s),
            ("max_speed_mps", self.max_speed_mps),
            ("max_yaw_rate_deg_s", self.max_yaw_rate_deg_s),
        ):
            if not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TargetCommandV2:
    """Clipped normalized action and its physical target values."""

    normalized_speed: float
    normalized_yaw_rate: float
    target_speed_mps: float
    target_yaw_rate_deg_s: float


@dataclass(frozen=True)
class VehicleControlV2:
    """Actuator commands and diagnostics from one 10-Hz control step."""

    steer: float
    throttle: float
    brake: float
    target_speed_mps: float
    measured_speed_mps: float
    speed_error_mps: float
    target_yaw_rate_deg_s: float
    measured_yaw_rate_deg_s: float
    yaw_rate_error_deg_s: float
    normalized_action: Tuple[float, float]


class HeadingRateEstimatorV2:
    """Estimate yaw rate from wrapped differences of consecutive headings."""

    def __init__(self, dt_s: float) -> None:
        if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        self.dt_s = float(dt_s)
        self._previous_heading_deg: float | None = None

    def reset(self, heading_deg: float) -> None:
        self._previous_heading_deg = _finite_v2(heading_deg, "heading_deg")

    def update(self, heading_deg: float) -> float:
        heading = _finite_v2(heading_deg, "heading_deg")
        if self._previous_heading_deg is None:
            raise RuntimeError("HeadingRateEstimatorV2.reset() must be called first")
        delta = wrap_heading_delta_deg_v2(heading - self._previous_heading_deg)
        self._previous_heading_deg = heading
        return delta / self.dt_s


class PIDControllerV2:
    """Bounded scalar PID with derivative initialization and anti-windup."""

    def __init__(
        self,
        gains: PIDGainsV2,
        *,
        dt_s: float,
        output_min: float = -1.0,
        output_max: float = 1.0,
    ) -> None:
        if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if not math.isfinite(float(output_min)) or not math.isfinite(float(output_max)):
            raise ValueError("PID output bounds must be finite")
        if output_min >= output_max:
            raise ValueError("output_min must be less than output_max")
        self.gains = gains
        self.dt_s = float(dt_s)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self._integral = 0.0
        self._previous_error: float | None = None

    @property
    def integral(self) -> float:
        return self._integral

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None

    def update(self, error: float) -> float:
        error = _finite_v2(error, "error")
        derivative = (
            0.0
            if self._previous_error is None
            else (error - self._previous_error) / self.dt_s
        )
        candidate_integral = _clip_v2(
            self._integral + error * self.dt_s,
            -self.gains.integral_limit,
            self.gains.integral_limit,
        )
        candidate_output = (
            self.gains.kp * error
            + self.gains.ki * candidate_integral
            + self.gains.kd * derivative
        )

        # Conditional integration: when saturated, retain only an integral
        # update that pushes the command back toward the valid output range.
        saturating_high = candidate_output > self.output_max and error > 0.0
        saturating_low = candidate_output < self.output_min and error < 0.0
        if not (saturating_high or saturating_low):
            self._integral = candidate_integral

        output = (
            self.gains.kp * error
            + self.gains.ki * self._integral
            + self.gains.kd * derivative
        )
        self._previous_error = error
        return _clip_v2(output, self.output_min, self.output_max)


class DualPIDControllerV2:
    """Map PPO intent to throttle, brake, and steering without driving rules."""

    def __init__(self, config: ControllerConfigV2 = ControllerConfigV2()) -> None:
        self.config = config
        self._speed_pid = PIDControllerV2(
            config.speed_pid,
            dt_s=config.dt_s,
        )
        self._yaw_rate_pid = PIDControllerV2(
            config.yaw_rate_pid,
            dt_s=config.dt_s,
        )
        self._heading_rate = HeadingRateEstimatorV2(config.dt_s)
        self._ready = False

    def reset(self, *, initial_heading_deg: float) -> None:
        """Start an episode from one localization heading sample."""
        self._speed_pid.reset()
        self._yaw_rate_pid.reset()
        self._heading_rate.reset(initial_heading_deg)
        self._ready = True

    def map_action(self, action: Sequence[float]) -> TargetCommandV2:
        """Clip PPO ``[-1, 1]^2`` action and map it to physical targets."""
        try:
            if len(action) != 2:
                raise ValueError
            raw_speed = _finite_v2(action[0], "action[0]")
            raw_yaw = _finite_v2(action[1], "action[1]")
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(
                "V2 action must contain exactly [target_speed, target_yaw_rate]"
            ) from error

        normalized_speed = _clip_v2(raw_speed, -1.0, 1.0)
        normalized_yaw = _clip_v2(raw_yaw, -1.0, 1.0)
        return TargetCommandV2(
            normalized_speed=normalized_speed,
            normalized_yaw_rate=normalized_yaw,
            target_speed_mps=(normalized_speed + 1.0) * 0.5 * self.config.max_speed_mps,
            target_yaw_rate_deg_s=(
                normalized_yaw * self.config.max_yaw_rate_deg_s
            ),
        )

    def update(
        self,
        action: Sequence[float],
        *,
        measured_speed_mps: float,
        current_heading_deg: float,
    ) -> VehicleControlV2:
        """Run one closed-loop update and return normalized actuator commands."""
        if not self._ready:
            raise RuntimeError("DualPIDControllerV2.reset() must be called first")
        speed = _finite_v2(measured_speed_mps, "measured_speed_mps")
        if speed < 0.0:
            raise ValueError("measured_speed_mps must be non-negative")
        target = self.map_action(action)
        measured_yaw_rate = self._heading_rate.update(current_heading_deg)
        speed_error = target.target_speed_mps - speed
        yaw_rate_error = target.target_yaw_rate_deg_s - measured_yaw_rate

        signed_longitudinal = self._speed_pid.update(speed_error)
        steer = self._yaw_rate_pid.update(yaw_rate_error)
        throttle = max(0.0, signed_longitudinal)
        brake = max(0.0, -signed_longitudinal)
        return VehicleControlV2(
            steer=steer,
            throttle=throttle,
            brake=brake,
            target_speed_mps=target.target_speed_mps,
            measured_speed_mps=speed,
            speed_error_mps=speed_error,
            target_yaw_rate_deg_s=target.target_yaw_rate_deg_s,
            measured_yaw_rate_deg_s=measured_yaw_rate,
            yaw_rate_error_deg_s=yaw_rate_error,
            normalized_action=(
                target.normalized_speed,
                target.normalized_yaw_rate,
            ),
        )


def _clip_v2(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _finite_v2(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


__all__ = [
    "CONTROLLER_CONTRACT_VERSION_V2",
    "ControllerConfigV2",
    "DualPIDControllerV2",
    "HeadingRateEstimatorV2",
    "PIDControllerV2",
    "PIDGainsV2",
    "TargetCommandV2",
    "VehicleControlV2",
    "wrap_heading_delta_deg_v2",
]
