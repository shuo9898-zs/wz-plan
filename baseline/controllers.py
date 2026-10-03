"""
controllers.py
===============
Shared low-level control primitives for hierarchical "motion planner" RL
baselines: an RL policy outputs high-level intent (target speed, heading
adjustment), and this module turns that into safe, smooth
[steer, throttle, brake] commands for CARLA.

Not specific to PPO — any baseline (TD3, SAC, ...) that wants the same
high-level action interface can reuse these classes via motion_planner_env.py.

Pipeline (see motion_planner_env.py for how these compose):
    ppo_target_speed, heading_adjustment          (from RL policy)
        -> ACCController                          (safety: slow down for lead vehicle)
        -> LongitudinalPID (target speed vs ego speed) -> throttle, brake
        -> LateralPID (target heading vs ego heading)  -> steer
"""
from __future__ import annotations

import math


def map_to_speed(normalized: float, max_speed_mps: float) -> float:
    """[-1, 1] -> [0, max_speed_mps]."""
    return (normalized + 1.0) / 2.0 * max_speed_mps


def map_to_heading(normalized: float, max_heading_deg: float) -> float:
    """[-1, 1] -> [-max_heading_deg, +max_heading_deg]."""
    return normalized * max_heading_deg


def _normalize_deg(angle: float) -> float:
    """Normalise an angle (degrees) to (-180, +180]."""
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


class ACCController:
    """
    Constant-time-gap Adaptive Cruise Control safety layer.

    Reduces the RL policy's desired speed when a lead vehicle is closer
    than the desired time-gap distance would allow; otherwise passes the
    desired speed through unchanged. This exists so the RL policy's own
    target-speed output can never directly command a rear-end collision —
    ACC is a safety clamp, not something the policy needs to have learned.

    Parameters
    ----------
    time_gap : float   desired following time gap (s)
    min_gap  : float   minimum following distance regardless of speed (m)
    kp       : float   proportional gain on gap error -> speed correction
    """

    def __init__(self, time_gap: float = 1.5, min_gap: float = 5.0, kp: float = 0.5) -> None:
        self.time_gap = time_gap
        self.min_gap  = min_gap
        self.kp       = kp

    def update(self,
               desired_speed: float,
               ego_speed: float,
               lead_distance: float | None,
               lead_speed: float | None) -> float:
        """Return the safe target speed (m/s), <= desired_speed."""
        if lead_distance is None or lead_speed is None:
            return desired_speed

        desired_gap = self.min_gap + self.time_gap * ego_speed
        gap_error   = lead_distance - desired_gap          # negative = too close
        acc_speed   = lead_speed + self.kp * gap_error
        return max(0.0, min(desired_speed, acc_speed))


class LongitudinalPID:
    """PID controller: target speed vs current speed -> (throttle, brake)."""

    def __init__(self, kp: float = 0.5, ki: float = 0.05, kd: float = 0.1, dt: float = 0.1) -> None:
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self._integral    = 0.0
        self._prev_error  = 0.0

    def reset(self) -> None:
        """Call at the start of every episode to clear accumulated state."""
        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, target_speed: float, current_speed: float) -> tuple[float, float]:
        error = target_speed - current_speed
        self._integral   += error * self.dt
        derivative        = (error - self._prev_error) / self.dt
        self._prev_error   = error

        control = self.kp * error + self.ki * self._integral + self.kd * derivative
        if control >= 0.0:
            return min(control, 1.0), 0.0
        return 0.0, min(-control, 1.0)


class LateralPID:
    """PID controller: target heading vs current heading (degrees) -> steer."""

    def __init__(self, kp: float = 0.03, ki: float = 0.0, kd: float = 0.01, dt: float = 0.1) -> None:
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self) -> None:
        """Call at the start of every episode to clear accumulated state."""
        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, target_heading: float, current_heading: float) -> float:
        error = _normalize_deg(target_heading - current_heading)
        self._integral   += error * self.dt
        derivative        = (error - self._prev_error) / self.dt
        self._prev_error   = error

        control = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-1.0, min(1.0, control))


def find_lead_vehicle(ego, bg_actor_map: dict, bg_speed_map: dict,
                       lane_half_width: float = 3.5):
    """
    Find the nearest background vehicle ahead of `ego`, roughly in the same
    lane (within `lane_half_width` metres perpendicular to ego's heading).

    Parameters
    ----------
    ego             : carla.Actor  the Ego vehicle
    bg_actor_map    : dict  from CarlaSumoEnv.get_background_actor_map()
    bg_speed_map    : dict  from CarlaSumoEnv.get_background_speed_map()
    lane_half_width : float  perpendicular distance (m) that counts as "same lane"

    Returns
    -------
    (distance, speed) of the nearest qualifying lead vehicle, or (None, None)
    if there is none.
    """
    t = ego.get_transform()
    yaw_rad = math.radians(t.rotation.yaw)
    fwd_x, fwd_y     = math.cos(yaw_rad), math.sin(yaw_rad)
    right_x, right_y = -fwd_y, fwd_x

    best_dist  = None
    best_speed = None
    for sid, actor in bg_actor_map.items():
        if not actor.is_alive:
            continue
        loc = actor.get_location()
        dx = loc.x - t.location.x
        dy = loc.y - t.location.y
        longitudinal = dx * fwd_x + dy * fwd_y
        lateral      = dx * right_x + dy * right_y

        if longitudinal <= 0.0:
            continue                       # behind ego
        if abs(lateral) > lane_half_width:
            continue                       # different lane
        if best_dist is None or longitudinal < best_dist:
            best_dist  = longitudinal
            best_speed = bg_speed_map.get(sid, 0.0)

    return best_dist, best_speed
