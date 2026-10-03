"""
motion_planner_env.py
=======================
Gymnasium ActionWrapper that turns a high-level "motion planner" action
into the low-level [steer, throttle, brake] action CarlaSumoGymEnv expects.

Rationale: PPO (or any RL baseline) should not directly output
steer/throttle/brake. It outputs *intent* — desired speed and a heading
adjustment relative to the road direction — and a conventional control
stack (ACC safety layer + longitudinal/lateral PID) turns that into safe,
smooth vehicle control. This mirrors how a real motion planner sits above
low-level vehicle control.

Data flow
---------
    Observation
        -> PPO (or other baseline)
        -> [target_speed_normalized, heading_adjustment_normalized]   (this wrapper's action_space)
        -> ACC safety adjustment + longitudinal/lateral PID controllers
        -> [steer, throttle, brake]                                   (CarlaSumoGymEnv's action_space)
        -> CARLA ego vehicle

IMPORTANT: this wrapper must wrap a CarlaSumoGymEnv directly (not another
wrapper) — it reads `self.env.engine` (the underlying CarlaSumoEnv) to get
ego speed/heading, the per-scenario road heading, and background-vehicle
state for lead-vehicle detection.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baseline.controllers import (
    ACCController, LateralPID, LongitudinalPID,
    find_lead_vehicle, map_to_heading, map_to_speed,
)


class MotionPlannerGymEnv(gym.Wrapper):
    """
    Deprecated compatibility wrapper. CarlaSumoGymEnv now performs the
    two-value intent-to-control conversion itself, so this wrapper passes the
    action through unchanged. The controller members below remain only for
    source compatibility and are not invoked by gym.Wrapper.step().

    Parameters
    ----------
    env             : CarlaSumoGymEnv  (wrapped directly, not nested further)
    max_speed_mps   : float  upper bound for the mapped target speed (default
                      13.89 m/s = the `car` vType's maxSpeed in the .rou.xml files)
    max_heading_deg : float  upper bound for the mapped heading adjustment,
                      applied on top of the scenario's road_heading_deg
    """

    def __init__(self,
                 env: gym.Env,
                 max_speed_mps: float = 13.89,
                 max_heading_deg: float = 30.0) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._max_speed   = max_speed_mps
        self._max_heading = max_heading_deg
        self._acc     = ACCController()
        self._lon_pid = LongitudinalPID()
        self._lat_pid = LateralPID()

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self._lon_pid.reset()
        self._lat_pid.reset()
        return self.env.reset(**kwargs)

    def action(self, action: np.ndarray) -> np.ndarray:
        engine = self.env.engine          # the underlying CarlaSumoEnv
        ego    = engine.ego

        ppo_target_speed   = map_to_speed(float(action[0]), self._max_speed)
        heading_adjustment = map_to_heading(float(action[1]), self._max_heading)

        t   = ego.get_transform()
        v   = ego.get_velocity()
        ego_speed   = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        ego_heading = t.rotation.yaw

        lead_distance, lead_speed = find_lead_vehicle(
            ego, engine.get_background_actor_map(), engine.get_background_speed_map(),
        )
        safe_target_speed = self._acc.update(ppo_target_speed, ego_speed, lead_distance, lead_speed)

        throttle, brake = self._lon_pid.update(safe_target_speed, ego_speed)

        route_heading  = engine.cfg.carla.road_heading_deg
        target_heading = route_heading + heading_adjustment
        steer          = self._lat_pid.update(target_heading, ego_heading)

        return np.array([steer, throttle, brake], dtype=np.float32)
