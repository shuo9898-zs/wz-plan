"""
CarlaSumoGymEnv
===============
Thin Gymnasium-compatible adapter around CarlaSumoEnv.

CarlaSumoEnv itself stays a plain engine class (no gym/gymnasium dependency)
so it can be used standalone; this wrapper only translates between its API
and the Gymnasium contract so it can be dropped into Stable-Baselines3 (or
any other Gymnasium-based library) unmodified:

    CarlaSumoEnv.reset() -> obs                                (old)
    CarlaSumoGymEnv.reset() -> (obs, info)                      (Gymnasium)

    CarlaSumoEnv.step(action) -> (obs, r, terminated, truncated, info)
    CarlaSumoGymEnv.step(action) -> same 5-tuple, unchanged     (already matches)

Action space      : Box([-1, -1], [1, 1]) — normalized target speed and angle
                     converted through ACC/PID to steer/throttle/brake
Observation space : Box(-1, +1, shape=(70,)) — normalized ego-relative state

In single-worker mode a Gymnasium reset seed is also forwarded to the Python
and NumPy global generators used by the legacy sampling components.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baseline.controllers import (
    ACCController,
    LateralPID,
    LongitudinalPID,
    find_lead_vehicle,
    map_to_heading,
    map_to_speed,
)
from config.scenario_config import ScenarioConfig
from config.scenario_selector import ScenarioSelector, StaticSelector
from env.carla_sumo_env import CarlaSumoEnv, _OBS_DIM
from env.corridor_heading import CorridorHeadingTracker
from logic.termination_checker import _open_corridor_phase


ACTION_CONTRACT_VERSION = "target_speed_base_heading_offset_v1"
DEFAULT_MAX_SPEED_MPS = 13.89
DEFAULT_MAX_HEADING_DEG = 30.0


class CarlaSumoGymEnv(gym.Env):
    """
    Parameters
    ----------
    scenario : str
        Scenario JSON name in config/scenarios/ (default "wz1").
        Ignored if ``scenario_selector`` is provided.
    mode : str
        Default mode passed to CarlaSumoEnv.reset() each episode
        ("train", "eval_45", "eval_50", ... — see OriginDestinationSampler).
        Override per-episode via `reset(options={"mode": "eval_45"})`.
    config : ScenarioConfig | None
        Pass a custom config to bypass the JSON loader entirely.
    scenario_selector : ScenarioSelector | None
        If given, the env calls ``scenario_selector.next()`` at the start of
        every ``reset()`` to pick which scenario to run the upcoming episode
        on (e.g. RoundRobinSelector cycling wz1 → wz2 → wz3).  This lets a
        single env (and single CARLA server) mix multiple work-zones into one
        rollout buffer — the RL algorithm is never aware of the rotation.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 scenario: str = "wz1",
                 mode: str = "train",
                 config: ScenarioConfig | None = None,
                 worker_id: int = 0,
                 no_rendering_mode: bool | None = None,
                 scenario_selector: ScenarioSelector | None = None,
                 episode_setup_callback: Callable[[str], None] | None = None,
                 max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
                 max_heading_deg: float = DEFAULT_MAX_HEADING_DEG) -> None:
        super().__init__()
        self._scenario_selector = scenario_selector or StaticSelector(scenario)
        self._episode_setup_callback = episode_setup_callback
        self.engine = CarlaSumoEnv(
            scenario,
            config,
            worker_id=worker_id,
            no_rendering_mode=no_rendering_mode,
        )
        self._mode = mode
        self._max_speed = float(max_speed_mps)
        self._max_heading = float(max_heading_deg)
        self._acc = ACCController()
        self._lon_pid = LongitudinalPID()
        self._lat_pid = LateralPID()
        self._corridor_heading_tracker: CorridorHeadingTracker | None = None
        self._last_s3_base_heading: float | None = None
        # S2 drives on the legal CARLA lane beside the forbidden work-zone.
        # Its steering reference therefore comes from the exact Driving
        # waypoint under the ego, never from the forbidden polygon's
        # centerline.  Keep the last accepted yaw so CARLA's +/-180 degree
        # representation cannot introduce a discontinuity into the PID.
        self._s2_local_heading_enabled = False
        self._last_s2_base_heading: float | None = None
        # Observation construction and action conversion query the same state.
        # Cache that state's exact controller reference so the stateful curve
        # tracker advances once, not twice, per 10-Hz frame.
        self._base_heading_cache_key: tuple[float, float, float] | None = None
        self._base_heading_cache_value: float | None = None
        self._last_control_diagnostics: dict[str, Any] = {}
        self.engine.set_observation_base_heading_provider(
            self._shared_base_heading_at
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32,
        )

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        mode = (options or {}).get("mode", self._mode)
        self._lon_pid.reset()
        self._lat_pid.reset()
        self._last_control_diagnostics = {}
        # Pick the scenario for the upcoming episode (e.g. round-robin across
        # work-zones) and switch the engine's config before resetting.
        scenario = self._scenario_selector.next()
        self.engine.apply_scenario(scenario)
        self._reset_heading_reference()
        if self._episode_setup_callback is not None:
            self._episode_setup_callback(scenario)
        obs = self.engine.reset(mode=mode)
        cfg = self.engine.cfg
        return obs, {
            "scenario_id": cfg.scenario_id,
            "wz_id": cfg.wz_id,
            "layout_id": cfg.layout_id,
            "setting_id": cfg.setting_id,
            "traffic_backend": cfg.traffic_backend,
            "observation_actor_ids": self.engine.observation_actor_ids,
        }

    def step(self, action: np.ndarray):
        result = self.engine.step(self._to_vehicle_control(action))
        _, _, terminated, truncated, info = result
        info.update(self._last_control_diagnostics)
        if (terminated or truncated) and hasattr(self._scenario_selector, "on_episode_end"):
            self._scenario_selector.on_episode_end(info)
        return result

    def _to_vehicle_control(self, action: np.ndarray) -> np.ndarray:
        """Convert PPO [speed, angle] intent into CARLA low-level control."""
        high_level = np.asarray(action, dtype=np.float32).reshape(-1)
        if high_level.shape != (2,):
            raise ValueError(
                "PPO action must contain exactly [target_speed, heading_adjustment]"
            )
        ego = self.engine.ego
        if ego is None:
            raise RuntimeError("Cannot convert a PPO action before the ego is spawned")

        target_speed = map_to_speed(float(high_level[0]), self._max_speed)
        heading_adjustment = map_to_heading(float(high_level[1]), self._max_heading)
        transform = ego.get_transform()
        velocity = ego.get_velocity()
        ego_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        lead_distance, lead_speed = find_lead_vehicle(
            ego,
            self.engine.get_background_actor_map(),
            self.engine.get_background_speed_map(),
        )
        safe_speed = self._acc.update(target_speed, ego_speed, lead_distance, lead_speed)
        throttle, brake = self._lon_pid.update(safe_speed, ego_speed)
        base_heading = self._shared_base_heading_at(transform)
        target_heading = base_heading + heading_adjustment
        steer = self._lat_pid.update(target_heading, transform.rotation.yaw)
        intervention = max(0.0, float(target_speed) - float(safe_speed))
        self._last_control_diagnostics = {
            "control_policy_target_speed_mps": float(target_speed),
            "control_acc_safe_speed_mps": float(safe_speed),
            "control_acc_intervention_mps": intervention,
            "control_acc_intervened": bool(intervention > 1e-3),
            "control_lead_distance_m": (
                float(lead_distance) if lead_distance is not None else None
            ),
            "control_lead_speed_mps": (
                float(lead_speed) if lead_speed is not None else None
            ),
            "control_ego_speed_mps": float(ego_speed),
            "control_base_heading_deg": float(base_heading),
            "control_heading_adjustment_deg": float(heading_adjustment),
            "control_target_heading_deg": float(target_heading),
            "control_steer": float(steer),
            "control_throttle": float(throttle),
            "control_brake": float(brake),
        }
        self.engine.set_observation_previous_action(
            float(high_level[0]), float(high_level[1])
        )
        return np.asarray([steer, throttle, brake], dtype=np.float32)

    def _reset_heading_reference(self) -> None:
        """Reset scenario-specific, episode-local heading references.

        S3 follows its owner-authored safe-corridor centerline.  S2 is
        intentionally different: the polygon describes *forbidden* space, so
        its centerline must never become a steering target.  S2 instead uses
        the exact CARLA Driving waypoint beneath the ego at every control step.
        """
        workzone = self.engine.cfg.workzone
        if (
            workzone.geometry_mode == "safe_corridor"
            and workzone.corridor_open_ends
            and workzone.corridor_left_boundary_points
            and workzone.corridor_right_boundary_points
        ):
            self._corridor_heading_tracker = CorridorHeadingTracker(
                workzone.corridor_left_boundary_points,
                workzone.corridor_right_boundary_points,
                lookahead_m=4.0,
            )
        else:
            self._corridor_heading_tracker = None
        self._last_s3_base_heading = None

        self._s2_local_heading_enabled = (
            self.engine.cfg.scenario_id == "s2"
            and workzone.geometry_mode == "forbidden_polygon"
        )
        self._last_s2_base_heading = None
        self._base_heading_cache_key = None
        self._base_heading_cache_value = None

    @staticmethod
    def _heading_state_key(transform: Any) -> tuple[float, float, float]:
        return (
            round(float(transform.location.x), 6),
            round(float(transform.location.y), 6),
            round(float(transform.rotation.yaw), 6),
        )

    def _shared_base_heading_at(self, transform: Any) -> float:
        """Return one exact controller reference per physical ego state."""
        key = self._heading_state_key(transform)
        if key == self._base_heading_cache_key:
            assert self._base_heading_cache_value is not None
            return self._base_heading_cache_value
        heading = float(self._base_heading_at(transform))
        self._base_heading_cache_key = key
        self._base_heading_cache_value = heading
        return heading

    def _base_heading_at(self, transform: Any) -> float:
        """Return the active scenario's safe steering-heading reference.

        S3: before the corridor uses the owner-authored initial heading,
        inside uses its safe centerline, and after the exit uses an exact
        CARLA Driving waypoint.  S2: uses exact local Driving waypoints for
        the entire episode because its polygon centerline is forbidden space.
        Every other scenario retains its configured fixed heading.
        """
        configured = float(self.engine.cfg.carla.road_heading_deg)
        tracker = self._corridor_heading_tracker
        if tracker is None:
            if not getattr(self, "_s2_local_heading_enabled", False):
                return configured
            return self._s2_base_heading_at(transform, configured)

        workzone = self.engine.cfg.workzone
        left = workzone.corridor_left_boundary_points
        right = workzone.corridor_right_boundary_points
        if not left or not right:
            return configured

        location = transform.location
        phase = _open_corridor_phase(
            (float(location.x), float(location.y)), left, right
        )
        if phase == "before":
            base_heading = configured
        elif phase == "inside":
            base_heading = tracker.heading_at(location.x, location.y)
        else:
            base_heading = self.engine.driving_heading_at(location)
            if base_heading is None:
                base_heading = (
                    self._last_s3_base_heading
                    if self._last_s3_base_heading is not None
                    else configured
                )

        # Keep equivalent CARLA yaws (e.g. 0.6 and -359.4) continuous for PID.
        if self._last_s3_base_heading is not None:
            delta = (
                float(base_heading) - self._last_s3_base_heading + 180.0
            ) % 360.0 - 180.0
            base_heading = self._last_s3_base_heading + delta
        self._last_s3_base_heading = float(base_heading)
        return float(base_heading)

    def _s2_base_heading_at(self, transform: Any, configured: float) -> float:
        """Return S2's exact local CARLA-lane yaw with a safe fallback.

        ``driving_heading_at`` deliberately calls CARLA with
        ``project_to_road=False``.  Thus an ego which has already left a
        Driving lane cannot silently snap to a nearby/opposing lane.  Missing
        waypoints retain the most recent valid reference (or the configured
        episode heading before the first valid sample).

        A near-opposite (>=135 degree) one-step change is rejected as an
        adjacent/opposing-lane lookup.  Genuine curve evolution at 10 Hz is
        continuous and far smaller; accepted yaws are unwrapped around the
        previous value before reaching the lateral PID.
        """
        local_heading = self.engine.driving_heading_at(transform.location)
        last_heading = getattr(self, "_last_s2_base_heading", None)
        fallback = (
            last_heading
            if last_heading is not None
            else configured
        )
        if local_heading is None:
            return float(fallback)

        reference = float(fallback)
        delta = (float(local_heading) - reference + 180.0) % 360.0 - 180.0
        if abs(delta) >= 135.0:
            return float(fallback)

        base_heading = reference + delta
        self._last_s2_base_heading = float(base_heading)
        return float(base_heading)

    def close(self) -> None:
        self.engine.close()

    def runtime_diagnostics(self) -> dict[str, Any]:
        """Expose the worker-local snapshot to vector-env tooling."""
        return self.engine.runtime_diagnostics()
