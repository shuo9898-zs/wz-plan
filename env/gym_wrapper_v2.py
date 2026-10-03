"""Official Gymnasium adapter for the PPO V2 environment and controller."""
from __future__ import annotations

import math
import random
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baseline.controllers_v2 import (
    CONTROLLER_CONTRACT_VERSION_V2,
    ControllerConfigV2,
    DualPIDControllerV2,
)
from config.scenario_config import ScenarioConfig
from config.scenario_selector import ScenarioSelector, StaticSelector
from env.carla_sumo_env_v2 import CarlaSumoEnvV2, _OBS_DIM


ACTION_CONTRACT_VERSION = CONTROLLER_CONTRACT_VERSION_V2
DEFAULT_MAX_SPEED_MPS = 13.89
DEFAULT_MAX_YAW_RATE_DEG_S = 30.0


class CarlaSumoGymEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "wz1",
        mode: str = "train",
        config: ScenarioConfig | None = None,
        worker_id: int = 0,
        no_rendering_mode: bool | None = None,
        scenario_selector: ScenarioSelector | None = None,
        episode_setup_callback: Callable[[str], None] | None = None,
        max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
        max_heading_deg: float = DEFAULT_MAX_YAW_RATE_DEG_S,
    ) -> None:
        super().__init__()
        self._scenario_selector = scenario_selector or StaticSelector(scenario)
        self._episode_setup_callback = episode_setup_callback
        self.engine = CarlaSumoEnvV2(
            scenario,
            config,
            worker_id=worker_id,
            no_rendering_mode=no_rendering_mode,
        )
        self._mode = mode
        self._controller_v2 = DualPIDControllerV2(ControllerConfigV2(
            dt_s=float(self.engine.cfg.episode.sim_dt),
            max_speed_mps=float(max_speed_mps),
            max_yaw_rate_deg_s=float(max_heading_deg),
        ))
        self._last_control_diagnostics: dict[str, Any] = {}
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        mode = (options or {}).get("mode", self._mode)
        self._last_control_diagnostics = {}
        scenario = self._scenario_selector.next()
        self.engine.apply_scenario(scenario)
        if self._episode_setup_callback is not None:
            self._episode_setup_callback(scenario)
        obs = self.engine.reset(mode=mode)
        ego = self.engine.ego
        if ego is None:
            raise RuntimeError("PPO V2 reset completed without an ego actor")
        self._controller_v2.reset(
            initial_heading_deg=float(ego.get_transform().rotation.yaw)
        )
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
        high_level = np.asarray(action, dtype=np.float32).reshape(-1)
        if high_level.shape != (2,):
            raise ValueError(
                "PPO V2 action must contain [target_speed, target_yaw_rate]"
            )
        ego = self.engine.ego
        if ego is None:
            raise RuntimeError("Cannot convert a PPO action before ego spawn")
        transform = ego.get_transform()
        velocity = ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        control = self._controller_v2.update(
            high_level,
            measured_speed_mps=speed,
            current_heading_deg=float(transform.rotation.yaw),
        )
        self._last_control_diagnostics = {
            "control_policy_target_speed_mps": control.target_speed_mps,
            "control_target_yaw_rate_deg_s": control.target_yaw_rate_deg_s,
            "control_measured_yaw_rate_deg_s": control.measured_yaw_rate_deg_s,
            "control_speed_error_mps": control.speed_error_mps,
            "control_yaw_rate_error_deg_s": control.yaw_rate_error_deg_s,
            "control_ego_speed_mps": control.measured_speed_mps,
            "control_acc_intervened": False,
            "control_steer": control.steer,
            "control_throttle": control.throttle,
            "control_brake": control.brake,
        }
        return np.asarray(
            [control.steer, control.throttle, control.brake], dtype=np.float32,
        )

    def close(self) -> None:
        self.engine.close()


# Preserve the constructor name expected by the existing PPO orchestration.
CarlaSumoGymEnv = CarlaSumoGymEnvV2

__all__ = [
    "ACTION_CONTRACT_VERSION",
    "CarlaSumoGymEnv",
    "CarlaSumoGymEnvV2",
]
