"""Gymnasium adapter for the controlled centre-ego PPO V2 experiment."""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baseline.controllers_v2 import ControllerConfigV2, DualPIDControllerV2
from config.scenario_config import ScenarioConfig
from config.scenario_selector import ScenarioSelector, StaticSelector
from env.carla_sumo_env_center_v2 import CarlaSumoEnvCenterV2, _OBS_DIM
from env.gym_wrapper_v2 import (
    DEFAULT_MAX_SPEED_MPS,
    DEFAULT_MAX_YAW_RATE_DEG_S,
    CarlaSumoGymEnvV2,
)


class CarlaSumoGymEnvCenterV2(CarlaSumoGymEnvV2):
    """Same observation/action/controller contract with a centre-ego engine."""

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
        # CarlaSumoGymEnvV2 hard-codes its engine class, so initialize the
        # identical public contract here and select only the centre variant.
        gym.Env.__init__(self)
        self._scenario_selector = scenario_selector or StaticSelector(scenario)
        self._episode_setup_callback = episode_setup_callback
        self.engine = CarlaSumoEnvCenterV2(
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
        self._last_control_diagnostics = {}
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(_OBS_DIM,),
            dtype=np.float32,
        )


CarlaSumoGymEnv = CarlaSumoGymEnvCenterV2


__all__ = ["CarlaSumoGymEnv", "CarlaSumoGymEnvCenterV2"]
