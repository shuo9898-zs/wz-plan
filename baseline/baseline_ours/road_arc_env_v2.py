"""Thin environment adapter that swaps only baseline_ours reward handling."""
from __future__ import annotations

import math
from typing import Any

import env.gym_wrapper_v2 as base_gym_wrapper
from env import carla_sumo_env as legacy_env
from env.carla_sumo_env import CarlaSumoEnv
from env.carla_sumo_env_v2 import CarlaSumoEnvV2
from logic.episode_termination_center_v2 import CenterPointEpisodeTerminationCheckerV2
from logic.scenario_geometry_adapter_v2 import finish_line_from_config_v2

from baseline.baseline_ours.road_arc_progress_reward import (
    RUNNING_V2,
    RewardCalculatorV2,
    RewardConfigV2,
)
from baseline.baseline_ours.road_arc_reference import reference_for_setting


REWARD_SOURCE = "road_arc_high_water_no_speed_bonus_r1"


class CarlaSumoEnvRoadArcV2(CarlaSumoEnvV2):
    """Original PPO V2 engine with road-arc progress and no speed bonus."""

    def _initialize_reward_progress(self) -> None:
        if self._ego is None or self._destination_transform is None:
            raise RuntimeError("Cannot initialize road-arc reward before ego/destination")
        location = self._ego.get_transform().location
        finish = finish_line_from_config_v2(self.cfg)
        self._reward_v2 = RewardCalculatorV2(
            origin=(float(location.x), float(location.y)),
            finish_line=(finish.first, finish.second),
            reference=reference_for_setting(self.cfg.setting_id),
            config=RewardConfigV2(
                progress_budget=50.0,
                timeout_penalty=-50.0,
                goal_reward=100.0,
                failure_penalty=-100.0,
            ),
        )
        self._reward_progress_source = REWARD_SOURCE

    def _compute_reward(
        self,
        terminated: bool,
        truncated: bool,
        success: bool,
        info: dict[str, Any],
    ) -> float:
        del success
        if self._reward_v2 is None or self._ego is None:
            raise RuntimeError("Road-arc reward is not initialized")
        location = self._ego.get_transform().location
        velocity = self._ego.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        reason = info.get("reason") if (terminated or truncated) else RUNNING_V2
        result = self._reward_v2.step(
            (float(location.x), float(location.y)),
            speed_mps=float(speed_mps),
            terminal_reason=reason,
        )
        progress_reward = result.reward if result.reason == RUNNING_V2 else 0.0
        info["reward_progress_source"] = REWARD_SOURCE
        info["reward_progress_bonus"] = float(progress_reward)
        info["reward_progress_fraction"] = float(result.best_progress)
        info["reward_normalized_progress"] = float(result.normalized_progress)
        info["reward_cumulative_progress"] = float(
            result.cumulative_progress_reward
        )
        info["reward_step_cost"] = 0.0
        info["reward_average_speed_mps"] = float(result.average_speed_mps)
        info["reward_speed_bonus"] = 0.0
        info["reward_terminal_base"] = float(result.terminal_base_reward)
        info["reward_total"] = float(result.reward)
        return float(result.reward)


class CarlaSumoEnvRoadArcCenterV2(CarlaSumoEnvRoadArcV2):
    """Optional centre-point judgement with the same road-arc reward."""

    def _reset_once(self, mode: str):
        self._reward_v2 = None
        previous_checker = legacy_env.EpisodeTerminationChecker
        legacy_env.EpisodeTerminationChecker = CenterPointEpisodeTerminationCheckerV2
        try:
            return CarlaSumoEnv._reset_once(self, mode)
        finally:
            legacy_env.EpisodeTerminationChecker = previous_checker


class CarlaSumoGymEnvV2(base_gym_wrapper.CarlaSumoGymEnvV2):
    """Reuse the unchanged Gym/controller adapter with the local engine."""

    ENGINE_CLASS = CarlaSumoEnvRoadArcV2

    def __init__(self, *args, **kwargs) -> None:
        original_engine = base_gym_wrapper.CarlaSumoEnvV2
        base_gym_wrapper.CarlaSumoEnvV2 = self.ENGINE_CLASS
        try:
            super().__init__(*args, **kwargs)
        finally:
            base_gym_wrapper.CarlaSumoEnvV2 = original_engine


class CarlaSumoGymEnvCenterV2(CarlaSumoGymEnvV2):
    ENGINE_CLASS = CarlaSumoEnvRoadArcCenterV2


CarlaSumoGymEnv = CarlaSumoGymEnvV2


__all__ = [
    "CarlaSumoEnvRoadArcV2",
    "CarlaSumoEnvRoadArcCenterV2",
    "CarlaSumoGymEnv",
    "CarlaSumoGymEnvCenterV2",
    "CarlaSumoGymEnvV2",
    "REWARD_SOURCE",
]
