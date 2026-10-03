"""PPO V2 environment variant using the previous ego-centre judgement.

Observation, controller, SUMO synchronization, reward calculation, scenario
geometry, and simulator lifecycle are inherited unchanged from
``CarlaSumoEnvV2``.  Only the checker selected during reset differs.
"""
from __future__ import annotations

# Import V2 first: it pins the matching SUMO 1.27.1 Python tools before the
# shared engine imports ``traci``.  Center validation imports this module in a
# fresh process, so this ordering is part of the portable runtime contract.
from env.carla_sumo_env_v2 import CarlaSumoEnvV2, _OBS_DIM
from env import carla_sumo_env as legacy_env
from env.carla_sumo_env import CarlaSumoEnv
from logic.episode_termination_center_v2 import (
    CenterPointEpisodeTerminationCheckerV2,
)


class CarlaSumoEnvCenterV2(CarlaSumoEnvV2):
    """Current PPO V2 environment with centre-point work-zone judgement."""

    def _reset_once(self, mode: str):
        self._reward_v2 = None
        previous_checker = legacy_env.EpisodeTerminationChecker
        legacy_env.EpisodeTerminationChecker = CenterPointEpisodeTerminationCheckerV2
        try:
            # Skip CarlaSumoEnvV2._reset_once because that method deliberately
            # installs the swept-OBB checker.  The shared legacy lifecycle is
            # otherwise exactly the same implementation used by V2.
            return CarlaSumoEnv._reset_once(self, mode)
        finally:
            legacy_env.EpisodeTerminationChecker = previous_checker


__all__ = ["CarlaSumoEnvCenterV2", "_OBS_DIM"]
