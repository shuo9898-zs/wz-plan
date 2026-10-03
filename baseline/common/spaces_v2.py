"""Algorithm-neutral Gym spaces for model construction and offline checks."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_SPEC_V2,
    ObservationSpecV2,
)


ACTION_DIM_V2 = 2


class RLSpaceOnlyEnvV2(gym.Env):
    """Expose V2 spaces without importing or connecting to CARLA/SUMO."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        observation: ObservationSpecV2 = DEFAULT_OBSERVATION_SPEC_V2,
    ) -> None:
        self.observation_spec = observation
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation.observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM_V2,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError("The space-only environment cannot be stepped")


__all__ = ["ACTION_DIM_V2", "RLSpaceOnlyEnvV2"]

