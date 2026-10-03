"""Thin Stable-Baselines3 adapter for the algorithm-independent encoder."""
from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_SPEC_V2,
    ObservationSpecV2,
)
from models.scene_encoder_v2 import (
    DEFAULT_SCENE_ENCODER_SPEC_V2,
    SceneEncoderMixinV2,
    SceneEncoderSpecV2,
)


class CrossAttentionFeatureExtractorV2(
    SceneEncoderMixinV2,
    BaseFeaturesExtractor,
):
    """Expose :class:`SceneEncoderV2` through SB3's extractor interface."""

    def __init__(
        self,
        observation_space: gym.Space,
        observation: ObservationSpecV2 = DEFAULT_OBSERVATION_SPEC_V2,
        network: SceneEncoderSpecV2 = DEFAULT_SCENE_ENCODER_SPEC_V2,
    ) -> None:
        expected_shape = (observation.observation_dim,)
        if observation_space.shape != expected_shape:
            raise ValueError(
                "RL V2 encoder expected observation shape "
                f"{expected_shape}, got {observation_space.shape}"
            )
        BaseFeaturesExtractor.__init__(
            self,
            observation_space,
            features_dim=network.features_dim,
        )
        self._initialize_scene_encoder_v2(observation, network)


__all__ = ["CrossAttentionFeatureExtractorV2"]

