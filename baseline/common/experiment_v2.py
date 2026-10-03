"""Algorithm-neutral observation/encoder experiment contract.

Algorithm-specific hyperparameters remain in the PPO/SAC/TD3 runtimes.  This
contract owns only the input schema and feature extractor, allowing a fair
algorithm comparison to reuse identical observations and representation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict

from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_SPEC_V2,
    ObservationSpecV2,
)
from models.sb3_extractor_v2 import CrossAttentionFeatureExtractorV2
from models.scene_encoder_v2 import (
    DEFAULT_SCENE_ENCODER_SPEC_V2,
    SceneEncoderSpecV2,
)


SUPPORTED_RL_BASELINES_V2 = ("ppo", "sac", "td3")
EXPERIMENT_COMPONENT_CONTRACT_V2 = "rl_observation_encoder_components_v2"


@dataclass(frozen=True)
class RLExperimentComponentsV2:
    algorithm: str
    observation: ObservationSpecV2 = field(
        default_factory=lambda: DEFAULT_OBSERVATION_SPEC_V2
    )
    encoder: SceneEncoderSpecV2 = field(
        default_factory=lambda: DEFAULT_SCENE_ENCODER_SPEC_V2
    )

    def __post_init__(self) -> None:
        normalized = str(self.algorithm).strip().lower()
        if normalized not in SUPPORTED_RL_BASELINES_V2:
            raise ValueError(
                f"algorithm must be one of {SUPPORTED_RL_BASELINES_V2}, got {self.algorithm!r}"
            )
        object.__setattr__(self, "algorithm", normalized)

    def sb3_feature_policy_kwargs(self, *, share: bool = False) -> Dict:
        return {
            "features_extractor_class": CrossAttentionFeatureExtractorV2,
            "features_extractor_kwargs": {
                "observation": self.observation,
                "network": self.encoder,
            },
            "share_features_extractor": bool(share),
        }

    def fingerprint_payload(self) -> dict:
        return {
            "contract": EXPERIMENT_COMPONENT_CONTRACT_V2,
            "algorithm": self.algorithm,
            "observation": asdict(self.observation),
            "observation_dim": self.observation.observation_dim,
            "encoder": self.encoder.fingerprint_payload(),
        }


__all__ = [
    "EXPERIMENT_COMPONENT_CONTRACT_V2",
    "RLExperimentComponentsV2",
    "SUPPORTED_RL_BASELINES_V2",
]

