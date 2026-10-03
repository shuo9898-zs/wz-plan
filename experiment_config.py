"""Single source of truth for the portable Aug-24 PPO experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# Everything produced or consumed by this experiment is below this directory.
BUNDLE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = BUNDLE_ROOT


@dataclass(frozen=True)
class TownConfig:
    town: str
    train_scenarios: Tuple[str, ...]
    validation_scenarios: Tuple[str, ...]
    rollout_steps: int
    carla_port: int
    tm_port: int
    sumo_port: int


TOWNS: Tuple[TownConfig, ...] = (
    TownConfig("Town02", ("s1", "s6"), ("s5", "s6"), 32_400, 2000, 8000, 8813),
    TownConfig("Town05", ("s2", "s5"), ("s1", "s2"), 43_000, 2020, 8020, 8833),
    TownConfig("Town10HD", ("s3", "s4"), ("s3", "s4"), 27_000, 2040, 8040, 8853),
)

ENCODER_NAME = "Encoder1/CrossAttentionEncoderV2"
ENCODER_OBSERVATION_DIM = 1_311
ENCODER_FEATURES_DIM = 256
ENCODER_TRAINABLE_POLICY_PARAMETERS = 1_103_109

TOTAL_UPDATES = 25
STEPS_PER_UPDATE = sum(item.rollout_steps for item in TOWNS)
TOTAL_TRAINING_STEPS = TOTAL_UPDATES * STEPS_PER_UPDATE
BATCH_SIZE = 512
MAX_EPOCHS_PER_UPDATE = 20
LEARNING_RATE = 3.0e-4
GAMMA = 0.999
GAE_LAMBDA = 0.995
SEED = 7

VALIDATION_EPISODES_PER_ORIGIN = 10
VALIDATION_TOTAL_EPISODES = 6 * 3 * VALIDATION_EPISODES_PER_ORIGIN
VALIDATION_DEVICE = "cpu"
VALIDATION_DETERMINISTIC = True
VALIDATION_SPAWN_PROPS = False

DEFAULT_RUN_ROOT = BUNDLE_ROOT / "runs"
MAP_LOADER = BUNDLE_ROOT / "set_carla_map.py"
VALIDATION_BEST_NAME = "validation_best_model.zip"


__all__ = [
    "BATCH_SIZE", "BUNDLE_ROOT", "DEFAULT_RUN_ROOT", "ENCODER_FEATURES_DIM",
    "ENCODER_NAME", "ENCODER_OBSERVATION_DIM", "ENCODER_TRAINABLE_POLICY_PARAMETERS",
    "GAE_LAMBDA", "GAMMA", "LEARNING_RATE", "MAP_LOADER",
    "MAX_EPOCHS_PER_UPDATE", "SEED", "STEPS_PER_UPDATE",
    "TOTAL_TRAINING_STEPS", "TOTAL_UPDATES", "TOWNS", "VALIDATION_BEST_NAME",
    "VALIDATION_DETERMINISTIC", "VALIDATION_DEVICE",
    "VALIDATION_EPISODES_PER_ORIGIN", "VALIDATION_SPAWN_PROPS",
    "VALIDATION_TOTAL_EPISODES", "WORKSPACE_ROOT",
]
