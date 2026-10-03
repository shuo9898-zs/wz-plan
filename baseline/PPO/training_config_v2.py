"""Single source of truth for the official three-server PPO V2 training run.

Edit the defaults in this file when defining a new formal training treatment.
Run scale, rollout, port, seed, device, and KL/epoch/batch values also remain
overridable from ``train_three_servers_v2``'s command line.  The remaining PPO
optimizer values are deliberately changed here so they cannot drift silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class PPOOptimizationConfigV2:
    """PPO/GAE values applied explicitly when constructing the SB3 model."""

    gamma: float = 0.999
    gae_lambda: float = 0.995
    learning_rate: float = 3.0e-4
    batch_size: int = 512
    max_epochs_per_update: int = 20
    target_kl: float | None = None
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    normalize_advantage: bool = True
    entropy_coefficient: float = 0.0
    value_function_coefficient: float = 0.5
    max_gradient_norm: float = 0.5


@dataclass(frozen=True)
class TownWorkerConfigV2:
    """One fixed-Town rollout worker and its per-update valid-step quota."""

    town: str
    scenarios: Tuple[str, ...]
    rollout_steps: int
    carla_port: int
    traffic_manager_port: int
    sumo_port: int


DEFAULT_TOWN_WORKERS_V2: Tuple[TownWorkerConfigV2, ...] = (
    TownWorkerConfigV2(
        town="Town02",
        scenarios=("s1", "s6"),
        rollout_steps=32_400,
        carla_port=2000,
        traffic_manager_port=8000,
        sumo_port=8813,
    ),
    TownWorkerConfigV2(
        town="Town05",
        scenarios=("s2", "s5"),
        rollout_steps=43_000,
        carla_port=2020,
        traffic_manager_port=8020,
        sumo_port=8833,
    ),
    TownWorkerConfigV2(
        town="Town10HD",
        scenarios=("s3", "s4"),
        rollout_steps=27_000,
        carla_port=2040,
        traffic_manager_port=8040,
        sumo_port=8853,
    ),
)


@dataclass(frozen=True)
class ThreeServerTrainingConfigV2:
    """Formal run-level, rollout, server, and optimizer defaults."""

    # Training scale: 25 synchronized buffers and therefore 25 policy updates.
    total_updates: int = 25
    episodes_per_origin: int = 1
    seed: int = 7
    device: str = "auto"

    # Runtime/logging defaults.
    run_root: Path = Path("runs/ppo_v2_three_servers")
    live_log_every_steps: int = 1_000
    carla_host: str = "127.0.0.1"
    map_loader: Path = Path(__file__).resolve().parents[2] / "set_carla_map.py"
    map_rpc_timeout_s: float = 60.0
    map_ready_sleep_s: float = 30.0
    no_rendering: bool = True

    ppo: PPOOptimizationConfigV2 = PPOOptimizationConfigV2()
    town_workers: Tuple[TownWorkerConfigV2, ...] = DEFAULT_TOWN_WORKERS_V2

    @property
    def steps_per_update(self) -> int:
        return sum(worker.rollout_steps for worker in self.town_workers)

    @property
    def total_valid_training_steps(self) -> int:
        return self.total_updates * self.steps_per_update

    @property
    def town_step_quotas(self) -> Dict[str, int]:
        return {worker.town: worker.rollout_steps for worker in self.town_workers}

    @property
    def phases(self) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        return tuple((worker.town, worker.scenarios) for worker in self.town_workers)


DEFAULT_THREE_SERVER_TRAINING_V2 = ThreeServerTrainingConfigV2()


__all__ = [
    "DEFAULT_THREE_SERVER_TRAINING_V2",
    "DEFAULT_TOWN_WORKERS_V2",
    "PPOOptimizationConfigV2",
    "ThreeServerTrainingConfigV2",
    "TownWorkerConfigV2",
]
