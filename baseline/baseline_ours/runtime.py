"""Small shared helpers for the PPO command-line entry points."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from stable_baselines3 import PPO

from baseline.baseline_ours.rollout_plan import RolloutPlan, choose_batch_size
from config.scenario_catalog import list_setting_ids
from config.scenario_config import ScenarioConfig

PPO_GAMMA = 0.997
PPO_GAE_LAMBDA = 0.995


def configure_initial_config(
    config: ScenarioConfig,
    *,
    carla_port: int,
    tm_port: int,
    sumo_port: int,
    no_rendering: bool,
) -> None:
    """Apply process-local runtime ports to an environment's first config."""
    config.carla.port = int(carla_port)
    config.carla.tm_port = int(tm_port)
    config.carla.no_rendering_mode = bool(no_rendering)
    if config.sumo is not None:
        config.sumo.port = int(sumo_port)


def build_or_load_ppo(
    env: Any,
    plan: RolloutPlan,
    *,
    model_path: str | None,
    ppo_epochs: int,
    seed: int,
    device: str = "auto",
) -> PPO:
    """Create PPO or load a compatible model with the requested rollout shape."""
    kwargs = {
        "n_steps": plan.n_steps_per_env,
        "batch_size": choose_batch_size(plan.buffer_size_steps),
        "n_epochs": ppo_epochs,
        "gamma": PPO_GAMMA,
        "gae_lambda": PPO_GAE_LAMBDA,
        "device": device,
    }
    if model_path:
        return PPO.load(model_path, env=env, **kwargs)
    return PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        **kwargs,
    )


def save_ppo(model: PPO, output: str | Path) -> None:
    """Create the output directory and save using SB3's normal path handling."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))


def collect_settings(scenarios: Sequence[str]) -> list[str]:
    """Collect runnable settings without changing caller or manifest order."""
    settings: list[str] = []
    for scenario in scenarios:
        settings.extend(list_setting_ids(scenario, runnable_only=True))
    if not settings:
        raise ValueError("A server group has no runnable settings")
    return settings


__all__ = [
    "PPO_GAMMA",
    "PPO_GAE_LAMBDA",
    "build_or_load_ppo",
    "collect_settings",
    "configure_initial_config",
    "save_ppo",
]
