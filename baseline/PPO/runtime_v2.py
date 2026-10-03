"""PPO V2 hyperparameter contract and physical horizon diagnostics.

The official PPO entry points use the model-construction helpers in this
module.  They can also include
:meth:`PPOHyperparameterSpecV2.checkpoint_fingerprint_payload` in a V2-only
checkpoint plan.

The checkpoint payload is deliberately namespaced as ``ppo_v2`` and explicitly
declares legacy checkpoints incompatible.  It must not be substituted into, or
compared with, the legacy ``baseline.PPO.runtime`` checkpoint contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Union

from stable_baselines3 import PPO
from torch import nn

from baseline.PPO.encoder_v2 import CrossAttentionEncoderV2
from baseline.PPO.rollout_plan import RolloutPlan, choose_batch_size
from baseline.PPO.training_config_v2 import DEFAULT_THREE_SERVER_TRAINING_V2
from config.scenario_catalog import list_setting_ids
from config.scenario_config import ScenarioConfig


_PPO_DEFAULTS_V2 = DEFAULT_THREE_SERVER_TRAINING_V2.ppo
V2_PPO_GAMMA = _PPO_DEFAULTS_V2.gamma
V2_PPO_GAE_LAMBDA = _PPO_DEFAULTS_V2.gae_lambda
V2_CONTROL_DT_S = 0.1
V2_MAX_EPISODE_STEPS = 600
V2_DEFAULT_PPO_EPOCHS = _PPO_DEFAULTS_V2.max_epochs_per_update
PPO_GAMMA = V2_PPO_GAMMA
PPO_GAE_LAMBDA = V2_PPO_GAE_LAMBDA

V2_PPO_HYPERPARAMETER_CONTRACT = "ppo_hyperparameters_v2"
V2_PPO_CHECKPOINT_NAMESPACE = "ppo_v2"
V2_PPO_CHECKPOINT_COMPATIBILITY = "v2_only"
V2_POLICY_HIDDEN_SIZES = (256, 256)
V2_POLICY_ACTIVATION = "tanh"
V2_POLICY_TRAINABLE_PARAMETERS = 1_103_109


@dataclass(frozen=True)
class PPOHorizonDiagnosticsV2:
    """Physical interpretation of discount and GAE decay at one control rate."""

    episode_duration_s: float
    discount_factor_at_episode_end: float
    discount_e_folding_s: float
    discount_half_life_s: float
    discount_mass_horizon_s: float
    gae_trace_factor: float
    gae_trace_factor_at_episode_end: float
    gae_trace_e_folding_s: float
    gae_trace_half_life_s: float
    gae_trace_mass_horizon_s: float


@dataclass(frozen=True)
class PPOHyperparameterSpecV2:
    """Validated PPO V2 discount/trace contract.

    ``gamma`` defines the discounted control objective.  GAE propagates TD
    residuals with the combined per-step factor ``gamma * gae_lambda``.
    ``control_dt_s`` converts both dimensionless factors into physical time.
    """

    gamma: float = V2_PPO_GAMMA
    gae_lambda: float = V2_PPO_GAE_LAMBDA
    control_dt_s: float = V2_CONTROL_DT_S
    max_episode_steps: int = V2_MAX_EPISODE_STEPS
    contract_version: str = field(
        default=V2_PPO_HYPERPARAMETER_CONTRACT,
        init=False,
    )
    checkpoint_namespace: str = field(
        default=V2_PPO_CHECKPOINT_NAMESPACE,
        init=False,
    )

    def __post_init__(self) -> None:
        gamma = _finite_float(self.gamma, "gamma")
        gae_lambda = _finite_float(self.gae_lambda, "gae_lambda")
        control_dt_s = _finite_float(self.control_dt_s, "control_dt_s")
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be strictly between 0 and 1")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if (
            isinstance(self.max_episode_steps, bool)
            or not isinstance(self.max_episode_steps, int)
            or self.max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be a positive integer")

        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "gae_lambda", gae_lambda)
        object.__setattr__(self, "control_dt_s", control_dt_s)

    @property
    def episode_duration_s(self) -> float:
        return self.control_dt_s * self.max_episode_steps

    @property
    def gae_trace_factor(self) -> float:
        return self.gamma * self.gae_lambda

    def sb3_kwargs(self) -> Dict[str, float]:
        """Return only the two kwargs owned by this V2 contract."""
        return {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
        }

    def horizon_diagnostics(self) -> PPOHorizonDiagnosticsV2:
        """Return discount and trace horizons in seconds.

        ``*_mass_horizon_s`` is ``dt / (1 - factor)``: the time-scaled mass of
        an infinite geometric series, not a hard cutoff.  Episode-end factors
        use exactly ``max_episode_steps`` decays for a stable contract metric.
        """
        discount_e_folding, discount_half_life, discount_mass = _decay_horizons(
            self.gamma,
            self.control_dt_s,
        )
        trace = self.gae_trace_factor
        trace_e_folding, trace_half_life, trace_mass = _decay_horizons(
            trace,
            self.control_dt_s,
        )
        return PPOHorizonDiagnosticsV2(
            episode_duration_s=self.episode_duration_s,
            discount_factor_at_episode_end=self.gamma ** self.max_episode_steps,
            discount_e_folding_s=discount_e_folding,
            discount_half_life_s=discount_half_life,
            discount_mass_horizon_s=discount_mass,
            gae_trace_factor=trace,
            gae_trace_factor_at_episode_end=trace ** self.max_episode_steps,
            gae_trace_e_folding_s=trace_e_folding,
            gae_trace_half_life_s=trace_half_life,
            gae_trace_mass_horizon_s=trace_mass,
        )

    def checkpoint_fingerprint_payload(
        self,
    ) -> Dict[str, Union[str, bool, Dict[str, Union[float, int]]]]:
        """Return a V2-only payload for a future checkpoint fingerprint.

        The namespace and explicit compatibility marker prevent a caller from
        silently treating a legacy PPO checkpoint as a V2 checkpoint.
        """
        return {
            "checkpoint_namespace": self.checkpoint_namespace,
            "checkpoint_compatibility": V2_PPO_CHECKPOINT_COMPATIBILITY,
            "legacy_checkpoint_compatible": False,
            "contract_version": self.contract_version,
            "ppo_v2": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "control_dt_s": self.control_dt_s,
                "max_episode_steps": self.max_episode_steps,
            },
        }


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _decay_horizons(factor: float, control_dt_s: float) -> tuple[float, float, float]:
    if factor == 0.0:
        return 0.0, 0.0, control_dt_s
    e_folding = -control_dt_s / math.log(factor)
    half_life = -control_dt_s * math.log(2.0) / math.log(factor)
    mass_horizon = control_dt_s / (1.0 - factor)
    return e_folding, half_life, mass_horizon


DEFAULT_PPO_HYPERPARAMETERS_V2 = PPOHyperparameterSpecV2()


def configure_initial_config(
    config: ScenarioConfig,
    *,
    carla_port: int,
    tm_port: int,
    sumo_port: int,
    no_rendering: bool,
) -> None:
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
    batch_size: int | None = None,
    target_kl: float | None = None,
) -> PPO:
    """Create/load the official PPO model with the V2 horizon contract."""
    resolved_batch_size = (
        choose_batch_size(plan.buffer_size_steps)
        if batch_size is None
        else int(batch_size)
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive")
    if target_kl is not None:
        target_kl = float(target_kl)
        if not math.isfinite(target_kl) or target_kl <= 0.0:
            raise ValueError("target_kl must be finite and positive")
    kwargs = {
        "n_steps": plan.n_steps_per_env,
        "batch_size": resolved_batch_size,
        "n_epochs": ppo_epochs,
        "gamma": V2_PPO_GAMMA,
        "gae_lambda": V2_PPO_GAE_LAMBDA,
        # Keep the full optimizer contract explicit instead of silently
        # inheriting whichever defaults happen to ship with the installed SB3.
        "learning_rate": _PPO_DEFAULTS_V2.learning_rate,
        "clip_range": _PPO_DEFAULTS_V2.clip_range,
        "clip_range_vf": _PPO_DEFAULTS_V2.clip_range_vf,
        "normalize_advantage": _PPO_DEFAULTS_V2.normalize_advantage,
        "ent_coef": _PPO_DEFAULTS_V2.entropy_coefficient,
        "vf_coef": _PPO_DEFAULTS_V2.value_function_coefficient,
        "max_grad_norm": _PPO_DEFAULTS_V2.max_gradient_norm,
        "device": device,
        "target_kl": target_kl,
        "policy_kwargs": {
            "features_extractor_class": CrossAttentionEncoderV2,
            "share_features_extractor": False,
            "net_arch": {
                "pi": list(V2_POLICY_HIDDEN_SIZES),
                "vf": list(V2_POLICY_HIDDEN_SIZES),
            },
            "activation_fn": nn.Tanh,
        },
    }
    if model_path:
        return PPO.load(model_path, env=env, **kwargs)
    return PPO("MlpPolicy", env, seed=seed, verbose=1, **kwargs)


def save_ppo(model: PPO, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))


def collect_settings(scenarios: Sequence[str]) -> list[str]:
    settings: list[str] = []
    for scenario in scenarios:
        settings.extend(list_setting_ids(scenario, runnable_only=True))
    if not settings:
        raise ValueError("A server group has no runnable settings")
    return settings


__all__ = [
    "DEFAULT_PPO_HYPERPARAMETERS_V2",
    "PPO_GAE_LAMBDA",
    "PPO_GAMMA",
    "PPOHorizonDiagnosticsV2",
    "PPOHyperparameterSpecV2",
    "V2_CONTROL_DT_S",
    "V2_DEFAULT_PPO_EPOCHS",
    "V2_MAX_EPISODE_STEPS",
    "V2_PPO_CHECKPOINT_COMPATIBILITY",
    "V2_PPO_CHECKPOINT_NAMESPACE",
    "V2_PPO_GAE_LAMBDA",
    "V2_PPO_GAMMA",
    "V2_PPO_HYPERPARAMETER_CONTRACT",
    "V2_POLICY_ACTIVATION",
    "V2_POLICY_HIDDEN_SIZES",
    "build_or_load_ppo",
    "collect_settings",
    "configure_initial_config",
    "save_ppo",
]
