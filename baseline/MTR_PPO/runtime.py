"""PPO construction for the input-matched full-size MTR baseline."""
from __future__ import annotations

from typing import Any, Optional

from stable_baselines3 import PPO
from torch import nn

from baseline.MTR_PPO.encoder import (
    InputMatchedMTREncoder,
    MTR_EXPECTED_POLICY_PARAMETERS,
)
from baseline.PPO.rollout_plan import RolloutPlan, choose_batch_size
from baseline.PPO.runtime_v2 import PPO_GAE_LAMBDA, PPO_GAMMA
from baseline.PPO.training_config_v2 import DEFAULT_THREE_SERVER_TRAINING_V2


MTR_POLICY_HIDDEN_SIZES = (256, 256)
MTR_POLICY_ACTIVATION = "tanh"
MTR_POLICY_TRAINABLE_PARAMETERS = MTR_EXPECTED_POLICY_PARAMETERS
_PPO_DEFAULTS = DEFAULT_THREE_SERVER_TRAINING_V2.ppo


def build_or_load_ppo(
    env: Any,
    plan: RolloutPlan,
    *,
    model_path: Optional[str],
    ppo_epochs: int,
    seed: int,
    device: str = "auto",
    batch_size: Optional[int] = None,
    target_kl: Optional[float] = None,
) -> PPO:
    """Build/load PPO while changing only the observation feature extractor."""
    resolved_batch_size = (
        choose_batch_size(plan.buffer_size_steps)
        if batch_size is None
        else int(batch_size)
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive")
    if target_kl is not None:
        raise ValueError("The formal MTR PPO baseline disables KL early stopping")

    kwargs = {
        "n_steps": plan.n_steps_per_env,
        "batch_size": resolved_batch_size,
        "n_epochs": int(ppo_epochs),
        "gamma": PPO_GAMMA,
        "gae_lambda": PPO_GAE_LAMBDA,
        "learning_rate": _PPO_DEFAULTS.learning_rate,
        "clip_range": _PPO_DEFAULTS.clip_range,
        "clip_range_vf": _PPO_DEFAULTS.clip_range_vf,
        "normalize_advantage": _PPO_DEFAULTS.normalize_advantage,
        "ent_coef": _PPO_DEFAULTS.entropy_coefficient,
        "vf_coef": _PPO_DEFAULTS.value_function_coefficient,
        "max_grad_norm": _PPO_DEFAULTS.max_gradient_norm,
        "device": device,
        "target_kl": None,
        "policy_kwargs": {
            "features_extractor_class": InputMatchedMTREncoder,
            "share_features_extractor": False,
            "net_arch": {
                "pi": list(MTR_POLICY_HIDDEN_SIZES),
                "vf": list(MTR_POLICY_HIDDEN_SIZES),
            },
            "activation_fn": nn.Tanh,
        },
    }
    if model_path:
        return PPO.load(model_path, env=env, **kwargs)
    return PPO("MlpPolicy", env, seed=int(seed), verbose=1, **kwargs)


__all__ = [
    "MTR_POLICY_ACTIVATION",
    "MTR_POLICY_HIDDEN_SIZES",
    "MTR_POLICY_TRAINABLE_PARAMETERS",
    "build_or_load_ppo",
]
