"""Independent PPO builder for the plain Transformer baseline."""
from __future__ import annotations

from typing import Any, Optional

from stable_baselines3 import PPO
from torch import nn

from baseline.PPO.rollout_plan import RolloutPlan, choose_batch_size
from baseline.PPO.runtime_v2 import PPO_GAE_LAMBDA, PPO_GAMMA
from baseline.PPO.training_config_v2 import DEFAULT_THREE_SERVER_TRAINING_V2
from baseline.Transformer_PPO.encoder import (
    PlainTransformerEncoder,
    TRANSFORMER_EXPECTED_POLICY_PARAMETERS,
)


TRANSFORMER_POLICY_HIDDEN_SIZES = (256, 256)
TRANSFORMER_POLICY_ACTIVATION = "tanh"
TRANSFORMER_POLICY_TRAINABLE_PARAMETERS = TRANSFORMER_EXPECTED_POLICY_PARAMETERS
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
    """Keep the existing PPO contract and replace only its feature extractor."""
    resolved_batch_size = (
        choose_batch_size(plan.buffer_size_steps)
        if batch_size is None
        else int(batch_size)
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive")
    if int(ppo_epochs) < 1:
        raise ValueError("ppo_epochs must be positive")
    if target_kl is not None:
        raise ValueError("The plain Transformer PPO baseline disables KL stopping")

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
            "features_extractor_class": PlainTransformerEncoder,
            "share_features_extractor": False,
            "net_arch": {
                "pi": list(TRANSFORMER_POLICY_HIDDEN_SIZES),
                "vf": list(TRANSFORMER_POLICY_HIDDEN_SIZES),
            },
            "activation_fn": nn.Tanh,
        },
    }
    if model_path:
        return PPO.load(model_path, env=env, **kwargs)
    return PPO("MlpPolicy", env, seed=int(seed), verbose=1, **kwargs)


__all__ = [
    "TRANSFORMER_POLICY_ACTIVATION",
    "TRANSFORMER_POLICY_HIDDEN_SIZES",
    "TRANSFORMER_POLICY_TRAINABLE_PARAMETERS",
    "build_or_load_ppo",
]
