"""SB3-compatible rollout collection that excludes infrastructure faults.

The all-scenario runner uses a fixed number of *valid* on-policy samples per
Town.  A recoverable CARLA/SUMO fault ends and auto-resets the current Gym
episode, but it is not an MDP transition and therefore must not consume a
RolloutBuffer row.  This collector mirrors Stable-Baselines3's on-policy
collector for the runner's single ``DummyVecEnv`` while retrying after those
faults.
"""
from __future__ import annotations

from typing import Collection

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from config.scenario_selector import CoverageSelector


def _is_infrastructure_fault(
    infos: list[dict], infrastructure_reasons: Collection[str]
) -> bool:
    """Return whether the sole VecEnv row represents a recoverable fault."""
    return bool(infos) and infos[0].get("reason") in infrastructure_reasons


def collect_valid_rollouts(
    model,
    env: VecEnv,
    callback: BaseCallback,
    rollout_buffer: RolloutBuffer,
    n_rollout_steps: int,
    *,
    infrastructure_reasons: Collection[str] = CoverageSelector.INFRASTRUCTURE_REASONS,
) -> bool:
    """Collect exactly ``n_rollout_steps`` valid samples for one environment.

    Infrastructure-fault rows are discarded after advancing ``model._last_obs``
    to the observation returned by the VecEnv's automatic reset.  They do not
    increment ``model.num_timesteps``, do not invoke ``callback.on_step()``, and
    do not enter the rollout buffer.  Policy outcomes, including the project's
    absorbing ``reason=timeout`` transition, retain normal SB3 behavior.

    ``rollout_buffer.physical_env_steps`` and
    ``rollout_buffer.infrastructure_faults`` are diagnostic counters only; PPO
    training and checkpoint serialization continue to use the valid rows.
    """
    if env.num_envs != 1:
        raise ValueError("collect_valid_rollouts supports exactly one VecEnv")
    if rollout_buffer.n_envs != 1:
        raise ValueError("collect_valid_rollouts requires a one-env RolloutBuffer")
    if n_rollout_steps < 1:
        raise ValueError("n_rollout_steps must be positive")
    assert model._last_obs is not None, "No previous observation was provided"

    model.policy.set_training_mode(False)
    valid_steps = 0
    physical_steps = 0
    infrastructure_faults = 0
    rollout_buffer.reset()
    if model.use_sde:
        model.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()

    while valid_steps < n_rollout_steps:
        if (
            model.use_sde
            and model.sde_sample_freq > 0
            and physical_steps % model.sde_sample_freq == 0
        ):
            model.policy.reset_noise(env.num_envs)

        with th.no_grad():
            obs_tensor = obs_as_tensor(model._last_obs, model.device)
            actions, values, log_probs = model.policy(obs_tensor)
        actions = actions.cpu().numpy()

        clipped_actions = actions
        if isinstance(model.action_space, spaces.Box):
            if model.policy.squash_output:
                clipped_actions = model.policy.unscale_action(clipped_actions)
            else:
                clipped_actions = np.clip(
                    actions, model.action_space.low, model.action_space.high
                )

        new_obs, rewards, dones, infos = env.step(clipped_actions)
        physical_steps += env.num_envs

        if _is_infrastructure_fault(infos, infrastructure_reasons):
            infrastructure_faults += 1
            record_fault = getattr(callback, "record_infrastructure_fault", None)
            if callable(record_fault):
                record_fault(infos[0])

            # DummyVecEnv has already reset the failed episode.  The returned
            # new_obs is therefore the first observation of the retried ticket,
            # and dones=True correctly marks its next stored action as the start
            # of a fresh episode.
            model._last_obs = new_obs
            model._last_episode_starts = dones
            continue

        # Only real MDP samples advance SB3's training clock and callback clock.
        model.num_timesteps += env.num_envs
        callback.update_locals(locals())
        if not callback.on_step():
            rollout_buffer.physical_env_steps = physical_steps
            rollout_buffer.infrastructure_faults = infrastructure_faults
            return False

        model._update_info_buffer(infos, dones)
        valid_steps += 1

        if isinstance(model.action_space, spaces.Discrete):
            actions = actions.reshape(-1, 1)

        # Preserve SB3's conventional non-absorbing TimeLimit bootstrap.  The
        # task timeout is returned as terminated=True,truncated=True, so
        # DummyVecEnv intentionally exposes TimeLimit.truncated=False and the
        # terminal -20 reward is not modified here.
        for index, done in enumerate(dones):
            if (
                done
                and infos[index].get("terminal_observation") is not None
                and infos[index].get("TimeLimit.truncated", False)
            ):
                terminal_obs = model.policy.obs_to_tensor(
                    infos[index]["terminal_observation"]
                )[0]
                with th.no_grad():
                    terminal_value = model.policy.predict_values(terminal_obs)[0]
                rewards[index] += model.gamma * terminal_value

        rollout_buffer.add(
            model._last_obs,
            actions,
            rewards,
            model._last_episode_starts,
            values,
            log_probs,
        )
        model._last_obs = new_obs
        model._last_episode_starts = dones

    with th.no_grad():
        last_values = model.policy.predict_values(
            obs_as_tensor(model._last_obs, model.device)
        )
    rollout_buffer.compute_returns_and_advantage(
        last_values=last_values, dones=model._last_episode_starts
    )
    rollout_buffer.physical_env_steps = physical_steps
    rollout_buffer.infrastructure_faults = infrastructure_faults

    callback.update_locals(locals())
    callback.on_rollout_end()
    return True


__all__ = ["collect_valid_rollouts"]
