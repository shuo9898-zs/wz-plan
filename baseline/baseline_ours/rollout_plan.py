"""Pure rollout-buffer sizing from per-setting episode step limits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RolloutPlan:
    setting_count: int
    repeats: int
    n_envs: int
    group_step_budgets: tuple[int, ...]
    n_steps_per_env: int
    buffer_size_steps: int

    @property
    def simulated_seconds_capacity(self) -> float:
        return self.buffer_size_steps * 0.1


def plan_single(
    episode_step_limits: Sequence[int],
    repeats: int,
    n_steps_override: int | None = None,
) -> RolloutPlan:
    if repeats < 1 or not episode_step_limits:
        raise ValueError("A rollout plan needs settings and positive repeats")
    budget = sum(int(limit) for limit in episode_step_limits) * repeats
    n_steps = int(n_steps_override) if n_steps_override is not None else budget
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    return RolloutPlan(
        setting_count=len(episode_step_limits),
        repeats=repeats,
        n_envs=1,
        group_step_budgets=(budget,),
        n_steps_per_env=n_steps,
        buffer_size_steps=n_steps,
    )


def plan_parallel(
    group_episode_step_limits: Sequence[Sequence[int]],
    repeats: int,
    n_steps_override: int | None = None,
) -> RolloutPlan:
    if repeats < 1 or not group_episode_step_limits:
        raise ValueError("A rollout plan needs groups and positive repeats")
    budgets = tuple(
        sum(int(limit) for limit in limits) * repeats
        for limits in group_episode_step_limits
    )
    if any(budget < 1 for budget in budgets):
        raise ValueError("Every worker group needs at least one setting")
    # SB3 uses the same n_steps for every vector environment. Size it for the
    # busiest town so that worker can complete its full N-ticket sweep.
    n_steps = int(n_steps_override) if n_steps_override is not None else max(budgets)
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    n_envs = len(budgets)
    return RolloutPlan(
        setting_count=sum(len(limits) for limits in group_episode_step_limits),
        repeats=repeats,
        n_envs=n_envs,
        group_step_budgets=budgets,
        n_steps_per_env=n_steps,
        buffer_size_steps=n_steps * n_envs,
    )


def describe_plan(plan: RolloutPlan) -> str:
    minutes = plan.simulated_seconds_capacity / 60.0
    return (
        f"settings={plan.setting_count} repeats={plan.repeats} n_envs={plan.n_envs} "
        f"group_budgets={list(plan.group_step_budgets)} "
        f"n_steps_per_env={plan.n_steps_per_env} "
        f"buffer_size_steps={plan.buffer_size_steps} "
        f"simulated_capacity={minutes:.1f}min"
    )


def choose_batch_size(buffer_size_steps: int, preferred: int = 64) -> int:
    """Largest batch no bigger than preferred that exactly divides a buffer."""
    for candidate in range(min(preferred, buffer_size_steps), 1, -1):
        if buffer_size_steps % candidate == 0:
            return candidate
    return 1
