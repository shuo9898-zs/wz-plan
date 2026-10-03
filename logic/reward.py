"""Small, explicit PPO reward contract shared by all scenarios."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

REWARD_CONTRACT_VERSION = "bounded_high_water_v2"


FAILURE_REASONS = {
    "offroad",
    "workzone_intrusion",
    "corridor_departure",
    "collision",
    "collision_walker",
    "collision_vehicle",
    "collision_bg_vehicle",
    "collision_sumo_vehicle",
    "collision_sumo_vehicle_proximity",
    "collision_carla_vehicle",
    "collision_static",
    "collision_other",
    "collision_unknown",
}


@dataclass(frozen=True)
class RewardWeights:
    failure: float = -100.0
    success: float = 100.0
    timeout: float = -20.0
    progress_budget: float = 20.0
    step_cost: float = -0.01


@dataclass
class BoundedProgressReward:
    """Pay a fixed episode budget only for newly reached forward progress.

    ``metric`` must increase in the legal travel direction.  Progress is
    normalized between the episode's authored start and target metrics, then
    clipped to [0, 1].  Remembering the high-water mark means reversing and
    re-driving the same section cannot harvest the dense reward twice.
    """

    start_metric: float
    target_metric: float
    budget: float = 20.0
    _high_water_fraction: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        values = (self.start_metric, self.target_metric, self.budget)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Progress tracker values must be finite")
        if self.target_metric <= self.start_metric:
            raise ValueError("Progress target metric must be greater than its start")
        if self.budget < 0.0:
            raise ValueError("Progress reward budget must be non-negative")

    @property
    def high_water_fraction(self) -> float:
        return self._high_water_fraction

    def advance(self, metric: float) -> float:
        """Return reward for the newly reached fraction of the route."""
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError("Progress metric must be finite")
        span = self.target_metric - self.start_metric
        fraction = (metric - self.start_metric) / span
        fraction = min(1.0, max(0.0, fraction))
        newly_reached = max(0.0, fraction - self._high_water_fraction)
        self._high_water_fraction = max(self._high_water_fraction, fraction)
        return newly_reached * self.budget


def compute_reward(
    *,
    terminated: bool,
    truncated: bool,
    success: bool,
    reason: str | None,
    weights: RewardWeights = RewardWeights(),
    progress_reward: float = 0.0,
) -> float:
    """Combine mutually exclusive terminal events with bounded dense reward."""
    if success:
        return weights.success
    # Episode-cap timeouts are absorbing task failures.  The environment sets
    # both Gym flags so SB3 will not bootstrap V(terminal_observation); reason
    # therefore takes precedence over the generic ``terminated`` branch.
    if reason == "timeout" or truncated:
        return weights.timeout
    if terminated:
        return weights.failure
    progress_reward = float(progress_reward)
    if not math.isfinite(progress_reward) or progress_reward < 0.0:
        raise ValueError("Per-step progress reward must be finite and non-negative")
    return progress_reward + weights.step_cost
