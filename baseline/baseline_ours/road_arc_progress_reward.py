"""Road-arc progress reward for baseline_ours.

Only the progress coordinate differs from the previous PPO reward: live ego
positions project onto a static CARLA-world road polyline instead of one global
O-to-D chord.  The +50 high-water budget and exact terminal rewards remain;
the former terminal speed bonus is intentionally removed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from baseline.baseline_ours.road_arc_reference import RoadArcReference


# This is a reward-design identifier, not a trainer/script version.  Run
# directories should add a timestamp separately.
REWARD_CONTRACT_VERSION_V2 = "road_arc_high_water_no_speed_bonus_r1"
Point2DV2 = Tuple[float, float]

GOAL_REACHED_V2 = "goal_reached"
COLLISION_JAYWALKER_V2 = "collision_jaywalker"
COLLISION_SUMO_VEHICLE_V2 = "collision_sumo_vehicle"
WORKZONE_VIOLATION_V2 = "workzone_violation"
OFF_ROAD_V2 = "off_road"
TIMEOUT_V2 = "timeout"
RUNNING_V2 = "running"


@dataclass(frozen=True)
class RewardConfigV2:
    """The four reward terms used by V2."""

    progress_budget: float = 50.0
    timeout_penalty: float = -50.0
    goal_reward: float = 100.0
    failure_penalty: float = -100.0

    def __post_init__(self) -> None:
        values = (
            self.progress_budget,
            self.timeout_penalty,
            self.goal_reward,
            self.failure_penalty,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Reward values must be finite")
        if not 0.0 <= self.progress_budget <= 100.0:
            raise ValueError("progress_budget must be in [0, 100]")
        if self.timeout_penalty >= 0.0:
            raise ValueError("timeout_penalty must be negative")
        if self.goal_reward <= 0.0:
            raise ValueError("goal_reward must be positive")
        if self.failure_penalty >= 0.0:
            raise ValueError("failure_penalty must be negative")

    @property
    def terminal_rewards(self) -> Mapping[str, float]:
        return {
            GOAL_REACHED_V2: self.goal_reward,
            COLLISION_JAYWALKER_V2: self.failure_penalty,
            COLLISION_SUMO_VEHICLE_V2: self.failure_penalty,
            WORKZONE_VIOLATION_V2: self.failure_penalty,
            OFF_ROAD_V2: self.failure_penalty,
            TIMEOUT_V2: self.timeout_penalty,
        }


@dataclass(frozen=True)
class ProgressUpdateV2:
    normalized_progress: float
    best_progress: float
    progress_delta: float
    reward: float
    cumulative_progress_reward: float


@dataclass(frozen=True)
class RewardStepV2:
    reward: float
    reason: str
    terminated: bool
    normalized_progress: float
    best_progress: float
    progress_delta: float
    cumulative_progress_reward: float
    average_speed_mps: float
    terminal_base_reward: float


class RoadArcProgressTrackerV2:
    """Track road-arc completion with an anti-oscillation high-water mark."""

    def __init__(
        self,
        *,
        origin: Point2DV2,
        finish_line: Tuple[Point2DV2, Point2DV2],
        reference: RoadArcReference,
        progress_budget: float,
    ) -> None:
        self.origin = _point_v2(origin, "origin")
        first = _point_v2(finish_line[0], "finish_line[0]")
        second = _point_v2(finish_line[1], "finish_line[1]")
        self.destination = (
            0.5 * (first[0] + second[0]),
            0.5 * (first[1] + second[1]),
        )
        self.reference = reference
        self._origin_s_m = reference.project(self.origin).distance_along_m
        self._destination_s_m = reference.project(self.destination).distance_along_m
        self._arc_span_m = self._destination_s_m - self._origin_s_m
        if self._arc_span_m <= 1e-6:
            raise ValueError("Road reference must run forward from origin to finish line")
        if not math.isfinite(float(progress_budget)) or not 0.0 <= progress_budget <= 100.0:
            raise ValueError("progress_budget must be finite and in [0, 100]")
        self.progress_budget = float(progress_budget)
        self.best_progress = 0.0
        self.cumulative_progress_reward = 0.0

    def _normalized_projection(self, position: Point2DV2) -> float:
        current = _point_v2(position, "position")
        current_s_m = self.reference.project(current).distance_along_m
        return float(np.clip(
            (current_s_m - self._origin_s_m) / self._arc_span_m,
            0.0,
            1.0,
        ))

    def advance(
        self,
        position: Point2DV2,
    ) -> ProgressUpdateV2:
        """Reward only new road-arc completion beyond the prior frontier."""
        normalized = self._normalized_projection(position)
        new_best = max(self.best_progress, normalized)
        delta = new_best - self.best_progress
        reward = self.progress_budget * delta
        self.best_progress = new_best
        self.cumulative_progress_reward = min(
            self.progress_budget,
            self.cumulative_progress_reward + reward,
        )
        return ProgressUpdateV2(
            normalized_progress=normalized,
            best_progress=self.best_progress,
            progress_delta=delta,
            reward=reward,
            cumulative_progress_reward=self.cumulative_progress_reward,
        )

    def observe_without_reward(
        self,
        position: Point2DV2,
        *,
        goal_reached: bool = False,
    ) -> float:
        """Update progress diagnostics without issuing another dense reward.

        Terminal reward replaces progress on the same tick, but the terminal
        position must still be reflected by ``best_progress`` for logging.
        ``cumulative_progress_reward`` intentionally remains the amount of
        dense reward that was actually paid before termination.
        """
        normalized = 1.0 if goal_reached else self._normalized_projection(position)
        self.best_progress = max(self.best_progress, normalized)
        return normalized


# Compatibility names keep the copied PPO interfaces unchanged.  The new
# contract fingerprint prevents old checkpoints from resuming under this reward.
ODProgressTrackerV2 = RoadArcProgressTrackerV2
ForwardDistanceProgressTrackerV2 = RoadArcProgressTrackerV2


class RewardCalculatorV2:
    """Apply dense progress or one exact terminal reward per environment step."""

    def __init__(
        self,
        *,
        origin: Point2DV2,
        finish_line: Tuple[Point2DV2, Point2DV2],
        reference: RoadArcReference,
        config: RewardConfigV2 = RewardConfigV2(),
    ) -> None:
        self.config = config
        self.progress = RoadArcProgressTrackerV2(
            origin=origin,
            finish_line=finish_line,
            reference=reference,
            progress_budget=config.progress_budget,
        )
        self._episode_ended = False
        self._speed_sum_mps = 0.0
        self._speed_samples = 0

    def step(
        self,
        position: Point2DV2,
        *,
        speed_mps: float,
        terminal_reason: Optional[str] = None,
    ) -> RewardStepV2:
        if self._episode_ended:
            raise RuntimeError("Reward requested after the episode ended")
        speed = float(speed_mps)
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError("speed_mps must be finite and non-negative")
        self._speed_sum_mps += speed
        self._speed_samples += 1
        average_speed = self._speed_sum_mps / self._speed_samples

        if terminal_reason is not None and terminal_reason != RUNNING_V2:
            normalized = self.progress.observe_without_reward(
                position, goal_reached=terminal_reason == GOAL_REACHED_V2
            )
            try:
                terminal_reward = self.config.terminal_rewards[terminal_reason]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported terminal reason: {terminal_reason!r}"
                ) from error
            self._episode_ended = True
            # The terminal step overrides progress, so a collision cannot also
            # receive a positive dense reward on the same tick.
            return RewardStepV2(
                reward=float(terminal_reward),
                reason=terminal_reason,
                terminated=True,
                normalized_progress=normalized,
                best_progress=self.progress.best_progress,
                progress_delta=0.0,
                cumulative_progress_reward=self.progress.cumulative_progress_reward,
                average_speed_mps=float(average_speed),
                terminal_base_reward=float(terminal_reward),
            )

        update = self.progress.advance(position)
        return RewardStepV2(
            reward=update.reward,
            reason=RUNNING_V2,
            terminated=False,
            normalized_progress=update.normalized_progress,
            best_progress=update.best_progress,
            progress_delta=update.progress_delta,
            cumulative_progress_reward=update.cumulative_progress_reward,
            average_speed_mps=float(average_speed),
            terminal_base_reward=0.0,
        )


def _point_v2(value: Point2DV2, name: str) -> Point2DV2:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly x and y")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "COLLISION_JAYWALKER_V2",
    "COLLISION_SUMO_VEHICLE_V2",
    "GOAL_REACHED_V2",
    "ForwardDistanceProgressTrackerV2",
    "ODProgressTrackerV2",
    "RoadArcProgressTrackerV2",
    "OFF_ROAD_V2",
    "ProgressUpdateV2",
    "REWARD_CONTRACT_VERSION_V2",
    "RUNNING_V2",
    "RewardCalculatorV2",
    "RewardConfigV2",
    "RewardStepV2",
    "TIMEOUT_V2",
    "WORKZONE_VIOLATION_V2",
]
