"""Shared Stable-Baselines3 callbacks for PPO validation entry points."""
from __future__ import annotations

from collections import Counter
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from config.scenario_selector import CoverageSelector


def format_live_update(info: dict[str, Any], reward: float, done: bool) -> str:
    """Build the stable one-line terminal status used by single-server demos."""
    collision_detail = ""
    if str(info.get("reason", "")).startswith("collision_"):
        collision_detail = (
            f" actor_id={info.get('collision_actor_id', 'n/a')}"
            f" actor_type={info.get('collision_actor_type', 'n/a')}"
            f" actor_role={info.get('collision_actor_role', 'n/a')}"
            f" sumo_id={info.get('collision_sumo_id', 'n/a')}"
            f" distance_m={info.get('collision_distance_m', 'n/a')}"
        )
    return (
        "LIVE_PPO "
        f"setting={info.get('setting_id', 'n/a')} "
        f"step={int(info.get('episode_step', 0))} "
        f"ego=({info.get('ego_x', 'n/a')},{info.get('ego_y', 'n/a')}) "
        f"speed={info.get('ego_speed_mps', 'n/a')} "
        f"sumo_cars={info.get('sumo_mirrors_in_carla', 0)} "
        f"walkers_active={info.get('jaywalker_active_count', 0)} "
        f"walkers_triggered={info.get('jaywalker_triggered_count', 0)} "
        f"walkers_spawned={info.get('jaywalker_spawned_count', 0)} "
        f"walkers_completed={info.get('jaywalker_completed_count', 0)} "
        f"reward={float(reward):.3f} done={bool(done)} "
        f"reason={info.get('reason', '-')}{collision_detail}"
    )


class CoverageStopCallback(BaseCallback):
    """Track outcomes and stop only after PPO optimizes a complete rollout."""

    def __init__(self, selector: Any, every_steps: int = 0) -> None:
        super().__init__()
        self.selector = selector
        self.every_steps = max(0, int(every_steps))
        self.outcomes: Counter[tuple[str, str]] = Counter()
        self._stop_before_next_rollout = False

    def _on_step(self) -> bool:
        if self._stop_before_next_rollout:
            return False
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for index, (done, info) in enumerate(zip(dones, infos)):
            if done:
                key = (
                    info.get("setting_id", "unknown"),
                    info.get("reason", "unknown"),
                )
                self.outcomes[key] += 1
            episode_step = int(info.get("episode_step", 0))
            if self.every_steps and (
                done or episode_step == 1 or episode_step % self.every_steps == 0
            ):
                reward = float(rewards[index]) if index < len(rewards) else 0.0
                print(format_live_update(info, reward, bool(done)), flush=True)
        return True

    def _on_rollout_end(self) -> None:
        # A False return from _on_step stops learning. Delay it until the next
        # rollout so SB3 still optimizes the complete buffer that met coverage.
        self._stop_before_next_rollout = self.selector.complete


class GroupCoverageCallback(BaseCallback):
    """Parent-process coverage tracker for three subprocess environments."""

    def __init__(self, settings: list[str], repeats: int) -> None:
        super().__init__()
        self.settings = settings
        self.repeats = repeats
        self.completed: Counter[str] = Counter()
        self.outcomes: Counter[tuple[str, str]] = Counter()
        self._stop_before_next_rollout = False

    def _on_step(self) -> bool:
        if self._stop_before_next_rollout:
            return False
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not done:
                continue
            setting = info.get("setting_id", "unknown")
            reason = info.get("reason", "unknown")
            self.outcomes[(setting, reason)] += 1
            if (
                setting in self.settings
                and reason not in CoverageSelector.INFRASTRUCTURE_REASONS
            ):
                self.completed[setting] = min(
                    self.repeats, self.completed[setting] + 1
                )
        return True

    def _on_rollout_end(self) -> None:
        self._stop_before_next_rollout = all(
            self.completed[setting] >= self.repeats for setting in self.settings
        )


__all__ = [
    "CoverageStopCallback",
    "GroupCoverageCallback",
    "format_live_update",
]
