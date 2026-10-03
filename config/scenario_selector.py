"""
Scenario setting selectors
==========================
Decides which work-zone scenario should be used to build the *next* RL
episode. It lives in the environment layer so the RL algorithm (PPO, SAC,
TD3, DDPG, ...) never has to know which scenario is active: the algorithm
only calls ``env.reset()`` / ``env.step()``.

Adding a new selector (e.g. a curriculum that advances after a fixed number
of episodes or once a reward threshold is met) means writing another class
that implements ``next()``; no environment or training-code changes required.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter, deque
from typing import List, Optional, Sequence


class ScenarioSelector:
    """Abstract scenario selector.

    ``next()`` returns the canonical setting for the upcoming episode
    (for example ``"s1/wz1/a"``). It is called once per ``env.reset()``.
    """

    def next(self) -> str:
        raise NotImplementedError


class RoundRobinSelector(ScenarioSelector):
    """Cycle through a fixed list of scenarios, one per episode.

    With ``scenarios=["wz1", "wz2", "wz3"]`` the sequence produced is
    ``wz1, wz2, wz3, wz1, wz2, ...`` — the standard on-policy mixture that
    feeds every work-zone's transitions into the same rollout buffer.
    """

    def __init__(self, scenarios: Sequence[str]) -> None:
        if not scenarios:
            raise ValueError("RoundRobinSelector requires at least one scenario")
        self._scenarios: List[str] = list(scenarios)
        self._cycle = itertools.cycle(self._scenarios)

    def next(self) -> str:
        return next(self._cycle)

    @property
    def scenarios(self) -> List[str]:
        """The fixed list of scenarios this selector cycles through."""
        return list(self._scenarios)


class StaticSelector(ScenarioSelector):
    """Always return a single fixed scenario (useful for a plain single-scenario run)."""

    def __init__(self, scenario: str) -> None:
        self._scenario = scenario

    def next(self) -> str:
        return self._scenario


class CoverageSelector(ScenarioSelector):
    """Deterministic setting schedule with exactly N valid episode tickets.

    Infrastructure-aborted episodes are appended to the queue and retried;
    policy outcomes such as success, collision, off-road and timeout are valid
    completed episodes and count toward coverage.
    """

    INFRASTRUCTURE_REASONS = {
        "carla_tick_or_rpc_failure",
        "sumo_connection_lost",
        "sumo_mirror_spawn_collision",
        "worker_runtime_fault",
    }

    def __init__(self, settings: Sequence[str], repeats: int) -> None:
        if not settings:
            raise ValueError("CoverageSelector requires at least one setting")
        if repeats < 1:
            raise ValueError("repeats must be positive")
        self.settings = list(settings)
        self.repeats = int(repeats)
        self._queue = deque(
            setting for _coverage_round in range(self.repeats) for setting in self.settings
        )
        self._completed: Counter[str] = Counter()
        self._current: str | None = None
        self._overflow_index = 0

    def next(self) -> str:
        if self._queue:
            self._current = self._queue.popleft()
        else:
            # Coverage can finish before an auto-sized rollout buffer is full.
            # Continue cycling all settings so the remaining buffer is diverse
            # instead of oversampling only the final layout.
            self._current = self.settings[self._overflow_index % len(self.settings)]
            self._overflow_index += 1
        return self._current

    def on_episode_end(self, info: dict) -> None:
        if self._current is None:
            return
        reason = info.get("reason")
        if reason in self.INFRASTRUCTURE_REASONS:
            self._queue.append(self._current)
        elif self._completed[self._current] < self.repeats:
            self._completed[self._current] += 1
        self._current = None

    @property
    def complete(self) -> bool:
        return all(self._completed[s] >= self.repeats for s in self.settings)

    @property
    def completed_counts(self) -> dict[str, int]:
        return {s: self._completed[s] for s in self.settings}


class RandomLayoutSelector(ScenarioSelector):
    """Randomize layouts while keeping the requested Scenario/WZ fixed.

    One shuffled bag contains every supplied layout exactly once. A fresh bag
    is shuffled after it is exhausted, so episode order is random without
    silently starving one A/B/C layout. ``repeats`` means exactly N valid
    episodes per layout; infrastructure-aborted tickets are retried.
    """

    INFRASTRUCTURE_REASONS = CoverageSelector.INFRASTRUCTURE_REASONS

    def __init__(
        self,
        settings: Sequence[str],
        repeats: int,
        *,
        seed: int | None = None,
    ) -> None:
        if not settings:
            raise ValueError("RandomLayoutSelector requires at least one setting")
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if len(set(settings)) != len(settings):
            raise ValueError("RandomLayoutSelector settings must be unique")

        parsed = [setting.split("/") for setting in settings]
        if any(len(parts) != 3 for parts in parsed):
            raise ValueError("settings must use canonical scenario/wz/layout ids")
        fixed_pairs = {(parts[0], parts[1]) for parts in parsed}
        if len(fixed_pairs) != 1:
            raise ValueError(
                "RandomLayoutSelector may randomize layouts only; scenario and WZ must match"
            )

        self.settings = list(settings)
        self.repeats = int(repeats)
        self.scenario_id, self.wz_id = next(iter(fixed_pairs))
        self._rng = random.Random(seed)
        self._queue: deque[str] = deque()
        self._completed: Counter[str] = Counter()
        self._current: str | None = None

    def _refill(self) -> None:
        remaining = [
            setting
            for setting in self.settings
            if self._completed[setting] < self.repeats
        ]
        # PPO may need transitions after coverage finishes because it collects
        # complete on-policy rollout buffers. Keep overflow episodes balanced.
        candidates = remaining or list(self.settings)
        self._rng.shuffle(candidates)
        self._queue.extend(candidates)

    def next(self) -> str:
        if not self._queue:
            self._refill()
        self._current = self._queue.popleft()
        return self._current

    def on_episode_end(self, info: dict) -> None:
        if self._current is None:
            return
        reason = info.get("reason")
        if reason in self.INFRASTRUCTURE_REASONS:
            self._queue.append(self._current)
        elif self._completed[self._current] < self.repeats:
            self._completed[self._current] += 1
        self._current = None

    @property
    def complete(self) -> bool:
        return all(self._completed[s] >= self.repeats for s in self.settings)

    @property
    def completed_counts(self) -> dict[str, int]:
        return {s: self._completed[s] for s in self.settings}


class RandomWorkZoneSelector(ScenarioSelector):
    """Randomize WZ and layout while keeping one scenario fixed.

    Work-zones use a shuffled bag and cannot repeat at a bag boundary when at
    least two WZs remain. Each WZ owns an independent shuffled layout bag.
    This produces random episode order while still completing exactly N valid
    tickets for every Scenario/WZ/layout setting.
    """

    INFRASTRUCTURE_REASONS = CoverageSelector.INFRASTRUCTURE_REASONS

    def __init__(
        self,
        settings: Sequence[str],
        repeats: int,
        *,
        seed: int | None = None,
    ) -> None:
        if not settings:
            raise ValueError("RandomWorkZoneSelector requires at least one setting")
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if len(set(settings)) != len(settings):
            raise ValueError("RandomWorkZoneSelector settings must be unique")

        parsed = [setting.split("/") for setting in settings]
        if any(len(parts) != 3 for parts in parsed):
            raise ValueError("settings must use canonical scenario/wz/layout ids")
        scenario_ids = {parts[0] for parts in parsed}
        if len(scenario_ids) != 1:
            raise ValueError("RandomWorkZoneSelector must stay inside one scenario")

        self.settings = list(settings)
        self.repeats = int(repeats)
        self.scenario_id = next(iter(scenario_ids))
        self._settings_by_wz: dict[str, list[str]] = {}
        for setting, parts in zip(self.settings, parsed):
            self._settings_by_wz.setdefault(parts[1], []).append(setting)
        self._wz_ids = list(self._settings_by_wz)
        self._rng = random.Random(seed)
        self._wz_queue: deque[str] = deque()
        self._layout_queues: dict[str, deque[str]] = {
            wz_id: deque() for wz_id in self._wz_ids
        }
        self._completed: Counter[str] = Counter()
        self._current: str | None = None
        self._last_wz: str | None = None

    def _eligible_wz_ids(self) -> list[str]:
        remaining = [
            wz_id
            for wz_id, settings in self._settings_by_wz.items()
            if any(self._completed[setting] < self.repeats for setting in settings)
        ]
        return remaining or list(self._wz_ids)

    def _refill_wz_queue(self) -> None:
        candidates = self._eligible_wz_ids()
        self._rng.shuffle(candidates)
        if (
            len(candidates) > 1
            and self._last_wz is not None
            and candidates[0] == self._last_wz
        ):
            candidates[0], candidates[1] = candidates[1], candidates[0]
        self._wz_queue.extend(candidates)

    def _next_layout(self, wz_id: str) -> str:
        queue = self._layout_queues[wz_id]
        if not queue:
            remaining = [
                setting
                for setting in self._settings_by_wz[wz_id]
                if self._completed[setting] < self.repeats
            ]
            candidates = remaining or list(self._settings_by_wz[wz_id])
            self._rng.shuffle(candidates)
            queue.extend(candidates)
        return queue.popleft()

    def next(self) -> str:
        while True:
            if not self._wz_queue:
                self._refill_wz_queue()
            wz_id = self._wz_queue.popleft()
            eligible = set(self._eligible_wz_ids())
            if self.complete or wz_id in eligible:
                break
        self._current = self._next_layout(wz_id)
        self._last_wz = wz_id
        return self._current

    def on_episode_end(self, info: dict) -> None:
        if self._current is None:
            return
        reason = info.get("reason")
        if reason in self.INFRASTRUCTURE_REASONS:
            wz_id = self._current.split("/")[1]
            self._layout_queues[wz_id].append(self._current)
        elif self._completed[self._current] < self.repeats:
            self._completed[self._current] += 1
        self._current = None

    @property
    def complete(self) -> bool:
        return all(self._completed[s] >= self.repeats for s in self.settings)

    @property
    def completed_counts(self) -> dict[str, int]:
        return {s: self._completed[s] for s in self.settings}
