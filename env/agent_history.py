"""Stable, perception-style dynamic-agent slots for PPO observations.

The policy never receives CARLA or SUMO actor identifiers.  Identifiers are
used only inside :class:`TrackedAgentEncoder` to keep a detected object in the
same slot from one 10-Hz frame to the next, and are exposed read-only for
diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


AgentKind = Literal["cone", "vehicle", "walker"]


@dataclass(frozen=True)
class AgentPoint:
    """One world-frame perception detection before ego-frame encoding."""

    actor_id: str
    kind: AgentKind
    x: float
    y: float
    vx: float
    vy: float


class TrackedAgentEncoder:
    """Encode at most ``max_agents`` nearby actors in stable current-frame slots.

    Each slot contains seven normalized values::

        presence, cone, vehicle, walker, rel_x, rel_y, rel_vx, rel_vy

    Position and velocity are expressed in the ego frame.  Actors outside the
    perception radius are not observations.  A vanished ID retains its empty
    slot for a short grace period so a one-frame mirror dropout does not
    reshuffle every other object.
    """

    AGENT_DIM = 8

    def __init__(
        self,
        *,
        max_agents: int = 8,
        radius_m: float = 50.0,
        relative_speed_scale_mps: float = 30.0,
        missing_ttl_steps: int = 2,
    ) -> None:
        if max_agents < 1:
            raise ValueError("max_agents must be positive")
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("radius_m must be finite and positive")
        if (
            not math.isfinite(relative_speed_scale_mps)
            or relative_speed_scale_mps <= 0.0
        ):
            raise ValueError(
                "relative_speed_scale_mps must be finite and positive"
            )
        if missing_ttl_steps < 0:
            raise ValueError("missing_ttl_steps must be non-negative")

        self.max_agents = int(max_agents)
        self.radius_m = float(radius_m)
        self.relative_speed_scale_mps = float(relative_speed_scale_mps)
        self.missing_ttl_steps = int(missing_ttl_steps)
        self._slots: list[str | None] = [None] * self.max_agents
        self._missed_steps: dict[str, int] = {}

    @property
    def observation_dim(self) -> int:
        return self.max_agents * self.AGENT_DIM

    @property
    def actor_ids(self) -> tuple[str | None, ...]:
        """Slot IDs for logging only; these values never enter the ndarray."""
        return tuple(self._slots)

    def reset(self) -> None:
        self._slots = [None] * self.max_agents
        self._missed_steps.clear()

    def encode(
        self,
        *,
        ego_x: float,
        ego_y: float,
        ego_yaw_deg: float,
        ego_vx: float,
        ego_vy: float,
        agents: Iterable[AgentPoint],
    ) -> np.ndarray:
        yaw = math.radians(float(ego_yaw_deg))
        c, s = math.cos(yaw), math.sin(yaw)

        def rotate_to_ego(world_x: float, world_y: float) -> tuple[float, float]:
            return c * world_x + s * world_y, -s * world_x + c * world_y

        visible: dict[str, tuple[AgentPoint, float, float, float]] = {}
        radius_sq = self.radius_m * self.radius_m
        for agent in agents:
            if agent.kind not in ("cone", "vehicle", "walker"):
                raise ValueError(f"Unsupported dynamic-agent kind: {agent.kind!r}")
            numeric = (
                agent.x,
                agent.y,
                agent.vx,
                agent.vy,
                ego_x,
                ego_y,
                ego_yaw_deg,
                ego_vx,
                ego_vy,
            )
            # A malformed detector/mirror must not truncate the whole episode.
            if not all(math.isfinite(float(value)) for value in numeric):
                continue
            world_dx = float(agent.x) - float(ego_x)
            world_dy = float(agent.y) - float(ego_y)
            distance_sq = world_dx * world_dx + world_dy * world_dy
            if distance_sq > radius_sq:
                continue
            rel_x, rel_y = rotate_to_ego(world_dx, world_dy)
            visible[str(agent.actor_id)] = (
                agent,
                rel_x,
                rel_y,
                distance_sq,
            )

        ranked_visible_ids = [
            actor_id
            for actor_id, _ in sorted(
                visible.items(),
                key=lambda item: (item[1][3], item[0]),
            )[: self.max_agents]
        ]
        selected_ids = set(ranked_visible_ids)

        # Preserve slots only for actors in the current nearest top-K.  A
        # farther incumbent is evicted immediately for a closer hazard.
        # Briefly missing IDs may reserve spare capacity, never capacity needed
        # by a currently visible top-K actor.
        for slot, actor_id in enumerate(self._slots):
            if actor_id is None:
                continue
            if actor_id in selected_ids:
                self._missed_steps.pop(actor_id, None)
                continue
            if actor_id in visible:
                self._slots[slot] = None
                self._missed_steps.pop(actor_id, None)
                continue
            missed = self._missed_steps.get(actor_id, 0) + 1
            if missed > self.missing_ttl_steps:
                self._slots[slot] = None
                self._missed_steps.pop(actor_id, None)
            else:
                self._missed_steps[actor_id] = missed

        tracked = {actor_id for actor_id in self._slots if actor_id is not None}
        for actor_id in ranked_visible_ids:
            if actor_id in tracked:
                continue
            try:
                free_slot = self._slots.index(None)
            except ValueError:
                # All remaining slots are missing-ID reservations.  Reclaim
                # the stalest one for the visible hazard.
                missing_slots = [
                    (self._missed_steps.get(slot_id or "", 0), slot)
                    for slot, slot_id in enumerate(self._slots)
                    if slot_id not in visible
                ]
                if not missing_slots:
                    raise RuntimeError("Top-K actor selection has no reclaimable slot")
                _, free_slot = max(missing_slots)
                old_id = self._slots[free_slot]
                if old_id is not None:
                    self._missed_steps.pop(old_id, None)
            self._slots[free_slot] = actor_id
            tracked.add(actor_id)

        encoded = np.zeros(self.observation_dim, dtype=np.float32)
        for slot, actor_id in enumerate(self._slots):
            if actor_id is None or actor_id not in visible:
                continue
            agent, rel_x, rel_y, _ = visible[actor_id]
            rel_vx_world = float(agent.vx) - float(ego_vx)
            rel_vy_world = float(agent.vy) - float(ego_vy)
            rel_vx, rel_vy = rotate_to_ego(rel_vx_world, rel_vy_world)
            start = slot * self.AGENT_DIM
            encoded[start : start + self.AGENT_DIM] = (
                1.0,
                1.0 if agent.kind == "cone" else 0.0,
                1.0 if agent.kind == "vehicle" else 0.0,
                1.0 if agent.kind == "walker" else 0.0,
                float(np.clip(rel_x / self.radius_m, -1.0, 1.0)),
                float(np.clip(rel_y / self.radius_m, -1.0, 1.0)),
                float(
                    np.clip(
                        rel_vx / self.relative_speed_scale_mps,
                        -1.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        rel_vy / self.relative_speed_scale_mps,
                        -1.0,
                        1.0,
                    )
                ),
            )
        return encoded


# The old name was never wired into the production environment.  Keep an
# import-compatible alias while making the compact current-frame contract
# explicit for downstream tools.
AgentHistoryEncoder = TrackedAgentEncoder
