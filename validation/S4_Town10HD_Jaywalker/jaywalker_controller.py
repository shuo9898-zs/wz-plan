"""CARLA-only three-crossing jaywalker lifecycle for S4.

No SUMO process or pedestrian proxy participates.  Each episode owns three
crossing events, nominally triggered when Ego is 45 m, 30 m, and 15 m before
the current layout's trigger anchor.  The owner-authored spawn and disappear
points define the actual walking direction; ``collision_target`` is metadata.
"""
from __future__ import annotations

import math
from typing import Callable

import carla

from config.scenario_config import JaywalkerConfig


class JaywalkerController:
    def __init__(
        self,
        world: carla.World,
        config: JaywalkerConfig,
        *,
        road_heading_deg: float = 0.0,
        actor_register: Callable[[carla.Actor, str], None] | None = None,
        actor_unregister: Callable[[int], None] | None = None,
    ) -> None:
        self.world = world
        self.config = config
        self.road_heading_deg = float(road_heading_deg)
        self.actor_register = actor_register
        self.actor_unregister = actor_unregister
        self._actors: list[carla.Actor] = []
        self._previous_remaining_m: float | None = None
        self._pending_gates: list[float] = []
        self._triggered_gates: set[float] = set()
        self._spawned_count = 0
        self._completed_count = 0

    def start_episode(self, ego: carla.Actor) -> None:
        """Reset wave state and queue any thresholds already passed at reset.

        If an owner-authored origin ever begins inside a gate, that event is
        marked due instead of being silently lost. ``update`` performs at most
        one spawn attempt per simulation tick so three actors are never created
        on one transform in the same CARLA frame.
        """
        self._actors.clear()
        self._pending_gates.clear()
        self._triggered_gates.clear()
        self._spawned_count = 0
        self._completed_count = 0
        remaining = self._remaining_to_anchor(ego)
        self._previous_remaining_m = remaining
        for gate in self.config.trigger_distances_m:
            if remaining <= gate:
                self._queue_gate(gate)

    def update(self, ego: carla.Actor) -> None:
        """Detect new gates, advance active walkers, then try one due spawn."""
        remaining = self._remaining_to_anchor(ego)
        previous = self._previous_remaining_m
        if previous is None:
            previous = remaining

        for gate in self.config.trigger_distances_m:
            if gate in self._triggered_gates:
                continue
            # Normal upstream crossing, with a fallback for teleport/reset
            # directly inside a gate.
            if (previous > gate >= remaining) or remaining <= gate:
                self._queue_gate(gate)
        self._previous_remaining_m = remaining

        self._advance_walkers()
        if self._pending_gates:
            self._try_spawn_next()

    def active_actors(self) -> list[carla.Actor]:
        """Return live walkers for the observation's other-agent sectors."""
        return [actor for actor in self._actors if actor.is_alive]

    def status(self) -> dict:
        total = len(self.config.trigger_distances_m)
        return {
            "triggered_count": len(self._triggered_gates),
            "spawned_count": self._spawned_count,
            "completed_count": self._completed_count,
            "pending_count": len(self._pending_gates),
            "active_count": len(self.active_actors()),
            "remaining_count": total - len(self._triggered_gates),
            "trigger_distances_m": tuple(self.config.trigger_distances_m),
        }

    def _remaining_to_anchor(self, ego: carla.Actor) -> float:
        anchor_x, anchor_y, _ = self.config.trigger_anchor
        ego_loc = ego.get_location()
        heading = math.radians(self.road_heading_deg)
        forward_x, forward_y = math.cos(heading), math.sin(heading)
        return (
            (anchor_x - float(ego_loc.x)) * forward_x
            + (anchor_y - float(ego_loc.y)) * forward_y
        )

    def _queue_gate(self, gate: float) -> None:
        value = float(gate)
        self._triggered_gates.add(value)
        self._pending_gates.append(value)

    def _try_spawn_next(self) -> None:
        bp = self.world.get_blueprint_library().find(self.config.blueprint)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "episode_pedestrian")
        x, y, z = self.config.spawn
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(yaw=self.config.yaw_deg),
        )
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            # The shared spawn may still be occupied by an earlier walker.
            # Retain the event and retry on the next simulation tick.
            return
        self._pending_gates.pop(0)
        self._actors.append(actor)
        self._spawned_count += 1
        if self.actor_register:
            self.actor_register(actor, "jaywalker")
        self._apply_control(actor)

    def _advance_walkers(self) -> None:
        target_x, target_y, _ = self.config.disappear
        survivors: list[carla.Actor] = []
        for actor in self._actors:
            if not actor.is_alive:
                self._unregister(actor.id)
                continue
            loc = actor.get_location()
            distance = math.hypot(target_x - loc.x, target_y - loc.y)
            if distance <= self.config.disappear_radius_m:
                self._destroy_actor(actor)
                self._completed_count += 1
                continue
            self._apply_control(actor)
            survivors.append(actor)
        self._actors = survivors

    def _apply_control(self, actor: carla.Actor) -> None:
        target_x, target_y, _ = self.config.disappear
        loc = actor.get_location()
        dx, dy = target_x - loc.x, target_y - loc.y
        norm = max(math.hypot(dx, dy), 1e-6)
        control = carla.WalkerControl()
        control.direction = carla.Vector3D(dx / norm, dy / norm, 0.0)
        control.speed = float(self.config.speed_mps)
        control.jump = False
        actor.apply_control(control)

    def _unregister(self, actor_id: int) -> None:
        if self.actor_unregister:
            self.actor_unregister(actor_id)

    def _destroy_actor(self, actor: carla.Actor) -> None:
        actor_id = actor.id
        try:
            if actor.is_alive:
                actor.destroy()
        finally:
            self._unregister(actor_id)

    def destroy(self) -> None:
        """Destroy every active walker and discard all unspawned events."""
        for actor in list(self._actors):
            self._destroy_actor(actor)
        self._actors.clear()
        self._pending_gates.clear()
