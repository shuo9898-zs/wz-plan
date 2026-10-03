"""PPO V2 background traffic with a pending-aware hard capacity."""
from __future__ import annotations

import math
import random

import traci

from sync.background_traffic import BackgroundTrafficSynchronizer, _is_proxy


class BackgroundTrafficSynchronizerV2(BackgroundTrafficSynchronizer):
    """Prevent accepted-but-not-departed vehicles from bypassing the cap.

    SUMO does not include pending insertion requests in ``vehicle.getIDList``.
    Counting only that list creates an unbounded queue whenever a route entry
    is congested.  V2 counts the union of active and pending background IDs.
    """

    def __init__(
        self,
        *args,
        route_depart_pos_ranges_m: dict[str, tuple[float, float]] | None = None,
        route_traffic_overrides: dict[str, dict[str, float | int]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._route_depart_pos_ranges_m = dict(
            route_depart_pos_ranges_m or {}
        )
        self._route_traffic_overrides = {
            route_id: dict(profile)
            for route_id, profile in (route_traffic_overrides or {}).items()
        }
        # SUMO exposes active routes directly, but a just-accepted request may
        # not appear in either the active or pending list until simulationStep.
        # Keep an episode-local ledger so those requests still consume their
        # route pool's hard capacity immediately.
        self._vehicle_route_ids: dict[str, str] = {}
        self._awaiting_observation_ids: set[str] = set()

    def set_connection(self, conn: traci.connection.Connection) -> None:
        super().set_connection(conn)
        self._vehicle_route_ids.clear()
        self._awaiting_observation_ids.clear()

    def start_episode(self, episode_id: int) -> None:
        super().start_episode(episode_id)
        self._vehicle_route_ids.clear()
        self._awaiting_observation_ids.clear()

    def _depart_pos_for_route(self, route_id: str) -> str | float:
        bounds = getattr(self, "_route_depart_pos_ranges_m", {}).get(route_id)
        if bounds is None:
            return "base"
        lower, upper = bounds
        return random.uniform(float(lower), float(upper))

    def _request_vehicle(self, route: str, vtype: str) -> bool:
        vehicle_id = self._next_vehicle_id()
        try:
            self._conn.vehicle.add(
                vehID=vehicle_id,
                routeID=route,
                typeID=vtype,
                depart="now",
                departLane="best",
                departPos=self._depart_pos_for_route(route),
                departSpeed="max",
            )
        except traci.TraCIException:
            return False
        self._vehicle_route_ids[vehicle_id] = route
        self._awaiting_observation_ids.add(vehicle_id)
        return True

    @staticmethod
    def _sample_poisson(rate_veh_s: float, dt: float) -> int:
        lam = rate_veh_s * dt
        if lam <= 0.0:
            return 0
        arrivals = 0
        probability = math.exp(-lam)
        cumulative = probability
        sample = random.random()
        while sample > cumulative:
            arrivals += 1
            probability *= lam / arrivals
            cumulative += probability
            if arrivals > 10:
                break
        return arrivals

    def _background_state(
        self,
    ) -> tuple[set[str], dict[str, str | None]]:
        active_ids = {
            vehicle_id
            for vehicle_id in self._conn.vehicle.getIDList()
            if not _is_proxy(vehicle_id)
        }
        pending_ids = {
            vehicle_id
            for vehicle_id in self._conn.simulation.getPendingVehicles()
            if not _is_proxy(vehicle_id)
        }
        awaiting_ids = getattr(self, "_awaiting_observation_ids", set())
        observed_awaiting = awaiting_ids & (active_ids | pending_ids)
        awaiting_ids.difference_update(observed_awaiting)
        occupied_ids = active_ids | pending_ids | awaiting_ids

        route_ledger = getattr(self, "_vehicle_route_ids", {})
        for vehicle_id in tuple(route_ledger):
            if vehicle_id not in occupied_ids:
                route_ledger.pop(vehicle_id, None)

        routes_by_id: dict[str, str | None] = {}
        for vehicle_id in occupied_ids:
            route_id = route_ledger.get(vehicle_id)
            if route_id is None and vehicle_id in active_ids | pending_ids:
                try:
                    route_id = self._conn.vehicle.getRouteID(vehicle_id)
                except (AttributeError, traci.TraCIException):
                    route_id = None
            routes_by_id[vehicle_id] = route_id
        return occupied_ids, routes_by_id

    def _spawn_pool(
        self,
        *,
        routes: tuple[str, ...],
        rate_veh_s: float,
        dt: float,
        vtype: str,
        maximum: int,
        occupied: int,
    ) -> None:
        if not routes or occupied >= maximum:
            return
        arrivals = self._sample_poisson(rate_veh_s, dt)
        for _ in range(min(arrivals, maximum - occupied)):
            self._request_vehicle(random.choice(routes), vtype)

    def spawn_initial_traffic(
        self,
        n: int,
        routes: tuple[str, ...],
        vtype: str = "car",
    ) -> None:
        """Populate an episode, honoring route-specific insertion ranges."""
        available = tuple(
            route for route in routes if route in self._conn.route.getIDList()
        )
        if not available:
            return

        overrides = getattr(self, "_route_traffic_overrides", {})
        # Submit explicit routes first.  In S3 the front route can share its
        # first edge with a rear route; putting its five requests behind the
        # rear batch would leave them pending until the ego reaches the entry.
        for route in available:
            profile = overrides.get(route)
            if profile is None:
                continue
            initial = min(
                int(profile["initial_background_vehicles"]),
                int(profile["max_background_vehicles"]),
            )
            for _ in range(initial):
                self._request_vehicle(route, vtype)

        default_routes = tuple(route for route in available if route not in overrides)
        for _ in range(n):
            if not default_routes:
                break
            self._request_vehicle(random.choice(default_routes), vtype)

    def tick_poisson_spawn(
        self,
        rate_veh_s: float,
        dt: float,
        routes: tuple[str, ...],
        vtype: str = "car",
        max_vehicles: int = 40,
    ) -> None:
        available = tuple(
            route for route in routes if route in self._conn.route.getIDList()
        )
        if not available:
            return

        occupied_ids, routes_by_id = self._background_state()
        overrides = getattr(self, "_route_traffic_overrides", {})
        if not overrides:
            self._spawn_pool(
                routes=available,
                rate_veh_s=rate_veh_s,
                dt=dt,
                vtype=vtype,
                maximum=max_vehicles,
                occupied=len(occupied_ids),
            )
            return

        override_route_ids = set(overrides)
        default_routes = tuple(
            route for route in available if route not in override_route_ids
        )
        default_route_set = set(default_routes)
        default_occupied = sum(
            route_id in default_route_set or route_id is None
            for route_id in routes_by_id.values()
        )
        self._spawn_pool(
            routes=default_routes,
            rate_veh_s=rate_veh_s,
            dt=dt,
            vtype=vtype,
            maximum=max_vehicles,
            occupied=default_occupied,
        )

        for route in available:
            profile = overrides.get(route)
            if profile is None:
                continue
            route_occupied = sum(
                route_id == route for route_id in routes_by_id.values()
            )
            self._spawn_pool(
                routes=(route,),
                rate_veh_s=float(profile["bg_spawn_rate_veh_s"]),
                dt=dt,
                vtype=vtype,
                maximum=int(profile["max_background_vehicles"]),
                occupied=route_occupied,
            )


__all__ = ["BackgroundTrafficSynchronizerV2"]
