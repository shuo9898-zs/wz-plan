"""
BackgroundTrafficSynchronizer
==============================
Mirrors SUMO background vehicles into CARLA as kinematic actors.

Closed-loop update order (per spec §7):
    traci.simulationStep()               ← SUMO has advanced
    → background_traffic.sync()          ← pull SUMO state into CARLA

Actor mapping maintained:
    sumo_vehicle_id → carla.Actor

Proxies (ego_proxy, ped_proxy_*) are excluded from mirroring.

Also provides:
    spawn_initial_traffic()  — call at episode reset to populate traffic.
    tick_poisson_spawn()     — call each step for continuous Poisson injection.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from typing import Callable

import carla
import traci

from sync.bridge import CarlaSumoCoordinateBridge

logger = logging.getLogger(__name__)

# Half-extents used for coordinate conversion of background vehicles
_DEFAULT_EXTENT = carla.Vector3D(2.5, 1.0, 0.75)

# SUMO ID prefixes that belong to the control layer (not real traffic)
_PROXY_PREFIXES = ("ego_proxy", "ped_proxy_")


class MirrorSpawnExhaustedError(RuntimeError):
    """The current valid SUMO transform is occupied in CARLA."""


def _is_proxy(sumo_id: str) -> bool:
    return any(sumo_id.startswith(p) for p in _PROXY_PREFIXES)


@dataclass
class _PlatoonState:
    """Episode-local state for a one-request-at-a-time SUMO platoon."""

    routes: tuple[str, ...]
    vtype: str
    max_vehicles: int
    size_min: int
    size_max: int
    headway_min_s: float
    headway_max_s: float
    gap_min_s: float
    gap_max_s: float
    group_size: int
    group_departures: int
    next_request_time_s: float
    pending_vehicle_id: str | None = None
    pending_requested_time_s: float | None = None
    pending_warning_emitted: bool = False


class BackgroundTrafficSynchronizer:
    """
    Manages CARLA ghost actors that reflect SUMO background vehicles.

    Parameters
    ----------
    bridge      : CarlaSumoCoordinateBridge
    carla_world : carla.World
    blueprint_id: str  CARLA blueprint used for all background vehicles.
    """

    def __init__(self,
                 bridge: CarlaSumoCoordinateBridge,
                 carla_world: carla.World,
                 conn: traci.connection.Connection,
                 worker_id: int = 0,
                 blueprint_id: str = "vehicle.tesla.model3",
                 actor_register: Callable[[carla.Actor, str], None] | None = None,
                 actor_unregister: Callable[[int], None] | None = None,
                 spawn_attempt_callback: Callable[[str, int, bool, str | None], None] | None = None) -> None:
        self._bridge = bridge
        self._world  = carla_world
        self._conn   = conn
        self._worker_id   = worker_id
        self._episode_id  = 0
        self._bg_counter  = 0
        self._bp     = carla_world.get_blueprint_library().find(blueprint_id)
        if self._bp.has_attribute("role_name"):
            self._bp.set_attribute("role_name", "sumo_background")
        self._actor_register = actor_register
        self._actor_unregister = actor_unregister
        self._spawn_attempt_callback = spawn_attempt_callback

        # sumo_id → carla.Actor
        self._actor_map: dict[str, carla.Actor] = {}
        # sumo_id → speed (m/s), read from SUMO directly since the mirrored
        # actors have physics disabled and their own get_velocity() is not
        # meaningful (set_transform() does not drive CARLA's velocity state).
        self._speed_map: dict[str, float] = {}
        self._platoon_state: _PlatoonState | None = None

    def set_connection(self, conn: traci.connection.Connection) -> None:
        """Rebind to a fresh TraCI connection after a SUMO process restart."""
        self._conn = conn
        self._platoon_state = None

    def start_episode(self, episode_id: int) -> None:
        """Call once per episode reset — resets the deterministic vehicle-ID
        counter so IDs are unique within this worker/episode and can never
        collide with another worker's or another episode's vehicles."""
        self._episode_id = episode_id
        self._bg_counter = 0
        self._platoon_state = None

    def _next_vehicle_id(self) -> str:
        vid = f"w{self._worker_id}_ep{self._episode_id}_bg{self._bg_counter}"
        self._bg_counter += 1
        return vid

    # ------------------------------------------------------------------ #
    #  Episode initialisation                                              #
    # ------------------------------------------------------------------ #

    def spawn_initial_traffic(self,
                               n: int,
                               routes: tuple[str, ...],
                               vtype: str = "car") -> None:
        """
        Spawn *n* background vehicles in SUMO at episode start.
        They will be mirrored to CARLA on the first sync() call.

        Vehicles are distributed randomly across the provided route IDs.
        Use departSpeed="max" so vehicles enter at road-speed immediately.
        """
        available = self._conn.route.getIDList()
        bg_routes  = [r for r in routes if r in available]
        if not bg_routes:
            logger.warning("None of the configured bg_routes found in SUMO. "
                           "Check routes.rou.xml.")
            return

        for _ in range(n):
            vid   = self._next_vehicle_id()
            route = random.choice(bg_routes)
            try:
                self._conn.vehicle.add(
                    vehID        = vid,
                    routeID      = route,
                    typeID       = vtype,
                    depart       = "now",
                    departLane   = "best",
                    departSpeed  = "max",
                )
            except traci.TraCIException as exc:
                logger.debug("Could not spawn bg vehicle %s: %s", vid, exc)

    def tick_poisson_spawn(self,
                            rate_veh_s: float,
                            dt: float,
                            routes: tuple[str, ...],
                            vtype: str = "car",
                            max_vehicles: int = 40) -> None:
        """
        Probabilistic per-step injection based on a Poisson process.

        rate_veh_s : desired vehicles/second arrival rate
        dt         : step duration in seconds (0.1 for 10 Hz)
        max_vehicles: hard cap on total SUMO background vehicles
        """
        current_bg = sum(1 for v in self._conn.vehicle.getIDList() if not _is_proxy(v))
        if current_bg >= max_vehicles:
            return

        # Expected arrivals this step ~ Poisson(lambda = rate * dt)
        import math
        lam = rate_veh_s * dt
        # Sample from Poisson distribution (simple approximation for small lambda)
        k   = 0
        p   = math.exp(-lam)
        s   = p
        u   = random.random()
        while u > s:
            k += 1
            p *= lam / k
            s += p
            if k > 10:
                break

        available = [r for r in routes if r in self._conn.route.getIDList()]
        if not available:
            return

        for i in range(k):
            if current_bg + i >= max_vehicles:
                break
            vid   = self._next_vehicle_id()
            route = random.choice(available)
            try:
                self._conn.vehicle.add(
                    vehID       = vid,
                    routeID     = route,
                    typeID      = vtype,
                    depart      = "now",
                    departLane  = "best",
                    departSpeed = "max",
                )
            except traci.TraCIException:
                pass

    # ------------------------------------------------------------------ #
    #  Per-step SUMO → CARLA sync                                         #
    # ------------------------------------------------------------------ #

    def start_platoon_traffic(
        self,
        *,
        routes: tuple[str, ...],
        vtype: str = "car",
        max_vehicles: int = 8,
        size_min: int = 3,
        size_max: int = 4,
        headway_min_s: float = 1.6,
        headway_max_s: float = 2.0,
        gap_min_s: float = 10.0,
        gap_max_s: float = 12.0,
    ) -> None:
        """Start compact traffic waves without filling SUMO's pending queue.

        Only the first vehicle is submitted here. A later request is not
        submitted until SUMO reports that the tracked request actually
        departed. Adding a whole platoon with ``depart='now'`` would instead
        create a hidden queue that fills every available road opening.
        """
        if size_min < 1 or size_max < size_min:
            raise ValueError("platoon size bounds must satisfy 1 <= min <= max")
        if headway_min_s < 0.0 or headway_max_s < headway_min_s:
            raise ValueError("platoon headway bounds must satisfy 0 <= min <= max")
        if gap_min_s < 0.0 or gap_max_s < gap_min_s:
            raise ValueError("platoon gap bounds must satisfy 0 <= min <= max")

        available = tuple(r for r in routes if r in self._conn.route.getIDList())
        if not available:
            logger.warning(
                "None of the configured platoon routes found in SUMO. "
                "Check routes.rou.xml."
            )
            self._platoon_state = None
            return

        now = float(self._conn.simulation.getTime())
        self._platoon_state = _PlatoonState(
            routes=available,
            vtype=vtype,
            max_vehicles=max_vehicles,
            size_min=size_min,
            size_max=size_max,
            headway_min_s=headway_min_s,
            headway_max_s=headway_max_s,
            gap_min_s=gap_min_s,
            gap_max_s=gap_max_s,
            group_size=random.randint(size_min, size_max),
            group_departures=0,
            next_request_time_s=now,
        )
        logger.info(
            "SUMO platoon started | episode=%d size=%d headway_s=%.1f-%.1f "
            "gap_s=%.1f-%.1f cap=%d",
            self._episode_id,
            self._platoon_state.group_size,
            headway_min_s,
            headway_max_s,
            gap_min_s,
            gap_max_s,
            max_vehicles,
        )
        self._try_queue_platoon_vehicle(now)

    def tick_platoon_spawn(self) -> None:
        """Advance the platoon scheduler using actual SUMO departures.

        ``vehicle.add`` accepting a request is not a departure: the vehicle
        may remain pending until its lane is free. Keeping at most one such
        request prevents the unbounded backlog produced by the former reset
        batch plus continuous Poisson arrivals.
        """
        state = self._platoon_state
        if state is None:
            return

        now = float(self._conn.simulation.getTime())
        departed_ids = set(self._conn.simulation.getDepartedIDList())
        active_ids = set(self._conn.vehicle.getIDList())
        # Usually the exact ID is present in this step's departed list. The
        # active-ID fallback covers a rare ego-proxy recovery that performs an
        # extra SUMO micro-step before this method can observe that list. It
        # anchors the delay to the later observation time, so the configured
        # minimum headway/gap can only grow, never shrink.
        pending_is_active = state.pending_vehicle_id in active_ids
        if state.pending_vehicle_id in departed_ids or pending_is_active:
            departed_id = state.pending_vehicle_id
            state.pending_vehicle_id = None
            state.pending_requested_time_s = None
            state.pending_warning_emitted = False
            state.group_departures += 1

            if state.group_departures >= state.group_size:
                completed_size = state.group_size
                state.group_size = random.randint(state.size_min, state.size_max)
                state.group_departures = 0
                delay = random.uniform(state.gap_min_s, state.gap_max_s)
                logger.debug(
                    "SUMO platoon complete | last_vehicle=%s size=%d "
                    "gap_s=%.2f next_size=%d",
                    departed_id,
                    completed_size,
                    delay,
                    state.group_size,
                )
            else:
                delay = random.uniform(
                    state.headway_min_s, state.headway_max_s
                )
            state.next_request_time_s = now + delay

        if (
            state.pending_vehicle_id is not None
            and state.pending_requested_time_s is not None
            and now - state.pending_requested_time_s >= 30.0
            and not state.pending_warning_emitted
        ):
            logger.warning(
                "SUMO platoon request still pending after %.1fs | vehicle=%s",
                now - state.pending_requested_time_s,
                state.pending_vehicle_id,
            )
            state.pending_warning_emitted = True

        if (
            state.pending_vehicle_id is None
            and now + 1e-9 >= state.next_request_time_s
        ):
            self._try_queue_platoon_vehicle(now)

    def _try_queue_platoon_vehicle(self, now: float) -> None:
        state = self._platoon_state
        if state is None or state.pending_vehicle_id is not None:
            return

        current_bg = sum(
            1
            for vehicle_id in self._conn.vehicle.getIDList()
            if not _is_proxy(vehicle_id)
        )
        if current_bg >= state.max_vehicles:
            return

        vehicle_id = self._next_vehicle_id()
        try:
            self._conn.vehicle.add(
                vehID=vehicle_id,
                routeID=random.choice(state.routes),
                typeID=state.vtype,
                depart="now",
                departLane="best",
                departSpeed="max",
            )
        except traci.TraCIException as exc:
            # No scheduler state advances on rejection. A later tick retries,
            # while at most one accepted request can ever remain pending.
            logger.debug("Could not queue platoon vehicle %s: %s", vehicle_id, exc)
            state.next_request_time_s = now
            return
        state.pending_vehicle_id = vehicle_id
        state.pending_requested_time_s = now
        state.pending_warning_emitted = False

    def sync(self) -> None:
        """
        Mirror all non-proxy SUMO vehicles into CARLA.
        Call AFTER traci.simulationStep().
        """
        sumo_bg = {v for v in self._conn.vehicle.getIDList() if not _is_proxy(v)}

        # Destroy CARLA actors for vehicles that left SUMO
        departed = set(self._actor_map) - sumo_bg
        for sid in departed:
            self._destroy(sid)

        # Spawn or update
        for sid in sumo_bg:
            try:
                pos_2d = self._conn.vehicle.getPosition(sid)
                angle  = self._conn.vehicle.getAngle(sid)
                t      = self._bridge.sumo_to_carla(
                    pos_2d[0], pos_2d[1], angle, _DEFAULT_EXTENT
                )
                self._speed_map[sid] = self._conn.vehicle.getSpeed(sid)
                if sid not in self._actor_map:
                    self._spawn(sid, t)
                else:
                    self._update(sid, t)
            except traci.TraCIException as exc:
                logger.debug("SUMO read error for %s: %s", sid, exc)

    # ------------------------------------------------------------------ #
    #  Full cleanup                                                        #
    # ------------------------------------------------------------------ #

    def destroy_all(self) -> None:
        """Destroy all tracked CARLA background actors."""
        for sid in list(self._actor_map):
            self._destroy(sid)

    def get_actor_map(self) -> dict[str, carla.Actor]:
        """Read-only snapshot of the current sumo_id → carla_actor mapping."""
        return dict(self._actor_map)

    def get_speed_map(self) -> dict[str, float]:
        """Read-only snapshot of the current sumo_id → speed (m/s) mapping.
        Use this instead of actor.get_velocity() — the mirrored actors have
        physics disabled, so their own velocity state is not meaningful."""
        return dict(self._speed_map)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _spawn(self, sumo_id: str, transform: carla.Transform) -> None:
        """Spawn a CARLA mirror actor for a SUMO background vehicle.

        If the spawn position is occupied (typically overlaps the Ego or
        another mirror), skip this vehicle for now — do NOT raise.  SUMO will
        advance next step and the vehicle's transform will move; sync() will
        retry the spawn on the next call.  This prevents transient mirror
        collisions from aborting the whole RL episode.
        """
        actor = self._world.try_spawn_actor(self._bp, transform)
        if actor is None:
            if self._spawn_attempt_callback:
                self._spawn_attempt_callback("sumo_mirror", 1, False, "collision")
            logger.debug(
                "CARLA mirror spawn collided for SUMO vehicle '%s'; will retry next tick.",
                sumo_id,
            )
            return   # skip this vehicle for now; sync() will retry next step

        if self._spawn_attempt_callback:
            self._spawn_attempt_callback("sumo_mirror", 1, True, None)
        # Register immediately so a failure while configuring this actor
        # cannot leave an untracked stationary Tesla behind.
        self._actor_map[sumo_id] = actor
        if self._actor_register:
            self._actor_register(actor, "sumo_mirror")
        actor.set_simulate_physics(False)
        try:
            actor.set_enable_gravity(False)
        except AttributeError:
            pass

    def _update(self, sumo_id: str, transform: carla.Transform) -> None:
        actor = self._actor_map.get(sumo_id)
        if actor and actor.is_alive:
            actor.set_transform(transform)
        else:
            self._actor_map.pop(sumo_id, None)
            self._spawn(sumo_id, transform)

    def _destroy(self, sumo_id: str) -> None:
        actor = self._actor_map.pop(sumo_id, None)
        self._speed_map.pop(sumo_id, None)
        if actor and actor.is_alive:
            try:
                actor.destroy()
            except Exception:
                logger.debug("Could not destroy SUMO mirror actor %s", actor.id, exc_info=True)
                return
        if actor and self._actor_unregister:
            self._actor_unregister(actor.id)
