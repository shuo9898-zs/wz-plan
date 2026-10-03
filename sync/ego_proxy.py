"""
EgoProxySynchronizer
====================
Keeps the SUMO "ego_proxy" vehicle aligned with the CARLA Ego every step.

Closed-loop update order (per spec §4):
    ... carla_world.tick()          ← CARLA has advanced
    → ego_proxy.sync(ego_actor)     ← push CARLA state into SUMO
    → traci.simulationStep()        ← SUMO reacts

Ego proxy configuration in SUMO:
    - Speed-mode  = 0  (disable all safety checks)
    - Lane-change = 0  (disable)
    - MinGap      = 0
    - MaxSpeed    = 200 m/s
    - keepRoute=2 in moveToXY (off-road position is tolerated)

SUMO background vehicles react to the proxy through their normal
car-following and lane-changing models.
"""
from __future__ import annotations

import logging
import math

import carla
import traci

from sync.bridge import CarlaSumoCoordinateBridge

logger = logging.getLogger(__name__)

# Standard passenger car half-extents used for the proxy
_PROXY_EXTENT = carla.Vector3D(2.5, 1.0, 0.75)


class EgoProxySynchronizer:
    """
    Manages the SUMO ego_proxy vehicle for one episode.

    Parameters
    ----------
    bridge       : CarlaSumoCoordinateBridge
    ego_proxy_id : str  (default "ego_proxy")
    """

    def __init__(self,
                 bridge: CarlaSumoCoordinateBridge,
                 conn: traci.connection.Connection,
                 ego_proxy_id: str = "ego_proxy") -> None:
        self._bridge   = bridge
        self._conn     = conn
        self._base_proxy_id = ego_proxy_id
        self._proxy_id = ego_proxy_id
        self._proxy_generation = 0
        self._active   = False

    def set_connection(self, conn: traci.connection.Connection) -> None:
        """Rebind to a fresh TraCI connection after a SUMO process restart."""
        self._conn = conn
        self._proxy_id = self._base_proxy_id
        self._active = False

    # ------------------------------------------------------------------ #
    #  Episode lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def reset(self, ego_actor: carla.Actor) -> None:
        """
        (Re)insert the ego proxy into the current SUMO simulation.
        Call this after SUMO is started and *before* the first
        traci.simulationStep() of the episode.
        """
        # Remove stale proxy if present
        if self._proxy_id in self._conn.vehicle.getIDList():
            self._conn.vehicle.remove(self._proxy_id)
            self._active = False

        # Pick the longest available route as a safety net. moveToXY normally
        # overrides the route position, but if the manual CARLA point does not
        # match a SUMO lane on the first reset, a one-edge placeholder can
        # arrive during the background-spread micro-steps. SUMO then reserves
        # that used vehicle ID and rejects a same-ID reinsertion.
        route_ids = self._conn.route.getIDList()
        if not route_ids:
            raise RuntimeError(
                "No SUMO routes available. Ensure routes.rou.xml is loaded."
            )
        route_id = max(
            route_ids,
            key=lambda candidate: len(self._conn.route.getEdges(candidate)),
        )

        candidate_id = self._proxy_id
        while True:
            try:
                self._conn.vehicle.add(
                    vehID=candidate_id,
                    routeID=route_id,
                    typeID="passenger",
                    depart="now",
                )
                self._proxy_id = candidate_id
                break
            except traci.TraCIException as exc:
                if "already exists" not in str(exc):
                    raise
                self._proxy_generation += 1
                candidate_id = f"{self._base_proxy_id}_retry_{self._proxy_generation}"

        # One micro-step to materialise the vehicle in SUMO's internal state
        self._conn.simulationStep()

        # Teleport to the CARLA Ego's current position
        pos = self._bridge.carla_to_sumo(ego_actor.get_transform(), _PROXY_EXTENT)
        try:
            self._conn.vehicle.moveToXY(
                vehID         = self._proxy_id,
                edgeID        = "",
                laneIndex     = -1, # -1 leaves the target lane unspecified.
                x             = pos["x"],
                y             = pos["y"],
                angle         = pos["angle"],
                keepRoute     = 2,
                matchThreshold= 20.0,
            )
        except traci.TraCIException:
            pass  # Off-road at spawn is acceptable; proxy will be re-placed each step

        # Disable all SUMO-internal safety / following logic for this proxy
        self._conn.vehicle.setSpeedMode(self._proxy_id, 0)
        self._conn.vehicle.setLaneChangeMode(self._proxy_id, 0)
        self._conn.vehicle.setMinGap(self._proxy_id, 0.0)
        self._conn.vehicle.setTau(self._proxy_id, 0.1)
        self._conn.vehicle.setMaxSpeed(self._proxy_id, 200.0) # Keep the SUMO proxy from limiting the CARLA ego vehicle's speed.
        self._conn.vehicle.setImperfection(self._proxy_id, 0.0)
        velocity = ego_actor.get_velocity()
        # SUMO speed is longitudinal road speed; CARLA's vertical settling
        # velocity must not make a stationary Ego appear to be moving.
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
        # The proxy is externally controlled by CARLA. Force both its current
        # and previous speed so SUMO traffic observes the real Ego state.
        self._conn.vehicle.setSpeed(self._proxy_id, speed)
        self._conn.vehicle.setPreviousSpeed(self._proxy_id, speed)

        self._active = True
        logger.debug("Ego proxy '%s' added at (%.2f, %.2f)",
                     self._proxy_id, pos["x"], pos["y"])

    def destroy(self) -> None:
        """Remove the ego proxy from SUMO. Call at episode cleanup."""
        if self._active:
            try:
                if self._proxy_id in self._conn.vehicle.getIDList():
                    self._conn.vehicle.remove(self._proxy_id)
            except Exception:
                pass
            self._active = False

    # ------------------------------------------------------------------ #
    #  Per-step sync                                                       #
    # ------------------------------------------------------------------ #

    def sync(self, ego_actor: carla.Actor) -> None:
        """
        Push the current CARLA Ego state into SUMO.

        Call AFTER carla_world.tick() and BEFORE traci.simulationStep().

        Uses moveToXY (keepRoute=2) for position and forces both current and
        previous speed so that SUMO background vehicles react to CARLA state.
        """
        if not self._active:
            return

        # Re-insert if proxy disappeared (e.g. SUMO removed it)
        if self._proxy_id not in self._conn.vehicle.getIDList():
            logger.debug("Ego proxy missing — re-inserting.")
            self.reset(ego_actor)
            return

        transform = ego_actor.get_transform()
        velocity  = ego_actor.get_velocity()
        speed     = math.sqrt(velocity.x ** 2 + velocity.y ** 2)

        pos = self._bridge.carla_to_sumo(transform, _PROXY_EXTENT)

        try:
            self._conn.vehicle.moveToXY(
                vehID         = self._proxy_id,
                edgeID        = "",
                laneIndex     = -1,
                x             = pos["x"],
                y             = pos["y"],
                angle         = pos["angle"],
                keepRoute     = 2,
                matchThreshold= 20.0,
            )
            self._conn.vehicle.setSpeed(self._proxy_id, speed)
            self._conn.vehicle.setPreviousSpeed(self._proxy_id, speed)
        except traci.TraCIException:
            pass  # Tolerate off-road; SUMO vehicles still perceive the proxy
