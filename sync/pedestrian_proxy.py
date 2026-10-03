"""
PedestrianProxySynchronizer
============================
Inserts zero-speed SUMO vehicle proxies for CARLA pedestrians near the
road centreline so that SUMO background vehicles yield via car-following.

No CARLA map API is used.  "On road" is a perpendicular-distance test
against the infinite line through `road_ref` in direction
`road_heading_deg` — this works for a road of any orientation, so the
same code serves every WZ1/WZ2/WZ3 scenario.
The proxy heading is aligned to road_heading_deg (the lane direction),
not the pedestrian's own facing.

The proxy is NEVER mirrored back to CARLA.  The real CARLA pedestrian
handles visualisation, collision detection, observation, and motion.
"""
from __future__ import annotations

import logging
import math

import carla
import traci

from sync.bridge import CarlaSumoCoordinateBridge

logger = logging.getLogger(__name__)

_PROXY_EXTENT = carla.Vector3D(1.0, 0.5, 0.9)


class PedestrianProxySynchronizer:
    """
    Parameters
    ----------
    bridge           : CarlaSumoCoordinateBridge
    road_ref         : (x, y)  any point on the road centreline (CARLA m)
    road_heading_deg : float   lane direction in CARLA yaw degrees
    road_half_width  : float   perpendicular distance (m) that counts as "on road"
    proxy_prefix     : str     SUMO IDs are "{prefix}{carla_actor_id}"
    """

    def __init__(self,
                 bridge: CarlaSumoCoordinateBridge,
                 conn: traci.connection.Connection,
                 road_ref: tuple[float, float],
                 road_heading_deg: float,
                 road_half_width: float = 5.0,
                 proxy_prefix: str = "ped_proxy_") -> None:
        self._bridge  = bridge
        self._conn    = conn
        self._prefix  = proxy_prefix
        self._ref_x, self._ref_y = road_ref
        theta = math.radians(road_heading_deg)
        self._fwd_x, self._fwd_y = math.cos(theta), math.sin(theta)
        self._half_w  = road_half_width
        self._heading = road_heading_deg
        # carla_id (str) → {proxy_id, …}
        self._active: dict[str, set[str]] = {}

    def set_connection(self, conn: traci.connection.Connection) -> None:
        """Rebind to a fresh TraCI connection after a SUMO process restart."""
        self._conn = conn

    # ------------------------------------------------------------------ #
    #  Per-step sync                                                       #
    # ------------------------------------------------------------------ #

    def sync(self, pedestrian_actors: list[carla.Actor]) -> None:
        """Call AFTER carla_world.tick() and BEFORE traci.simulationStep()."""
        current = {str(p.id) for p in pedestrian_actors}
        for pid in set(self._active) - current:
            self._remove(pid)
        for actor in pedestrian_actors:
            self._update(actor, str(actor.id))

    def destroy(self) -> None:
        for pid in list(self._active):
            self._remove(pid)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _on_road(self, x: float, y: float) -> bool:
        """Perpendicular distance from the road centreline through road_ref."""
        dx, dy = x - self._ref_x, y - self._ref_y
        return abs(self._fwd_x * dy - self._fwd_y * dx) <= self._half_w

    def _update(self, actor: carla.Actor, pid: str) -> None:
        loc = actor.get_location()
        if not self._on_road(loc.x, loc.y):
            self._remove(pid)
            return
        # Proxy aligned to road heading, NOT the pedestrian's own yaw
        proxy_t  = carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=0.3),
            carla.Rotation(yaw=self._heading),
        )
        self._ensure(f"{self._prefix}{pid}", proxy_t, pid)

    def _ensure(self, proxy_id: str, t: carla.Transform, pid: str) -> None:
        route_ids = self._conn.route.getIDList()
        if not route_ids:
            return
        if proxy_id not in self._conn.vehicle.getIDList():
            try:
                self._conn.vehicle.add(proxy_id, route_ids[0], "passenger", depart="now")
                self._conn.vehicle.setSpeedMode(proxy_id, 0)
                self._conn.vehicle.setLaneChangeMode(proxy_id, 0)
                self._conn.vehicle.setMaxSpeed(proxy_id, 0.0)
                self._conn.vehicle.setMinGap(proxy_id, 0.0)
                self._active.setdefault(pid, set()).add(proxy_id)
            except traci.TraCIException as exc:
                logger.debug("ped proxy spawn failed: %s", exc)
                return
        pos = self._bridge.carla_to_sumo(t, _PROXY_EXTENT)
        try:
            self._conn.vehicle.moveToXY(
                proxy_id, "", -1, pos["x"], pos["y"], pos["angle"],
                keepRoute=2, matchThreshold=20.0,
            )
            self._conn.vehicle.setSpeed(proxy_id, 0.0)
        except traci.TraCIException:
            pass

    def _remove(self, pid: str) -> None:
        for proxy_id in self._active.get(pid, set()):
            try:
                if proxy_id in self._conn.vehicle.getIDList():
                    self._conn.vehicle.remove(proxy_id)
            except Exception:
                pass
        self._active.pop(pid, None)
