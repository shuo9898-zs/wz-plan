"""
CarlaSumoCoordinateBridge
=========================
Bidirectional coordinate conversion between CARLA and SUMO.

Based on the verified BridgeHelper from history_sumo (see
history_sumo/July27_S1/SUMO/sumo_integration/bridge_helper.py).
Re-implemented as an instance (not a static class) so multiple
environments can run in parallel with independent offsets.

Key conventions
---------------
SUMO  : right-hand system, angle 0° = North (+Y), 90° = East (+X), clockwise.
         Forward vector: fwd_x = sin(yaw),  fwd_y = cos(yaw)
         Vehicle position = front-bumper centre.

CARLA : left-hand system, Y axis inverted w.r.t. SUMO.
         Yaw  0° = East (+X),  90° = South (+Y),  increases clockwise.
         Vehicle position = bounding-box centroid.

Formulas (CARLA ↔ SUMO)
------------------------
  sumo_yaw  = normalize(carla_yaw + 90°)
  carla_yaw = normalize(sumo_yaw  − 90°)
  sumo_y    = −carla_y  + offset_y
  sumo_x    =  carla_x  + offset_x  [before bumper & lateral corrections]
"""
from __future__ import annotations

import math

import carla


def _normalize(angle: float) -> float:
    """Normalise to (−180, +180]."""
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


class CarlaSumoCoordinateBridge:
    """
    Instance-based CARLA ↔ SUMO coordinate bridge.

    Parameters
    ----------
    net_offset : (x_off, y_off)
        Extracted from the SUMO .net.xml <location netOffset="x,y"> tag.
    lateral_shift : float
        Lane-alignment correction in metres (e.g. −4.0 for Town02).
        Applied perpendicular to the vehicle's forward direction.
    """

    def __init__(self, net_offset: tuple[float, float],
                 lateral_shift: float = 0.0) -> None:
        self.net_offset   = net_offset      # (off_x, off_y)
        self.lateral_shift = lateral_shift

    # ------------------------------------------------------------------ #
    #  CARLA → SUMO                                                        #
    # ------------------------------------------------------------------ #

    def carla_to_sumo(self, carla_transform: carla.Transform,
                      extent: carla.Vector3D) -> dict[str, float]:
        """
        Convert a CARLA actor transform (centroid) to a SUMO position dict.

        Returns
        -------
        {'x': float, 'y': float, 'angle': float}
            SUMO coordinate and heading in degrees (0°=North, clockwise).
        """
        carla_x   = carla_transform.location.x
        carla_y   = carla_transform.location.y
        carla_yaw = carla_transform.rotation.yaw

        # Step 1 — angle
        sumo_yaw = _normalize(carla_yaw + 90.0)
        yaw_rad  = math.radians(sumo_yaw)

        # Step 2 — Y inversion (CARLA left-hand → SUMO right-hand)
        x = carla_x
        y = -carla_y

        # Step 3 — remove lateral shift (in SUMO coordinates)
        if self.lateral_shift != 0.0:
            fwd_x  =  math.sin(yaw_rad)
            fwd_y  =  math.cos(yaw_rad)
            right_x =  fwd_y
            right_y = -fwd_x
            x -= right_x * self.lateral_shift
            y -= right_y * self.lateral_shift

        # Step 4 — centroid → front-bumper centre
        fwd_x = math.sin(yaw_rad)
        fwd_y = math.cos(yaw_rad)
        x += fwd_x * extent.x
        y += fwd_y * extent.x

        # Step 5 — add net offset
        x += self.net_offset[0]
        y += self.net_offset[1]

        return {'x': x, 'y': y, 'angle': sumo_yaw}

    # ------------------------------------------------------------------ #
    #  SUMO → CARLA                                                        #
    # ------------------------------------------------------------------ #

    def sumo_to_carla(self, sumo_x: float, sumo_y: float,
                      sumo_angle: float, extent: carla.Vector3D,
                      carla_z: float = 0.5) -> carla.Transform:
        """
        Convert a SUMO vehicle position (front-bumper centre) to a CARLA
        Transform (bounding-box centroid).

        Parameters
        ----------
        sumo_x, sumo_y : float
            SUMO vehicle position (typically from traci.vehicle.getPosition).
        sumo_angle : float
            SUMO heading in degrees (from traci.vehicle.getAngle).
        extent : carla.Vector3D
            Half-dimensions of the vehicle (extent.x = half-length).
        carla_z : float
            Height to use for the spawned actor in CARLA (default 0.5 m).
        """
        # Step 1 — angle
        carla_yaw = _normalize(sumo_angle - 90.0)
        yaw_rad   = math.radians(sumo_angle)

        # Step 2 — remove net offset
        x = sumo_x - self.net_offset[0]
        y = sumo_y - self.net_offset[1]

        # Step 3 — front-bumper → centroid (in SUMO coordinate space)
        fwd_x =  math.sin(yaw_rad)
        fwd_y =  math.cos(yaw_rad)
        x -= fwd_x * extent.x
        y -= fwd_y * extent.x

        # Step 4 — add lateral shift
        if self.lateral_shift != 0.0:
            right_x =  fwd_y
            right_y = -fwd_x
            x += right_x * self.lateral_shift
            y += right_y * self.lateral_shift

        # Step 5 — Y inversion (SUMO right-hand → CARLA left-hand)
        carla_x = x
        carla_y = -y

        return carla.Transform(
            carla.Location(x=carla_x, y=carla_y, z=carla_z),
            carla.Rotation(pitch=0.0, yaw=carla_yaw, roll=0.0),
        )

    # ------------------------------------------------------------------ #
    # CARLA and SUMO use different global origins; read netOffset from the SUMO network.
    # ------------------------------------------------------------------ #

    @staticmethod
    def net_offset_from_cfg(sumocfg_path: str) -> tuple[float, float]:
        """
        Parse <location netOffset="x,y"> from the SUMO .net.xml referenced
        by the given .sumocfg file.  Falls back to (0, 0) on any error.
        """
        import xml.etree.ElementTree as ET
        import os

        try:
            cfg_tree = ET.parse(sumocfg_path)
            net_elem = cfg_tree.getroot().find('.//net-file')
            if net_elem is None:
                return (0.0, 0.0)

            net_rel = net_elem.get('value', '')
            net_path = (
                net_rel if os.path.isabs(net_rel)
                else os.path.join(os.path.dirname(sumocfg_path), net_rel)
            )

            net_tree = ET.parse(net_path)
            loc = net_tree.getroot().find('.//location')
            if loc is not None:
                parts = loc.get('netOffset', '0,0').split(',')
                return (float(parts[0]), float(parts[1]))
        except Exception:
            pass

        return (0.0, 0.0)

    @staticmethod
    def net_offset_from_net(net_path: str) -> tuple[float, float]:
        """
        Parse <location netOffset="x,y"> directly from a .net.xml file.
        Simpler than net_offset_from_cfg — no .sumocfg needed.
        Falls back to (0, 0) on any error.
        """
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(net_path)
            loc  = tree.getroot().find(".//location")
            if loc is not None:
                parts = loc.get("netOffset", "0,0").split(",")
                return (float(parts[0]), float(parts[1]))
        except Exception:
            pass
        return (0.0, 0.0)
