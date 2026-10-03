"""S4 case logic: CARLA-only Ego and three staged jaywalker crossings."""
from __future__ import annotations

import math

CORE_LOOP = True
LOGIC_PROFILE = "carla_only_jaywalker"


def validate_config_data(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("sumo") is not None:
        errors.append("S4 must not bind SUMO")
    jaywalker = data.get("jaywalker")
    if not jaywalker:
        errors.append("S4 requires jaywalker parameters")
        return errors

    origin = data.get("origin", {})
    if not origin.get("spawn_points"):
        errors.append("S4 requires manual origin.spawn_points")
    destination = data.get("destination", {})
    line = destination.get("line_cm")
    if not isinstance(line, list) or len(line) != 2 or line[0] == line[1]:
        errors.append("S4 requires a finite manual destination.line_cm")

    for key in ("spawn", "collision_target", "disappear", "trigger_anchor"):
        point = jaywalker.get(key)
        if not isinstance(point, list) or len(point) != 3:
            errors.append(f"S4 jaywalker requires three-value {key}")
        elif not all(math.isfinite(float(value)) for value in point):
            errors.append(f"S4 jaywalker {key} must be finite")
    if jaywalker.get("spawn") == jaywalker.get("disappear"):
        errors.append("S4 jaywalker spawn and disappear must differ")
    gates = jaywalker.get("trigger_distances_m")
    if gates != [45.0, 30.0, 15.0]:
        errors.append("S4 jaywalker trigger_distances_m must be [45, 30, 15]")
    if float(jaywalker.get("speed_mps", 0.0)) != 2.0:
        errors.append("S4 jaywalker speed_mps must be 2.0")
    if float(jaywalker.get("disappear_radius_m", 0.0)) <= 0.0:
        errors.append("S4 jaywalker disappear_radius_m must be positive")
    if not jaywalker.get("blueprint"):
        errors.append("S4 jaywalker blueprint is required")

    # Validation may deliberately start inside the outer 45 m gate.  The
    # shared controller's start_episode() queues every already-crossed gate,
    # so the event is not lost; each origin must only remain upstream of the
    # trigger anchor.  Work-zone/walker geometry is in metres while manual
    # origins remain in Unreal centimetres.
    spawn_points = origin.get("spawn_points", [])
    anchor = jaywalker.get("trigger_anchor")
    if spawn_points and isinstance(anchor, list) and len(anchor) == 3 and gates:
        heading = math.radians(float(data.get("carla", {}).get("road_heading_deg", 0.0)))
        forward_x, forward_y = math.cos(heading), math.sin(heading)
        for index, point in enumerate(spawn_points):
            location_cm = point.get("location_cm") if isinstance(point, dict) else None
            if not isinstance(location_cm, list) or len(location_cm) != 2:
                errors.append(f"S4 origin.spawn_points[{index}] requires location_cm [x, y]")
                continue
            ego_x = float(location_cm[0]) / 100.0
            ego_y = float(location_cm[1]) / 100.0
            remaining = (
                (float(anchor[0]) - ego_x) * forward_x
                + (float(anchor[1]) - ego_y) * forward_y
            )
            if remaining <= 0.0:
                errors.append(
                    f"S4 origin.spawn_points[{index}] must remain upstream of "
                    f"the trigger anchor (remaining={remaining:.2f} m)"
                )

    if isinstance(line, list) and len(line) == 2 and anchor:
        heading = math.radians(float(data.get("carla", {}).get("road_heading_deg", 0.0)))
        forward_x, forward_y = math.cos(heading), math.sin(heading)
        finish_x = (float(line[0][0]) + float(line[1][0])) / 200.0
        finish_y = (float(line[0][1]) + float(line[1][1])) / 200.0
        downstream = (
            (finish_x - float(anchor[0])) * forward_x
            + (finish_y - float(anchor[1])) * forward_y
        )
        if downstream <= 0.0:
            errors.append("S4 destination line must be downstream of its trigger anchor")
    return errors
