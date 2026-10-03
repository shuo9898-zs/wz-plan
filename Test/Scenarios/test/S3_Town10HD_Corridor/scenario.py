"""S3 case logic: an open-ended polygon is the required work-zone corridor."""
from __future__ import annotations

CORE_LOOP = False
LOGIC_PROFILE = "safe_corridor"


def validate_config_data(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("geometry_mode") != "drivable_corridor":
        errors.append("S3 requires top-level geometry_mode=drivable_corridor")

    corridor = data.get("corridor") or {}
    left = corridor.get("left_boundary_points") or []
    right = corridor.get("right_boundary_points") or []
    if len(left) < 2 or len(right) < 2:
        errors.append("S3 requires at least two points on each corridor boundary")
    if corridor.get("geometry_mode") != "safe_corridor":
        errors.append("S3 corridor requires geometry_mode=safe_corridor")
    if corridor.get("corridor_open_ends") is not True:
        errors.append("S3 requires corridor_open_ends=true")

    coordinates = data.get("coordinates") or {}
    if float(coordinates.get("units_per_meter", 0.0)) != 1.0:
        errors.append("S3 corridor coordinates must be stored directly in CARLA metres")
    if not (data.get("origin") or {}).get("spawn_points"):
        errors.append("S3 requires explicit origin.spawn_points")
    line = (data.get("destination") or {}).get("line_cm") or []
    if len(line) != 2:
        errors.append("S3 requires a two-point destination finish line")
    return errors
