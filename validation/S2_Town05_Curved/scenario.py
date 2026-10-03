"""S2 case logic: curved forbidden polygon with direction-specific routes."""
from __future__ import annotations

CORE_LOOP = True
LOGIC_PROFILE = "curved_forbidden"


def validate_config_data(data: dict) -> list[str]:
    workzone = data.get("workzone", {})
    errors: list[str] = []
    if workzone.get("geometry_mode") != "forbidden_polygon":
        errors.append("S2 requires forbidden_polygon geometry")
    if not workzone.get("polygon_config"):
        errors.append("S2 requires polygon_config")
    return errors
