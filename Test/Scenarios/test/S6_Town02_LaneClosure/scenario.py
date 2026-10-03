"""S6 case logic: standard straight-road forbidden work zone."""
from __future__ import annotations

CORE_LOOP = True
LOGIC_PROFILE = "straight_forbidden"


def validate_config_data(data: dict) -> list[str]:
    mode = data.get("workzone", {}).get("geometry_mode", "forbidden_rect")
    return [] if mode == "forbidden_rect" else [f"S6 requires forbidden_rect, got {mode}"]
