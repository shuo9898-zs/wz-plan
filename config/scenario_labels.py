"""Experiment-facing scenario labels without changing runtime data bindings."""
from __future__ import annotations


_TRAINING_DISPLAY_ID = {
    "s1": "s5",
    "s5": "s1",
}

SCENARIO_DISPLAY_NAMES = {
    "s1": "Normal Traffic",
    "s2": "Curved Work Zone",
    "s3": "Safe Corridor",
    "s4": "Jaywalker",
    "s5": "High Traffic",
    "s6": "Oncoming Platoon",
}


def training_display_scenario_id(runtime_scenario_id: str) -> str:
    """Map stable training IDs to the experiment-facing S1/S5 numbering."""
    normalized = str(runtime_scenario_id).strip().lower()
    return _TRAINING_DISPLAY_ID.get(normalized, normalized)


def training_display_setting_id(runtime_setting_id: str) -> str:
    """Swap only the scenario prefix of ``scenario/wz/layout``."""
    value = str(runtime_setting_id)
    pieces = value.split("/", 1)
    display = training_display_scenario_id(pieces[0])
    return display if len(pieces) == 1 else f"{display}/{pieces[1]}"


__all__ = [
    "SCENARIO_DISPLAY_NAMES",
    "training_display_scenario_id",
    "training_display_setting_id",
]
