"""Canonical six-scenario catalog.

The stable runtime identity is ``scenario/wz/layout``.  A layout changes
CARLA geometry only; every SUMO-backed layout under one WZ shares that WZ's
route file.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_ROOT = PROJECT_ROOT / "scenarios"
SCENARIO_FOLDERS = {
    "s1": "S1_Town02_Normal",
    "s2": "S2_Town05_Curved",
    "s3": "S3_Town10HD_Corridor",
    "s4": "S4_Town10HD_Jaywalker",
    "s5": "S5_Town05_HighTraffic",
    "s6": "S6_Town02_LaneClosure",
}
CORE_SCENARIOS = ("s1", "s2", "s3", "s4", "s5", "s6")
DEFERRED_SCENARIOS: tuple[str, ...] = ()


@dataclass(frozen=True)
class SettingRecord:
    scenario_id: str
    wz_id: str
    layout_id: str
    setting_id: str
    town: str
    status: str
    traffic_backend: str
    geometry_profile: str
    reward_profile: str
    offset_m: float
    config_path: Path | None
    network_path: Path | None
    route_path: Path | None


def scenario_ids() -> list[str]:
    return list(SCENARIO_FOLDERS)


def scenario_root(scenario_id: str) -> Path:
    scenario_id = scenario_id.lower()
    try:
        return SCENARIOS_ROOT / SCENARIO_FOLDERS[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario id: {scenario_id}") from exc


def scenario_module_name(scenario_id: str) -> str:
    return f"scenarios.{SCENARIO_FOLDERS[scenario_id.lower()]}.scenario"


def load_manifest(scenario_id: str) -> dict[str, Any]:
    scenario_id = scenario_id.lower()
    path = scenario_root(scenario_id) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing scenario manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("scenario_id") != scenario_id:
        raise ValueError(f"Manifest id mismatch in {path}")
    return data


def iter_settings(scenario_id: str, *, include_unconfigured: bool = True) -> Iterator[SettingRecord]:
    manifest = load_manifest(scenario_id)
    root = scenario_root(scenario_id)
    network = manifest.get("sumo_network")
    network_path = root / network if network else None
    for wz in manifest.get("work_zones", []):
        route = wz.get("route")
        route_path = root / route if route else None
        for layout in wz.get("layouts", []):
            config_rel = layout.get("config", wz.get("config"))
            config_path = root / config_rel if config_rel else None
            if not include_unconfigured and config_path is None:
                continue
            layout_id = str(layout["id"])
            wz_id = str(wz["id"])
            yield SettingRecord(
                scenario_id=scenario_id,
                wz_id=wz_id,
                layout_id=layout_id,
                setting_id=f"{scenario_id}/{wz_id}/{layout_id}",
                town=str(manifest["town"]),
                status=str(manifest.get("status", "blocked")),
                traffic_backend=str(manifest["traffic_backend"]),
                geometry_profile=str(manifest["geometry_profile"]),
                reward_profile=str(manifest["reward_profile"]),
                offset_m=float(layout.get("offset_m", 0.0)),
                config_path=config_path,
                network_path=network_path,
                route_path=route_path,
            )


def list_setting_ids(scenario_id: str, *, runnable_only: bool = True) -> list[str]:
    manifest = load_manifest(scenario_id)
    if runnable_only and manifest.get("status") not in {"ready", "experimental"}:
        return []
    return [s.setting_id for s in iter_settings(scenario_id, include_unconfigured=not runnable_only)
            if not runnable_only or (s.config_path and s.config_path.is_file())]


def resolve_setting(setting_id: str) -> tuple[SettingRecord, dict[str, Any]]:
    parts = setting_id.replace("\\", "/").lower().split("/")
    if len(parts) == 1 and parts[0] in {"wz1", "wz2", "wz3"}:
        parts = ["s1", parts[0], "b"]
    elif len(parts) == 2 and parts[0] == "s1":
        parts.append("b")
    if len(parts) != 3:
        raise ValueError("Setting id must be scenario/wz/layout, for example s1/wz1/a")
    canonical = "/".join(parts)
    matches = [s for s in iter_settings(parts[0]) if s.setting_id == canonical]
    if not matches:
        raise KeyError(f"Unknown setting: {setting_id}")
    record = matches[0]
    if record.config_path is None or not record.config_path.is_file():
        raise FileNotFoundError(f"Setting {canonical} has no runnable config")

    data = json.loads(record.config_path.read_text(encoding="utf-8-sig"))
    data = materialize_layout(data, record.layout_id)
    _apply_longitudinal_offset(data, record.offset_m)
    data["traffic_backend"] = record.traffic_backend
    data["geometry_profile"] = record.geometry_profile
    data["reward_profile"] = record.reward_profile
    data["_setting"] = {
        "scenario_id": record.scenario_id,
        "wz_id": record.wz_id,
        "layout_id": record.layout_id,
        "setting_id": record.setting_id,
        "offset_m": record.offset_m,
    }
    data.setdefault("carla", {})["town"] = record.town
    if record.traffic_backend == "carla_only":
        data["sumo"] = None
    else:
        if record.network_path is None or record.route_path is None:
            raise ValueError(f"SUMO-backed setting {canonical} lacks network/route binding")
        sumo = data.setdefault("sumo", {})
        sumo["net_file"] = str(record.network_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        sumo["route_file"] = str(record.route_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return record, data


def materialize_layout(data: dict[str, Any], layout_id: str) -> dict[str, Any]:
    """Select an explicit a/b/c geometry embedded in one WZ JSON file."""
    result = copy.deepcopy(data)
    layouts = result.pop("layouts", None)
    if layouts is None:
        return result
    if layout_id not in layouts:
        raise KeyError(f"Config has no explicit layout {layout_id}")
    return _deep_merge(result, layouts[layout_id])


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _apply_longitudinal_offset(data: dict[str, Any], offset_m: float) -> None:
    if abs(offset_m) < 1e-9:
        return
    heading = float(data.get("carla", {}).get("road_heading_deg", 0.0))
    radians = math.radians(heading)
    dx = math.cos(radians) * offset_m
    dy = math.sin(radians) * offset_m
    wz = data.get("workzone")
    if wz:
        for key in ("x_min", "x_max"):
            if key in wz:
                wz[key] = float(wz[key]) + dx
        for key in ("y_min", "y_max"):
            if key in wz:
                wz[key] = float(wz[key]) + dy
        for key in ("warning_signs", "traffic_cones", "corridor_boundary_points"):
            if wz.get(key):
                wz[key] = [[float(p[0]) + dx, float(p[1]) + dy] for p in wz[key]]
        polygon = wz.get("polygon_config")
        if polygon:
            for key in ("head_x", "tail_x"):
                polygon[key] = float(polygon[key]) + dx
            for key in ("head_y", "tail_y"):
                polygon[key] = float(polygon[key]) + dy
    jaywalker = data.get("jaywalker")
    if jaywalker:
        for key in ("spawn", "collision_target", "disappear", "trigger_anchor"):
            if key in jaywalker:
                point = jaywalker[key]
                point[0] = float(point[0]) + dx
                point[1] = float(point[1]) + dy
