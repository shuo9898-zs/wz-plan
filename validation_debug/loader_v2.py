"""Load one owner-authored validation case without touching the train catalog.

The normal catalog deliberately points only at ``workspace/scenarios``.  This
module mirrors its materialisation contract for the isolated
``workspace/validation`` root, while keeping the stable internal scenario IDs
needed by the environment (``s1`` ... ``s6``).
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.scenario_catalog import SettingRecord, materialize_layout
from config.scenario_config import ScenarioConfig, _load_classic, _load_corridor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = PROJECT_ROOT / "validation"


@dataclass(frozen=True)
class ValidationDebugSpecV2:
    # scenario_id is the experiment-facing validation identity.  The runtime
    # identity remains tied to the legacy folder/manifest so existing env
    # scenario-specific logic stays stable.
    scenario_id: str
    runtime_scenario_id: str
    folder: str
    town: str
    carla_port: int
    tm_port: int
    sumo_port: int
    display_name: str


VALIDATION_DEBUG_SPECS_V2: dict[str, ValidationDebugSpecV2] = {
    "s1": ValidationDebugSpecV2(
        "s1", "s5", "S5_Town05_HighTraffic", "Town05", 2020, 8020, 8833,
        "S1 Normal Traffic",
    ),
    "s2": ValidationDebugSpecV2(
        "s2", "s2", "S2_Town05_Curved", "Town05", 2020, 8020, 8833,
        "S2 Curved Work Zone",
    ),
    "s3": ValidationDebugSpecV2(
        "s3", "s3", "S3_Town10HD_Corridor", "Town10HD", 2040, 8040, 8853,
        "S3 Safe Corridor",
    ),
    "s4": ValidationDebugSpecV2(
        "s4", "s4", "S4_Town10HD_Jaywalker", "Town10HD", 2040, 8040, 8853,
        "S4 Jaywalker",
    ),
    "s5": ValidationDebugSpecV2(
        "s5", "s1", "S1_Town02_Normal", "Town02", 2000, 8000, 8813,
        "S5 High Traffic",
    ),
    "s6": ValidationDebugSpecV2(
        "s6", "s6", "S6_Town02_LaneClosure", "Town02", 2000, 8000, 8813,
        "S6 Oncoming Platoon",
    ),
}


@dataclass(frozen=True)
class ValidationCaseV2:
    spec: ValidationDebugSpecV2
    config: ScenarioConfig
    materialized_data: dict[str, Any]
    manifest: dict[str, Any]
    scenario_root: Path
    manifest_path: Path
    config_path: Path
    network_path: Path | None
    route_path: Path | None

    @property
    def status(self) -> str:
        return str(self.manifest.get("status", "blocked"))


def validation_debug_spec_v2(scenario_id: str) -> ValidationDebugSpecV2:
    normalized = str(scenario_id).strip().lower()
    try:
        return VALIDATION_DEBUG_SPECS_V2[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown validation scenario: {scenario_id!r}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_validation_case_v2(scenario_id: str) -> ValidationCaseV2:
    """Materialize the single WZ1/layout-a/origin validation case.

    This function intentionally never calls ``resolve_setting`` or
    ``load_scenario``.  Those functions are train-root loaders.
    """
    spec = validation_debug_spec_v2(scenario_id)
    scenario_root = VALIDATION_ROOT / spec.folder
    manifest_path = scenario_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if str(manifest.get("scenario_id", "")).lower() != spec.runtime_scenario_id:
        raise ValueError(f"manifest ID mismatch: {manifest_path}")
    if str(manifest.get("town")) != spec.town:
        raise ValueError(
            f"validation town mismatch for {spec.scenario_id}: "
            f"{manifest.get('town')!r} != {spec.town!r}"
        )

    work_zones = manifest.get("work_zones") or []
    if len(work_zones) != 1 or str(work_zones[0].get("id")) != "wz1":
        raise ValueError(f"{manifest_path} must expose only wz1")
    work_zone = work_zones[0]
    layouts = work_zone.get("layouts") or []
    if layouts != [{"id": "a"}]:
        raise ValueError(f"{manifest_path} must expose only layout a")

    config_rel = layouts[0].get("config", work_zone.get("config"))
    if not config_rel:
        raise ValueError(f"{manifest_path} does not bind a config")
    config_path = scenario_root / str(config_rel)
    materialized = materialize_layout(_read_json(config_path), "a")

    module = importlib.import_module(f"validation.{spec.folder}.scenario")
    errors = tuple(module.validate_config_data(materialized))
    if errors:
        joined = "; ".join(str(error) for error in errors)
        raise ValueError(f"invalid validation config {config_path}: {joined}")

    traffic_backend = str(manifest["traffic_backend"])
    network_rel = manifest.get("sumo_network")
    route_rel = work_zone.get("route")
    network_path = scenario_root / str(network_rel) if network_rel else None
    route_path = scenario_root / str(route_rel) if route_rel else None
    if traffic_backend == "carla_sumo":
        if network_path is None or not network_path.is_file():
            raise FileNotFoundError(
                f"missing validation SUMO network for {spec.scenario_id}: "
                f"{network_path}"
            )
        if route_path is None or not route_path.is_file():
            raise FileNotFoundError(
                f"missing validation SUMO route for {spec.scenario_id}: "
                f"{route_path}"
            )
    elif traffic_backend != "carla_only":
        raise ValueError(f"unsupported traffic backend: {traffic_backend}")

    setting_id = f"validation/{spec.scenario_id}/wz1/a"
    record = SettingRecord(
        scenario_id=spec.runtime_scenario_id,
        wz_id="wz1",
        layout_id="a",
        setting_id=setting_id,
        town=spec.town,
        status=str(manifest.get("status", "blocked")),
        traffic_backend=traffic_backend,
        geometry_profile=str(manifest["geometry_profile"]),
        reward_profile=str(manifest["reward_profile"]),
        offset_m=0.0,
        config_path=config_path,
        network_path=network_path,
        route_path=route_path,
    )

    materialized["traffic_backend"] = record.traffic_backend
    materialized["geometry_profile"] = record.geometry_profile
    materialized["reward_profile"] = record.reward_profile
    materialized["_setting"] = {
        "scenario_id": record.scenario_id,
        "wz_id": record.wz_id,
        "layout_id": record.layout_id,
        "setting_id": record.setting_id,
        "offset_m": record.offset_m,
        "dataset_split": "validation",
    }
    materialized.setdefault("carla", {})["town"] = record.town
    if traffic_backend == "carla_only":
        materialized["sumo"] = None
    else:
        assert network_path is not None and route_path is not None
        sumo = materialized.setdefault("sumo", {})
        sumo["net_file"] = str(network_path.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        )
        sumo["route_file"] = str(route_path.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        )

    config = (
        _load_corridor(record, materialized)
        if "corridor" in materialized
        else _load_classic(record, materialized)
    )
    if len(config.origin.spawn_points) != 3:
        raise ValueError(
            f"{setting_id} requires exactly three validation origins, got "
            f"{len(config.origin.spawn_points)}"
        )
    if config.destination.finish_line is None:
        raise ValueError(f"{setting_id} requires a destination line")

    return ValidationCaseV2(
        spec=spec,
        config=config,
        materialized_data=materialized,
        manifest=manifest,
        scenario_root=scenario_root,
        manifest_path=manifest_path,
        config_path=config_path,
        network_path=network_path,
        route_path=route_path,
    )


__all__ = [
    "PROJECT_ROOT",
    "VALIDATION_DEBUG_SPECS_V2",
    "VALIDATION_ROOT",
    "ValidationCaseV2",
    "ValidationDebugSpecV2",
    "load_validation_case_v2",
    "validation_debug_spec_v2",
]
