"""Load one isolated final-test scenario without touching train/validation data."""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.scenario_catalog import SettingRecord, materialize_layout
from config.scenario_config import ScenarioConfig, _load_classic, _load_corridor
from validation_debug.loader_v2 import (
    VALIDATION_DEBUG_SPECS_V2 as TEST_SPECS,
    ValidationDebugSpecV2 as TestSpec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCENARIO_ROOT = Path(__file__).resolve().parent / "Scenarios" / "test"


@dataclass(frozen=True)
class TestCase:
    spec: TestSpec
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_test_case(scenario_id: str, *, allow_blocked: bool = False) -> TestCase:
    """Materialize the test WZ1/layout-a case for experiment-facing S1-S6."""
    from Test.Scenarios.test.check_templates import check as check_templates

    preflight_errors = check_templates()
    if preflight_errors:
        raise ValueError(
            "test scenario package failed preflight: "
            + "; ".join(preflight_errors)
        )

    normalized = str(scenario_id).strip().lower()
    try:
        spec = TEST_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown test scenario: {scenario_id!r}") from exc

    scenario_root = TEST_SCENARIO_ROOT / spec.folder
    manifest_path = scenario_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if str(manifest.get("scenario_id", "")).lower() != spec.runtime_scenario_id:
        raise ValueError(f"manifest runtime ID mismatch: {manifest_path}")
    if str(manifest.get("town")) != spec.town:
        raise ValueError(
            f"test town mismatch for {spec.scenario_id}: "
            f"{manifest.get('town')!r} != {spec.town!r}"
        )
    if str(manifest.get("dataset_split", "")).lower() != "test":
        raise ValueError(f"manifest is not a test split: {manifest_path}")

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

    module = importlib.import_module(
        f"Test.Scenarios.test.{spec.folder}.scenario"
    )
    errors = tuple(module.validate_config_data(materialized))
    if errors:
        raise ValueError(
            f"invalid test config {config_path}: "
            + "; ".join(str(error) for error in errors)
        )

    traffic_backend = str(manifest["traffic_backend"])
    network_rel = manifest.get("sumo_network")
    route_rel = work_zone.get("route")
    network_path = scenario_root / str(network_rel) if network_rel else None
    route_path = scenario_root / str(route_rel) if route_rel else None
    if traffic_backend == "carla_sumo":
        if network_path is None or not network_path.is_file():
            raise FileNotFoundError(f"missing test SUMO network: {network_path}")
        if route_path is None or not route_path.is_file():
            raise FileNotFoundError(f"missing test SUMO route: {route_path}")
    elif traffic_backend != "carla_only":
        raise ValueError(f"unsupported traffic backend: {traffic_backend}")

    status = str(manifest.get("status", "blocked")).lower()
    if status != "ready" and not allow_blocked:
        raise RuntimeError(
            f"{spec.scenario_id} test manifest status={status!r}; complete the "
            "live CARLA/SUMO geometry check and set status='ready', or use "
            "--allow-blocked only for that live verification"
        )

    setting_id = f"test/{spec.scenario_id}/wz1/a"
    record = SettingRecord(
        scenario_id=spec.runtime_scenario_id,
        wz_id="wz1",
        layout_id="a",
        setting_id=setting_id,
        town=spec.town,
        status=status,
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
        "dataset_split": "test",
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
            f"{setting_id} requires exactly three test origins, got "
            f"{len(config.origin.spawn_points)}"
        )
    if config.destination.finish_line is None:
        raise ValueError(f"{setting_id} requires a destination finish line")

    return TestCase(
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
    "TEST_SCENARIO_ROOT",
    "TEST_SPECS",
    "TestCase",
    "load_test_case",
]
