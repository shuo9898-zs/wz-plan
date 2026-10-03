"""Independent preflight for one canonical scenario package."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from config.scenario_catalog import (
    iter_settings,
    load_manifest,
    materialize_layout,
    scenario_module_name,
)
from config.scenario_config import load_scenario


@dataclass
class Result:
    scenario_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_scenario(scenario_id: str) -> Result:
    manifest = load_manifest(scenario_id)
    result = Result(scenario_id)
    settings = list(iter_settings(scenario_id))
    work_zones = manifest.get("work_zones", [])
    backend = manifest.get("traffic_backend")
    status = manifest.get("status")
    result.checks.append(f"status={status}")
    result.checks.append(f"work_zones={len(work_zones)} settings={len(settings)}")
    if status in {"blocked", "deferred"}:
        result.errors.append(f"scenario status is {status}: {manifest.get('blockers', [])}")
    try:
        case_module = importlib.import_module(scenario_module_name(scenario_id))
        case_validator = case_module.validate_config_data
        result.checks.append(f"logic_profile={case_module.LOGIC_PROFILE}")
    except (ImportError, AttributeError) as exc:
        result.errors.append(f"missing scenario.py case logic: {exc}")
        case_validator = lambda data: []

    network_paths = {s.network_path for s in settings if s.network_path is not None}
    route_paths = {s.route_path for s in settings if s.route_path is not None}
    if backend == "carla_only":
        if network_paths or route_paths or manifest.get("sumo_network"):
            result.errors.append("CARLA-only scenario must have zero SUMO bindings")
    else:
        if len(network_paths) != 1:
            result.errors.append(f"expected exactly one SUMO network, found {len(network_paths)}")
        if len(route_paths) != len(work_zones):
            result.errors.append(
                f"expected one route file per WZ ({len(work_zones)}), found {len(route_paths)}"
            )

    for path in sorted(network_paths | route_paths, key=str):
        if not path.is_file():
            result.errors.append(f"missing file: {path}")
            continue
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            result.errors.append(f"invalid XML {path}: {exc}")

    configured_count = 0
    for setting in settings:
        if setting.config_path is None or not setting.config_path.is_file():
            result.errors.append(f"{setting.setting_id}: missing config")
            continue
        configured_count += 1
        try:
            raw = json.loads(setting.config_path.read_text(encoding="utf-8-sig"))
            raw = materialize_layout(raw, setting.layout_id)
            _check_raw_geometry(setting.setting_id, raw, result)
            for error in case_validator(raw):
                result.errors.append(f"{setting.setting_id}: {error}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            result.errors.append(f"{setting.setting_id}: invalid raw config: {exc}")
            continue
        # Geometry can be audited even while a SUMO route binding is still
        # missing. The route-count error above is the authoritative blocker.
        if backend == "carla_sumo" and (setting.network_path is None or setting.route_path is None):
            continue
        try:
            cfg = load_scenario(setting.setting_id)
        except Exception as exc:
            result.errors.append(f"{setting.setting_id}: config load failed: {exc}")
            continue
        if cfg.carla.town != manifest["town"]:
            result.errors.append(f"{setting.setting_id}: town mismatch")
        if backend == "carla_only" and (cfg.sumo is not None or cfg.jaywalker is None):
            result.errors.append(f"{setting.setting_id}: S4 requires jaywalker and no SUMO")
        if backend == "carla_sumo" and cfg.sumo is None:
            result.errors.append(f"{setting.setting_id}: SUMO config missing")
    result.checks.append(f"configured_settings={configured_count}/{len(settings)}")

    if backend == "carla_sumo" and len(network_paths) == 1:
        network = next(iter(network_paths))
        if network.is_file():
            for route in route_paths:
                if route.is_file():
                    _check_routes(network, route, result)
    result.warnings.extend(str(w) for w in manifest.get("warnings", []))
    return result


def _check_raw_geometry(setting_id: str, data: dict, result: Result) -> None:
    if "corridor" in data:
        coordinates = data.get("coordinates", {})
        scale = float(coordinates.get("units_per_meter", 100.0))
        corridor = data["corridor"]
        if corridor.get("left_boundary_points") and corridor.get("right_boundary_points"):
            points = corridor["left_boundary_points"] + corridor["right_boundary_points"]
            xs = [float(p[0]) / scale for p in points]
            ys = [float(p[1]) / scale for p in points]
            values = [min(xs), max(xs), min(ys), max(ys)]
        else:
            values = [
                float(corridor[key]) / scale
                for key in ("x_min", "x_max", "y_min", "y_max")
            ]
    else:
        wz = data["workzone"]
        values = [float(wz[key]) for key in ("x_min", "x_max", "y_min", "y_max")]
    if not all(math.isfinite(value) for value in values):
        result.errors.append(f"{setting_id}: non-finite geometry")
    if not (values[0] < values[1] and values[2] < values[3]):
        result.errors.append(f"{setting_id}: unordered work-zone bounds")
    if max(abs(value) for value in values) > 1000.0:
        result.errors.append(f"{setting_id}: likely centimetres loaded as metres")


def _check_routes(network: Path, routes: Path, result: Result) -> None:
    net_root = ET.parse(network).getroot()
    route_root = ET.parse(routes).getroot()
    edges = {e.attrib["id"] for e in net_root.findall("edge") if not e.attrib["id"].startswith(":")}
    connections = {
        (c.attrib.get("from"), c.attrib.get("to"))
        for c in net_root.findall("connection")
        if c.attrib.get("from") and c.attrib.get("to")
    }
    for route in route_root.findall("route"):
        route_id = route.attrib.get("id", "(anonymous)")
        sequence = route.attrib.get("edges", "").split()
        missing = [edge for edge in sequence if edge not in edges]
        disconnected = [f"{a}->{b}" for a, b in zip(sequence, sequence[1:]) if (a, b) not in connections]
        if missing or disconnected:
            result.errors.append(
                f"{routes.name}:{route_id}: missing={missing} disconnected={disconnected}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=[f"s{i}" for i in range(1, 7)])
    args = parser.parse_args(argv)
    result = validate_scenario(args.scenario)
    print(f"scenario={result.scenario_id} ok={result.ok}")
    for line in result.checks:
        print(f"CHECK {line}")
    for line in result.warnings:
        print(f"WARN  {line}")
    for line in result.errors:
        print(f"ERROR {line}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
