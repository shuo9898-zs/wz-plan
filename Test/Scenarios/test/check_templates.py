"""Validate the isolated one-layout/final-test templates."""
from __future__ import annotations

import importlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


sys.dont_write_bytecode = True

try:
    from config.scenario_config import SUMOConfig
except ModuleNotFoundError:
    from workspace.config.scenario_config import SUMOConfig


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = __package__ or "test"
SCENARIOS = {
    "s1": ("S1_Town02_Normal", True),
    "s2": ("S2_Town05_Curved", True),
    "s3": ("S3_Town10HD_Corridor", True),
    "s4": ("S4_Town10HD_Jaywalker", False),
    "s5": ("S5_Town05_HighTraffic", True),
    "s6": ("S6_Town02_LaneClosure", True),
}


def _merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _route_data(path: Path) -> tuple[dict[str, list[str]], set[str]]:
    root = ET.parse(path).getroot()
    routes = {
        node.attrib["id"]: node.attrib.get("edges", "").split()
        for node in root.findall("route")
        if "id" in node.attrib
    }
    vtype_ids = {node.attrib["id"] for node in root.findall("vType") if "id" in node.attrib}
    return routes, vtype_ids


def _network_data(path: Path) -> tuple[set[str], set[tuple[str, str]], dict[str, float]]:
    root = ET.parse(path).getroot()
    edge_ids: set[str] = set()
    edge_lengths: dict[str, float] = {}
    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue
        edge_ids.add(edge_id)
        lengths = [float(lane.attrib["length"]) for lane in edge.findall("lane") if "length" in lane.attrib]
        if lengths:
            edge_lengths[edge_id] = max(lengths)
    connections = {
        (node.attrib["from"], node.attrib["to"])
        for node in root.findall("connection")
        if "from" in node.attrib and "to" in node.attrib
    }
    return edge_ids, connections, edge_lengths


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def check() -> list[str]:
    errors: list[str] = []
    for scenario_id, (folder, uses_sumo) in SCENARIOS.items():
        scenario_root = ROOT / folder
        manifest_path = scenario_root / "manifest.json"
        config_path = scenario_root / "configs" / "wz1.json"
        prefix = f"{scenario_id}:"

        _require(manifest_path.is_file(), f"{prefix} missing manifest.json", errors)
        _require(config_path.is_file(), f"{prefix} missing configs/wz1.json", errors)
        _require((scenario_root / "scenario.py").is_file(), f"{prefix} missing scenario.py", errors)
        if not manifest_path.is_file() or not config_path.is_file():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        work_zones = manifest.get("work_zones") or []
        _require(manifest.get("scenario_id") == scenario_id, f"{prefix} manifest id mismatch", errors)
        _require(manifest.get("dataset_split") == "test", f"{prefix} split is not test", errors)
        _require(manifest.get("status") in {"blocked", "ready"}, f"{prefix} invalid status", errors)
        _require(len(work_zones) == 1, f"{prefix} manifest must expose exactly one WZ", errors)
        if len(work_zones) != 1:
            continue
        work_zone = work_zones[0]
        layouts = work_zone.get("layouts") or []
        _require(work_zone.get("id") == "wz1", f"{prefix} only wz1 is reserved", errors)
        _require(layouts == [{"id": "a"}], f"{prefix} manifest must expose only layout a", errors)
        _require(set((config.get("layouts") or {}).keys()) == {"a"}, f"{prefix} config must contain only layout a", errors)
        if "a" not in (config.get("layouts") or {}):
            continue

        base = {key: value for key, value in config.items() if key != "layouts"}
        materialized = _merge(base, config["layouts"]["a"])
        source = materialized.get("source") or {}
        _require(source.get("dataset_split") == "test", f"{prefix} config source split is not test", errors)
        origins = (materialized.get("origin") or {}).get("spawn_points") or []
        finish = (materialized.get("destination") or {}).get("line_cm") or []
        _require(len(origins) == 3, f"{prefix} requires exactly three origins", errors)
        _require(len(finish) == 2 and finish[0] != finish[1], f"{prefix} requires one finite two-point finish line", errors)

        module = importlib.import_module(f"{PACKAGE_ROOT}.{folder}.scenario")
        for message in module.validate_config_data(materialized):
            errors.append(f"{prefix} scenario validator: {message}")

        sumo = materialized.get("sumo")
        if uses_sumo:
            network_path = scenario_root / str(manifest.get("sumo_network", ""))
            route_path = scenario_root / str(work_zone.get("route", ""))
            _require(isinstance(sumo, dict), f"{prefix} SUMO config is required", errors)
            _require(network_path.is_file(), f"{prefix} missing SUMO network XML", errors)
            _require(route_path.is_file(), f"{prefix} missing SUMO route XML", errors)
            if isinstance(sumo, dict) and route_path.is_file():
                routes, vtype_ids = _route_data(route_path)
                _require(set(sumo.get("bg_routes") or []).issubset(routes), f"{prefix} bg_routes missing from route XML", errors)
                _require("ego_placeholder" in routes, f"{prefix} route XML lacks ego_placeholder", errors)
                _require(str(sumo.get("bg_vtype")) in vtype_ids, f"{prefix} bg_vtype missing from route XML", errors)
                episode_dt = float((materialized.get("episode") or {}).get("sim_dt", -1.0))
                _require(abs(float(sumo.get("step_length", -2.0)) - episode_dt) < 1e-9, f"{prefix} SUMO/CARLA step mismatch", errors)
                try:
                    SUMOConfig(
                        net_file=str(network_path),
                        route_file=str(route_path),
                        **sumo,
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(f"{prefix} invalid SUMOConfig: {exc}")

                if network_path.is_file():
                    edge_ids, connections, edge_lengths = _network_data(network_path)
                    for route_id in ["ego_placeholder", *(sumo.get("bg_routes") or [])]:
                        route_edges = routes.get(route_id) or []
                        _require(bool(route_edges), f"{prefix} route {route_id} has no edges", errors)
                        _require(set(route_edges).issubset(edge_ids), f"{prefix} route {route_id} contains unknown edges", errors)
                        for first, second in zip(route_edges, route_edges[1:]):
                            _require((first, second) in connections, f"{prefix} route {route_id} is disconnected at {first}->{second}", errors)
                    for route_id, bounds in (sumo.get("bg_route_depart_pos_ranges_m") or {}).items():
                        route_edges = routes.get(route_id) or []
                        if route_edges:
                            first_length = edge_lengths.get(route_edges[0], -1.0)
                            _require(float(bounds[1]) <= first_length + 1e-9, f"{prefix} departPos for {route_id} exceeds first edge length", errors)
        else:
            _require(manifest.get("sumo_network") is None, f"{prefix} CARLA-only manifest must not bind SUMO", errors)
            _require(work_zone.get("route") is None, f"{prefix} CARLA-only WZ must not bind a route", errors)
            _require(sumo is None, f"{prefix} CARLA-only config must set sumo=null", errors)
            _require(not (scenario_root / "sumo").exists(), f"{prefix} CARLA-only template must not contain a sumo directory", errors)
            _require((scenario_root / "jaywalker_controller.py").is_file(), f"{prefix} missing jaywalker controller", errors)

        if scenario_id == "s2":
            workzone = materialized.get("workzone") or {}
            _require(workzone.get("geometry_mode") == "forbidden_polygon", "s2: curved polygon interface lost", errors)
            _require(bool(workzone.get("polygon_config")), "s2: polygon_config interface lost", errors)
            _require(bool(sumo.get("bg_route_traffic_overrides")), "s2: route traffic override lost", errors)
            _require(bool(sumo.get("bg_route_depart_pos_ranges_m")), "s2: route depart-position interface lost", errors)
        elif scenario_id == "s3":
            corridor = materialized.get("corridor") or {}
            _require(materialized.get("geometry_mode") == "drivable_corridor", "s3: top-level corridor mode lost", errors)
            _require(corridor.get("geometry_mode") == "safe_corridor", "s3: safe-corridor interface lost", errors)
            _require(corridor.get("corridor_open_ends") is True, "s3: open-end interface lost", errors)
            _require(bool(sumo.get("bg_route_traffic_overrides")), "s3: route traffic override lost", errors)
        elif scenario_id == "s4":
            _require(bool(materialized.get("jaywalker")), "s4: jaywalker interface lost", errors)
        elif scenario_id == "s6":
            _require(sumo.get("traffic_pattern") == "platoon", "s6: platoon interface lost", errors)
            for key in (
                "platoon_size_min", "platoon_size_max",
                "platoon_headway_min_s", "platoon_headway_max_s",
                "platoon_gap_min_s", "platoon_gap_max_s",
            ):
                _require(key in sumo, f"s6: missing {key}", errors)
    return errors


def main() -> int:
    errors = check()
    if errors:
        print("Test template check FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Test template check passed: 6 scenarios, 1 WZ/layout and 3 origins each.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
