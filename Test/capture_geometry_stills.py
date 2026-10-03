"""Capture one non-video audit still for a blocked final-test scenario."""
from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from baseline.PPO.runtime_v2 import configure_initial_config
from baseline.PPO.validate_allSwithoneExe import OrderedOriginSelector, OriginBinder
from Test.record_policy_demos import (
    _FrameSink,
    _OriginTrackingWrapper,
    _destroy_camera,
    _look_at_transform,
)
from Test.run_policy_test import _require_carla
from Test.test_case import PROJECT_ROOT, TEST_SPECS, load_test_case
from validation_debug.visuals_v2 import ValidationWorkZonePropsV2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=tuple(TEST_SPECS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int)
    parser.add_argument("--tm-port", type=int)
    parser.add_argument("--sumo-port", type=int)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "Test" / "geometry_checks" / "stills"),
    )
    return parser


def _xy_points(case: Any) -> list[tuple[float, float]]:
    points = [(float(p.x), float(p.y)) for p in case.config.origin.spawn_points]
    finish = case.config.destination.finish_line
    assert finish is not None
    points.extend((tuple(map(float, finish.start)), tuple(map(float, finish.end))))
    data = case.materialized_data
    workzone = data.get("workzone") or {}
    points.extend(tuple(map(float, p[:2])) for p in workzone.get("warning_signs", []))
    points.extend(tuple(map(float, p[:2])) for p in workzone.get("traffic_cones", []))
    corridor = data.get("corridor") or {}
    points.extend(tuple(map(float, p[:2])) for p in corridor.get("left_boundary_points", []))
    points.extend(tuple(map(float, p[:2])) for p in corridor.get("right_boundary_points", []))
    jaywalker = data.get("jaywalker") or {}
    for key in ("spawn", "collision_target", "disappear", "trigger_anchor"):
        point = jaywalker.get(key)
        if point:
            points.append(tuple(map(float, point[:2])))
    return points


def _ground_location(carla: Any, carla_map: Any, point: tuple[float, float], z: float):
    query = carla.Location(x=point[0], y=point[1], z=1.0)
    waypoint = carla_map.get_waypoint(
        query, project_to_road=True, lane_type=carla.LaneType.Any
    )
    ground_z = float(waypoint.transform.location.z) if waypoint else 0.0
    return carla.Location(x=point[0], y=point[1], z=ground_z + z)


def _draw_polyline(
    carla: Any,
    world: Any,
    points: Iterable[tuple[float, float]],
    color: Any,
    *,
    closed: bool = False,
) -> None:
    values = list(points)
    if closed and values:
        values.append(values[0])
    carla_map = world.get_map()
    for start, end in zip(values, values[1:]):
        world.debug.draw_line(
            _ground_location(carla, carla_map, start, 0.45),
            _ground_location(carla, carla_map, end, 0.45),
            thickness=0.16,
            color=color,
            life_time=12.0,
        )


def _draw_audit_geometry(carla: Any, world: Any, case: Any) -> None:
    carla_map = world.get_map()
    cyan = carla.Color(0, 230, 255)
    green = carla.Color(20, 255, 40)
    red = carla.Color(255, 30, 30)
    magenta = carla.Color(255, 20, 220)
    yellow = carla.Color(255, 220, 0)
    for index, spawn in enumerate(case.config.origin.spawn_points):
        point = (float(spawn.x), float(spawn.y))
        location = _ground_location(carla, carla_map, point, 0.55)
        world.debug.draw_point(location, size=0.35, color=cyan, life_time=12.0)
        world.debug.draw_string(location, f"O{index}", color=cyan, life_time=12.0)
    finish = case.config.destination.finish_line
    assert finish is not None
    _draw_polyline(
        carla,
        world,
        [tuple(map(float, finish.start)), tuple(map(float, finish.end))],
        green,
    )
    data = case.materialized_data
    workzone = data.get("workzone") or {}
    if workzone:
        rectangle = [
            (float(workzone["x_min"]), float(workzone["y_min"])),
            (float(workzone["x_max"]), float(workzone["y_min"])),
            (float(workzone["x_max"]), float(workzone["y_max"])),
            (float(workzone["x_min"]), float(workzone["y_max"])),
        ]
        _draw_polyline(carla, world, rectangle, red, closed=True)
        cones = [tuple(map(float, p[:2])) for p in workzone.get("traffic_cones", [])]
        _draw_polyline(carla, world, cones, yellow)
    corridor = data.get("corridor") or {}
    if corridor:
        _draw_polyline(
            carla,
            world,
            [tuple(map(float, p[:2])) for p in corridor["left_boundary_points"]],
            red,
        )
        _draw_polyline(
            carla,
            world,
            [tuple(map(float, p[:2])) for p in corridor["right_boundary_points"]],
            red,
        )
    jaywalker = data.get("jaywalker") or {}
    if jaywalker:
        _draw_polyline(
            carla,
            world,
            [tuple(map(float, jaywalker["spawn"][:2])), tuple(map(float, jaywalker["disappear"][:2]))],
            magenta,
        )


def main() -> int:
    args = _parser().parse_args()
    case = load_test_case(args.scenario, allow_blocked=True)
    carla_port = int(args.carla_port or case.spec.carla_port)
    tm_port = int(args.tm_port or case.spec.tm_port)
    sumo_port = int(args.sumo_port or case.spec.sumo_port)
    configure_initial_config(
        case.config,
        carla_port=carla_port,
        tm_port=tm_port,
        sumo_port=sumo_port,
        no_rendering=False,
    )
    case.config.carla.host = args.host
    _require_carla(case, args.host, carla_port, 10.0)

    # Import the V2 engine first so it pins the matching SUMO/TraCI tools
    # before the shared legacy module is imported for the S4 controller swap.
    env_type = importlib.import_module("env.gym_wrapper_center_v2").CarlaSumoGymEnv
    legacy_env_module = None
    original_controller = None
    if case.spec.runtime_scenario_id == "s4":
        controller_module = importlib.import_module(
            "Test.Scenarios.test.S4_Town10HD_Jaywalker.jaywalker_controller"
        )
        legacy_env_module = importlib.import_module("env.carla_sumo_env")
        original_controller = legacy_env_module.JaywalkerController
        legacy_env_module.JaywalkerController = controller_module.JaywalkerController

    origins = tuple(case.config.origin.spawn_points)
    selector = OrderedOriginSelector(
        [case.config.setting_id], {case.config.setting_id: len(origins)}, repeats=1
    )
    binder = OriginBinder(selector, {case.config.setting_id: origins})

    def setup(setting_id: str) -> None:
        binder.bind(setting_id)

    base_env = env_type(
        scenario=case.config.setting_id,
        config=case.config,
        mode="eval",
        worker_id=0,
        no_rendering_mode=False,
        scenario_selector=selector,
        episode_setup_callback=setup,
    )
    binder.attach(base_env)
    env = _OriginTrackingWrapper(base_env, selector)
    props = ValidationWorkZonePropsV2(case, enabled=True, cones_only=False)
    camera = None
    try:
        env.reset(seed=1007)
        world = base_env.engine.world
        if world is None:
            raise RuntimeError("CARLA world unavailable after reset")
        props.start(world)
        _draw_audit_geometry(importlib.import_module("carla"), world, case)
        points = _xy_points(case)
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        span = max(max_x - min_x, max_y - min_y, 25.0)
        eye_z = max(35.0, 0.72 * span)
        carla = importlib.import_module("carla")
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(args.width))
        blueprint.set_attribute("image_size_y", str(args.height))
        blueprint.set_attribute("fov", "90")
        blueprint.set_attribute("sensor_tick", "0.100000")
        transform = _look_at_transform(
            carla,
            (center_x, center_y, eye_z),
            (center_x, center_y, 0.0),
        )
        sink = _FrameSink()
        camera = world.spawn_actor(blueprint, transform)
        camera.listen(sink)
        frame = None
        neutral = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(12):
            env.step(neutral)
            frame = sink.next(timeout_s=1.0)
            if frame is not None:
                break
        if frame is None:
            raise RuntimeError("RGB camera produced no frame")
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        output = output_root / f"{args.scenario}_geometry.png"
        image = Image.frombytes(
            "RGBA", (args.width, args.height), frame, "raw", "BGRA"
        ).convert("RGB")
        image.save(output)
        print(output)
    finally:
        _destroy_camera(camera)
        props.close()
        env.close()
        if legacy_env_module is not None:
            legacy_env_module.JaywalkerController = original_controller
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
