"""Visualize one setting's manual work-zone props in CARLA.

The reusable ``WorkZonePropVisualizer`` is used by the standalone
``validate_one.py`` demo and can also be run directly. It never participates
in the normal training path.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import carla

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenario_catalog import resolve_setting
from config.scenario_config import load_scenario


CONE_BLUEPRINT_CANDIDATES = (
    "static.prop.constructioncone",
)
SIGN_BLUEPRINT_CANDIDATES = (
    "static.prop.warningconstruction",
    "static.prop.warningother",
    "static.prop.warningaccident",
)


def _asset_key(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _visual_prop_blueprint_ids(asset_hint: str) -> tuple[str, ...]:
    """Translate owner/Unreal asset names to CARLA's spawnable blueprint IDs."""
    key = _asset_key(asset_hint)
    if "warningconstruction" in key:
        return ("static.prop.warningconstruction",)
    if "trafficcones03" in key or "trafficcone03" in key:
        # Default.Package.json exposes SM_TrafficCones_03 under this ID.
        return ("static.prop.trafficwarning",)
    if "constructioncone" in key:
        return ("static.prop.constructioncone",)
    return ()


def _visual_debug_props(setting_id: str) -> tuple[dict, ...]:
    """Return materialized, layout-specific visual props without affecting training."""
    _, data = resolve_setting(setting_id)
    visual_debug = data.get("visual_debug") or {}
    return tuple(dict(prop) for prop in visual_debug.get("props", ()))


def _canonical_town(name: str) -> str:
    normalized = name.strip().lower()
    return normalized[:-4] if normalized.endswith("_opt") else normalized


def _canonical_wz(value: str) -> str:
    normalized = value.strip().lower()
    return f"wz{normalized}" if normalized.isdigit() else normalized


def _choose_blueprint(library, exact_ids: tuple[str, ...], filters: tuple[str, ...]):
    for blueprint_id in exact_ids:
        try:
            return library.find(blueprint_id)
        except (IndexError, RuntimeError):
            pass
    for pattern in filters:
        matches = list(library.filter(pattern))
        if matches:
            return matches[0]
    return None


def _surface_transform(
    carla_map,
    x: float,
    y: float,
    yaw: float,
    z_offset: float,
    *,
    pitch: float = 0.0,
    roll: float = 0.0,
):
    query = carla.Location(x=float(x), y=float(y), z=1.0)
    try:
        waypoint = carla_map.get_waypoint(
            query, project_to_road=True, lane_type=carla.LaneType.Any
        )
    except TypeError:
        waypoint = carla_map.get_waypoint(query, project_to_road=True)
    ground_z = waypoint.transform.location.z if waypoint is not None else 0.0
    return carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(ground_z + z_offset)),
        carla.Rotation(pitch=float(pitch), yaw=float(yaw), roll=float(roll)),
    )


class WorkZonePropVisualizer:
    """Own and clean the debug props for one canonical setting."""

    def __init__(
        self,
        setting_id: str,
        *,
        host: str = "localhost",
        carla_port: int = 2000,
        timeout: float = 10.0,
        z_offset: float = 0.08,
        sign_yaw_offset: float = 180.0,
        cones_only: bool = False,
        draw_finish_line: bool = True,
        load_town: bool = True,
        move_spectator: bool = True,
        spectator_height: float = 150.0,
    ) -> None:
        self.setting_id = setting_id
        self.host = host
        self.carla_port = int(carla_port)
        self.timeout = float(timeout)
        self.z_offset = float(z_offset)
        self.sign_yaw_offset = float(sign_yaw_offset)
        self.cones_only = bool(cones_only)
        self.draw_finish_line = bool(draw_finish_line)
        self.load_town = bool(load_town)
        self.move_spectator = bool(move_spectator)
        self.spectator_height = float(spectator_height)
        self.actors: list[carla.Actor] = []
        self.world = None

    def switch(self, setting_id: str) -> None:
        """Replace visible props when the next episode selects a new layout."""
        if setting_id == self.setting_id and self.actors:
            return
        self.close()
        self.setting_id = setting_id
        self.start()

    def start(self) -> "WorkZonePropVisualizer":
        cfg = load_scenario(self.setting_id)
        client = carla.Client(self.host, self.carla_port)
        client.set_timeout(self.timeout)
        world = client.get_world()
        current_town = world.get_map().name.split("/")[-1]
        if _canonical_town(current_town) != _canonical_town(cfg.carla.town):
            if not self.load_town:
                raise RuntimeError(
                    f"CARLA has {current_town}, but {self.setting_id} requires {cfg.carla.town}"
                )
            print(f"Loading {cfg.carla.town} (current map: {current_town}) ...")
            world = client.load_world(cfg.carla.town)
        self.world = world

        library = world.get_blueprint_library()
        cone_bp = _choose_blueprint(
            library, CONE_BLUEPRINT_CANDIDATES, ("*constructioncone*",)
        )
        visual_props = _visual_debug_props(self.setting_id)
        sign_bp = None
        if not visual_props:
            sign_bp = _choose_blueprint(
                library, SIGN_BLUEPRINT_CANDIDATES, ("*warning*", "*construction*")
            )
        if cone_bp is None:
            raise RuntimeError("No CARLA traffic-cone blueprint was found")
        print("visual_setting", self.setting_id)
        print("cone_blueprint", cone_bp.id)
        if not self.cones_only and not visual_props:
            print("sign_blueprint", sign_bp.id if sign_bp else "NOT FOUND")

        carla_map = world.get_map()
        try:
            for index, (x, y) in enumerate(cfg.workzone.traffic_cones, start=1):
                transform = _surface_transform(
                    carla_map, x, y, cfg.carla.road_heading_deg, self.z_offset
                )
                self._spawn(cone_bp, transform, f"cone {index}")

            if not self.cones_only and visual_props:
                for index, prop in enumerate(visual_props, start=1):
                    asset_hint = str(prop.get("asset_hint_supplied", ""))
                    blueprint_ids = _visual_prop_blueprint_ids(asset_hint)
                    prop_bp = _choose_blueprint(library, blueprint_ids, ())
                    if prop_bp is None:
                        print(
                            f"WARN no CARLA blueprint mapping for visual prop {index}: "
                            f"{asset_hint or 'MISSING ASSET HINT'}"
                        )
                        continue
                    location = prop.get("location", ())
                    if len(location) < 2:
                        print(f"WARN visual prop {index} has no valid XY location")
                        continue
                    rotation = prop.get("rotation_deg", ())
                    pitch = float(rotation[0]) if len(rotation) > 0 else 0.0
                    roll = float(rotation[1]) if len(rotation) > 1 else 0.0
                    yaw = (
                        float(rotation[2])
                        if len(rotation) > 2
                        else cfg.carla.road_heading_deg
                    )
                    authored_z = float(location[2]) if len(location) > 2 else 0.0
                    transform = _surface_transform(
                        carla_map,
                        float(location[0]),
                        float(location[1]),
                        yaw,
                        self.z_offset + authored_z,
                        pitch=pitch,
                        roll=roll,
                    )
                    print("visual_prop_blueprint", index, prop_bp.id, asset_hint)
                    self._spawn(prop_bp, transform, f"visual prop {index}")

            elif not self.cones_only and sign_bp is not None:
                sign_yaw = cfg.carla.road_heading_deg + self.sign_yaw_offset
                for index, (x, y) in enumerate(cfg.workzone.warning_signs, start=1):
                    transform = _surface_transform(
                        carla_map, x, y, sign_yaw, self.z_offset
                    )
                    self._spawn(sign_bp, transform, f"warning sign {index}")

            finish_line = cfg.destination.finish_line
            if self.draw_finish_line and finish_line is not None:
                start = _surface_transform(
                    carla_map, finish_line.start[0], finish_line.start[1],
                    cfg.carla.road_heading_deg, 0.15,
                ).location
                end = _surface_transform(
                    carla_map, finish_line.end[0], finish_line.end[1],
                    cfg.carla.road_heading_deg, 0.15,
                ).location
                world.debug.draw_line(
                    start, end,
                    thickness=0.25,
                    color=carla.Color(0, 255, 0),
                    life_time=600.0,
                )
                print("finish_line", finish_line.start, finish_line.end)

            if self.move_spectator:
                points = list(cfg.workzone.traffic_cones)
                if not self.cones_only:
                    if visual_props:
                        points.extend(
                            (float(prop["location"][0]), float(prop["location"][1]))
                            for prop in visual_props
                            if len(prop.get("location", ())) >= 2
                        )
                    else:
                        points.extend(cfg.workzone.warning_signs)
                points.extend((point.x, point.y) for point in cfg.origin.spawn_points)
                if finish_line is not None:
                    points.extend((finish_line.start, finish_line.end))
                if points:
                    center_x = sum(point[0] for point in points) / len(points)
                    center_y = sum(point[1] for point in points) / len(points)
                    scene_span = max(
                        max(point[0] for point in points) - min(point[0] for point in points),
                        max(point[1] for point in points) - min(point[1] for point in points),
                    )
                    camera_height = max(self.spectator_height, scene_span * 0.75)
                    ground = _surface_transform(
                        carla_map, center_x, center_y, cfg.carla.road_heading_deg, 0.0
                    )
                    world.get_spectator().set_transform(
                        carla.Transform(
                            carla.Location(
                                x=center_x,
                                y=center_y,
                                z=ground.location.z + camera_height,
                            ),
                            carla.Rotation(pitch=-80.0, yaw=cfg.carla.road_heading_deg),
                        )
                    )
        except Exception:
            self.close()
            raise

        print(f"Spawned {len(self.actors)} visual debug props for {self.setting_id}.")
        return self

    def _spawn(self, blueprint, transform, label: str) -> None:
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "debug_workzone_prop")
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            print(
                f"WARN could not spawn {label} at "
                f"({transform.location.x:.2f}, {transform.location.y:.2f}, "
                f"{transform.location.z:.2f})"
            )
            return
        try:
            actor.set_simulate_physics(False)
            actor.set_enable_gravity(False)
        except (AttributeError, RuntimeError):
            pass
        self.actors.append(actor)

    def close(self) -> None:
        destroyed = 0
        for actor in reversed(self.actors):
            try:
                if actor.is_alive and actor.destroy():
                    destroyed += 1
            except RuntimeError:
                pass
        if self.actors:
            print(f"Destroyed {destroyed}/{len(self.actors)} visual debug props.")
        self.actors.clear()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--wz", required=True)
    parser.add_argument("--layout", "--abc", dest="layout", required=True,
                        choices=("a", "b", "c"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--z-offset", type=float, default=0.08)
    parser.add_argument("--sign-yaw-offset", type=float, default=180.0)
    parser.add_argument("--cones-only", action="store_true")
    parser.add_argument("--no-finish-line", action="store_true")
    parser.add_argument("--spectator-height", type=float, default=150.0)
    parser.add_argument("--no-move-spectator", action="store_true")
    parser.add_argument("--no-load-town", action="store_true")
    parser.add_argument("--lifetime-s", type=float, default=0.0)
    args = parser.parse_args(argv)

    setting = f"{args.scenario.lower()}/{_canonical_wz(args.wz)}/{args.layout}"
    visualizer = WorkZonePropVisualizer(
        setting,
        host=args.host,
        carla_port=args.carla_port,
        timeout=args.timeout,
        z_offset=args.z_offset,
        sign_yaw_offset=args.sign_yaw_offset,
        cones_only=args.cones_only,
        draw_finish_line=not args.no_finish_line,
        load_town=not args.no_load_town,
        move_spectator=not args.no_move_spectator,
        spectator_height=args.spectator_height,
    ).start()
    try:
        if args.lifetime_s > 0.0:
            time.sleep(args.lifetime_s)
        else:
            print("Press Ctrl+C to remove the visual props.")
            while True:
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
