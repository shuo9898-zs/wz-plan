"""Validate or draw the exact PPO V2 work-zone geometry in CARLA.

Forbidden geometry is red for S1/S2/S4/S5/S6.  S3 draws its origin-specific
entry approach, corridor, and exit polygon in green because that union
is the allowed driving area.  CARLA's DebugHelper has no filled polygon API,
so the apparent fill is made from finite-lived outline and hatch line segments.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenario_config import load_scenario
from logic.scenario_geometry_adapter_v2 import (
    Point2DV2,
    build_forbidden_area_from_config_v2,
    build_s3_drivable_area_from_config_v2,
    finish_line_from_config_v2,
)
from logic.termination_checker_v2 import FinishLineV2


FORBIDDEN_RED_V2 = (255, 35, 35)
DRIVABLE_GREEN_V2 = (25, 255, 80)
FINISH_YELLOW_V2 = (255, 230, 25)
CENTERLINE_WHITE_V2 = (245, 245, 245)
ORIGIN_CYAN_V2 = (30, 225, 255)


@dataclass(frozen=True)
class PolygonLayerV2:
    label: str
    vertices: Tuple[Point2DV2, ...]
    color_rgb: Tuple[int, int, int]
    holes: Tuple[Tuple[Point2DV2, ...], ...] = ()

    @property
    def polygon(self) -> Tuple[Point2DV2, ...]:
        """Compatibility name for callers that treat a layer as one polygon."""
        return self.vertices

    @property
    def color(self) -> Tuple[int, int, int]:
        return self.color_rgb


@dataclass(frozen=True)
class VisualizationGeometryV2:
    setting_id: str
    scenario_id: str
    layers: Tuple[PolygonLayerV2, ...]
    finish_line: FinishLineV2
    centerline: Tuple[Point2DV2, ...] = ()
    origin: Point2DV2 | None = None

    @property
    def polygons(self) -> Tuple[PolygonLayerV2, ...]:
        return self.layers


def build_visualization_geometry_v2(
    config,
    *,
    origin_index: int | None = None,
    carla_map: object | None = None,
) -> VisualizationGeometryV2:
    """Build render layers without importing CARLA unless S2 needs its map."""
    scenario_id = str(config.scenario_id).strip().lower()
    finish_line = finish_line_from_config_v2(config)
    if scenario_id == "s3":
        if origin_index is None:
            raise ValueError("--origin-index is required for S3 geometry")
        area = build_s3_drivable_area_from_config_v2(config, origin_index)
        origin_cfg = config.origin.spawn_points[origin_index]
        layers = []
        if area.entry_approach_polygon:
            layers.append(
                PolygonLayerV2(
                    "entry_approach_polygon",
                    area.entry_approach_polygon,
                    DRIVABLE_GREEN_V2,
                )
            )
        layers.extend(
            (
                PolygonLayerV2(
                    "corridor_polygon", area.corridor_polygon, DRIVABLE_GREEN_V2
                ),
                PolygonLayerV2(
                    (
                        "exit_corridor_polygon"
                        if area.exit_corridor_polygon
                        else "exit_quadrilateral"
                    ),
                    area.exit_polygon,
                    DRIVABLE_GREEN_V2,
                ),
            )
        )
        return VisualizationGeometryV2(
            setting_id=config.setting_id,
            scenario_id=scenario_id,
            layers=tuple(layers),
            finish_line=finish_line,
            origin=(float(origin_cfg.x), float(origin_cfg.y)),
        )

    if origin_index is not None:
        raise ValueError("--origin-index may only be used with S3 settings")

    forbidden = build_forbidden_area_from_config_v2(config, carla_map=carla_map)
    return VisualizationGeometryV2(
        setting_id=config.setting_id,
        scenario_id=scenario_id,
        layers=(
            PolygonLayerV2(
                "forbidden_area",
                forbidden.polygon,
                FORBIDDEN_RED_V2,
                forbidden.holes,
            ),
        ),
        finish_line=finish_line,
        centerline=forbidden.centerline,
    )


def _hatch_segments_v2(
    vertices: Sequence[Point2DV2],
    spacing_m: float,
    crosshatch: bool = False,
    *,
    holes: Sequence[Sequence[Point2DV2]] = (),
) -> Tuple[Tuple[Point2DV2, Point2DV2], ...]:
    """Return deterministic even/odd scanline fill segments for a polygon.

    The implementation is pure Python and handles concavity and holes.  A
    half-open edge rule prevents double intersections at polygon vertices.
    """
    spacing_m = float(spacing_m)
    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    rings = (_normalized_ring_v2(vertices, "vertices"),) + tuple(
        _normalized_ring_v2(ring, f"holes[{index}]")
        for index, ring in enumerate(holes)
    )
    segments = list(_axis_hatch_segments_v2(rings, spacing_m, vertical=False))
    if crosshatch:
        segments.extend(_axis_hatch_segments_v2(rings, spacing_m, vertical=True))
    return tuple(segments)


def hatch_segments_v2(
    vertices: Sequence[Point2DV2],
    spacing_m: float,
    crosshatch: bool = False,
    *,
    holes: Sequence[Sequence[Point2DV2]] = (),
) -> Tuple[Tuple[Point2DV2, Point2DV2], ...]:
    """Public spelling of :func:`_hatch_segments_v2` for tests and reuse."""
    return _hatch_segments_v2(
        vertices,
        spacing_m,
        crosshatch,
        holes=holes,
    )


def _axis_hatch_segments_v2(
    rings: Sequence[Sequence[Point2DV2]],
    spacing_m: float,
    *,
    vertical: bool,
) -> Iterable[Tuple[Point2DV2, Point2DV2]]:
    primary_index = 0 if vertical else 1
    secondary_index = 1 - primary_index
    primary_values = [point[primary_index] for ring in rings for point in ring]
    primary_min = min(primary_values)
    primary_max = max(primary_values)
    scan = primary_min + 0.5 * spacing_m
    epsilon = 1e-10
    while scan < primary_max - epsilon:
        intersections: list[float] = []
        for ring in rings:
            for start, end in zip(ring, ring[1:] + ring[:1]):
                start_primary = start[primary_index]
                end_primary = end[primary_index]
                if abs(end_primary - start_primary) <= epsilon:
                    continue
                lower = min(start_primary, end_primary)
                upper = max(start_primary, end_primary)
                if not (lower <= scan < upper):
                    continue
                fraction = (scan - start_primary) / (end_primary - start_primary)
                intersections.append(
                    start[secondary_index]
                    + fraction * (end[secondary_index] - start[secondary_index])
                )
        intersections.sort()
        for index in range(0, len(intersections) - 1, 2):
            first = intersections[index]
            second = intersections[index + 1]
            if second - first <= epsilon:
                continue
            if vertical:
                yield ((scan, first), (scan, second))
            else:
                yield ((first, scan), (second, scan))
        scan += spacing_m


def build_parser_v2() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setting",
        required=True,
        help="canonical scenario/wz/layout id, for example s3/wz3/a",
    )
    parser.add_argument(
        "--origin-index",
        type=_nonnegative_int_v2,
        default=None,
        help="zero-based ego origin index; required for S3 only",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--carla-port", "--port", type=_port_v2, default=2000)
    parser.add_argument("--timeout", type=_positive_float_v2, default=10.0)

    town_group = parser.add_mutually_exclusive_group()
    town_group.add_argument(
        "--load-town",
        dest="load_town",
        action="store_true",
        help="load the setting's town if the connected world differs",
    )
    town_group.add_argument(
        "--no-load-town",
        dest="load_town",
        action="store_false",
        help="fail rather than changing the connected CARLA world (default)",
    )
    parser.set_defaults(load_town=False)

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and summarize geometry without drawing it",
    )
    parser.add_argument(
        "--fill-style",
        choices=("outline", "hatch", "crosshatch"),
        default="hatch",
    )
    parser.add_argument("--fill-spacing-m", type=_positive_float_v2, default=0.75)
    parser.add_argument("--outline-thickness", type=_positive_float_v2, default=0.12)
    parser.add_argument("--fill-thickness", type=_positive_float_v2, default=0.035)
    parser.add_argument("--z-offset", type=_nonnegative_float_v2, default=0.18)
    parser.add_argument(
        "--alpha",
        type=_byte_v2,
        default=160,
        help="best-effort DebugHelper color alpha (some CARLA builds ignore it)",
    )
    parser.add_argument(
        "--lifetime-s",
        type=_positive_float_v2,
        default=30.0,
        help="finite server-side lifetime for a single drawing",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="redraw short-lived primitives until Ctrl+C",
    )
    parser.add_argument(
        "--refresh-period-s",
        type=_positive_float_v2,
        default=0.75,
        help="redraw interval used by --hold",
    )
    parser.add_argument("--max-fill-segments", type=_positive_int_v2, default=2000)
    parser.add_argument("--show-centerline", action="store_true")
    parser.add_argument("--show-vertices", action="store_true")
    parser.add_argument("--move-spectator", action="store_true")
    parser.add_argument("--spectator-height", type=_positive_float_v2, default=120.0)
    return parser


def _validate_runtime_args_v2(args, config) -> None:
    if str(config.scenario_id).strip().lower() == "s3":
        if args.origin_index is None:
            raise ValueError("--origin-index is required for S3 settings")
        count = len(config.origin.spawn_points)
        if args.origin_index >= count:
            raise ValueError(
                f"--origin-index must be in 0..{count - 1} for {config.setting_id}"
            )
    elif args.origin_index is not None:
        raise ValueError("--origin-index may only be used with S3 settings")


def _import_carla_v2():
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "Drawing work-zone polygons requires the CARLA PythonAPI in this "
            "Python environment"
        ) from exc
    return carla


def _connect_world_v2(carla_module, config, args):
    client = carla_module.Client(args.host, args.carla_port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    current_town = world.get_map().name.split("/")[-1]
    if _canonical_town_v2(current_town) != _canonical_town_v2(config.carla.town):
        if not args.load_town:
            raise RuntimeError(
                f"CARLA has {current_town}, but {config.setting_id} requires "
                f"{config.carla.town}; pass --load-town to change it"
            )
        print(f"Loading {config.carla.town} (current map: {current_town}) ...")
        world = client.load_world(config.carla.town)
    return world


def _draw_geometry_once_v2(
    carla_module,
    world,
    geometry: VisualizationGeometryV2,
    args,
    *,
    life_time: float,
) -> None:
    if not math.isfinite(life_time) or life_time <= 0.0:
        raise ValueError("Debug primitives must have a finite positive life_time")
    carla_map = world.get_map()
    debug = world.debug
    projected: dict[tuple[float, float, float], object] = {}

    def location(point: Point2DV2, extra_z: float = 0.0):
        key = (float(point[0]), float(point[1]), float(extra_z))
        if key not in projected:
            projected[key] = _surface_location_v2(
                carla_module,
                carla_map,
                point,
                args.z_offset + extra_z,
            )
        return projected[key]

    for layer in geometry.layers:
        color = _debug_color_v2(carla_module, layer.color_rgb, args.alpha)
        for ring in (layer.vertices,) + layer.holes:
            for start, end in zip(ring, ring[1:] + ring[:1]):
                debug.draw_line(
                    location(start),
                    location(end),
                    thickness=args.outline_thickness,
                    color=color,
                    life_time=life_time,
                    persistent_lines=False,
                )

        if args.fill_style != "outline":
            segments = _hatch_segments_v2(
                layer.vertices,
                args.fill_spacing_m,
                args.fill_style == "crosshatch",
                holes=layer.holes,
            )
            if len(segments) > args.max_fill_segments:
                stride = math.ceil(len(segments) / args.max_fill_segments)
                segments = segments[::stride]
            for start, end in segments:
                debug.draw_line(
                    location(start, 0.01),
                    location(end, 0.01),
                    thickness=args.fill_thickness,
                    color=color,
                    life_time=life_time,
                    persistent_lines=False,
                )

        if args.show_vertices:
            for index, point in enumerate(layer.vertices):
                debug.draw_point(
                    location(point, 0.04),
                    size=0.12,
                    color=color,
                    life_time=life_time,
                    persistent_lines=False,
                )
                debug.draw_string(
                    location(point, 0.12),
                    f"{layer.label}:{index}",
                    draw_shadow=True,
                    color=color,
                    life_time=life_time,
                    persistent_lines=False,
                )

    if getattr(args, "show_finish_line", True):
        finish_color = _debug_color_v2(carla_module, FINISH_YELLOW_V2, 255)
        debug.draw_line(
            location(geometry.finish_line.first, 0.05),
            location(geometry.finish_line.second, 0.05),
            thickness=max(args.outline_thickness, 0.18),
            color=finish_color,
            life_time=life_time,
            persistent_lines=False,
        )

    if args.show_centerline and geometry.centerline:
        centerline_color = _debug_color_v2(carla_module, CENTERLINE_WHITE_V2, 220)
        for start, end in zip(geometry.centerline, geometry.centerline[1:]):
            debug.draw_line(
                location(start, 0.03),
                location(end, 0.03),
                thickness=args.fill_thickness,
                color=centerline_color,
                life_time=life_time,
                persistent_lines=False,
            )

    if geometry.origin is not None:
        origin_color = _debug_color_v2(carla_module, ORIGIN_CYAN_V2, 255)
        debug.draw_point(
            location(geometry.origin, 0.08),
            size=0.2,
            color=origin_color,
            life_time=life_time,
            persistent_lines=False,
        )


def _surface_location_v2(carla_module, carla_map, point: Point2DV2, z_offset: float):
    query = carla_module.Location(x=float(point[0]), y=float(point[1]), z=1.0)
    try:
        waypoint = carla_map.get_waypoint(
            query,
            project_to_road=True,
            lane_type=carla_module.LaneType.Any,
        )
    except TypeError:
        waypoint = carla_map.get_waypoint(query, project_to_road=True)
    ground_z = waypoint.transform.location.z if waypoint is not None else 0.0
    return carla_module.Location(
        x=float(point[0]),
        y=float(point[1]),
        z=float(ground_z + z_offset),
    )


def _debug_color_v2(carla_module, rgb: Tuple[int, int, int], alpha: int):
    try:
        return carla_module.Color(rgb[0], rgb[1], rgb[2], int(alpha))
    except TypeError:
        return carla_module.Color(rgb[0], rgb[1], rgb[2])


def _move_spectator_v2(carla_module, world, geometry, args) -> None:
    points = [point for layer in geometry.layers for point in layer.vertices]
    points.extend((geometry.finish_line.first, geometry.finish_line.second))
    if geometry.origin is not None:
        points.append(geometry.origin)
    if not points:
        return
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    span = max(
        max(point[0] for point in points) - min(point[0] for point in points),
        max(point[1] for point in points) - min(point[1] for point in points),
    )
    ground = _surface_location_v2(
        carla_module,
        world.get_map(),
        (center_x, center_y),
        0.0,
    )
    world.get_spectator().set_transform(
        carla_module.Transform(
            carla_module.Location(
                x=center_x,
                y=center_y,
                z=ground.z + max(args.spectator_height, span * 0.75),
            ),
            carla_module.Rotation(
                pitch=-82.0,
                yaw=float(getattr(args, "road_heading_deg", 0.0)),
            ),
        )
    )


def _print_geometry_summary_v2(geometry: VisualizationGeometryV2) -> None:
    print("setting", geometry.setting_id)
    print("semantics", "drivable_union" if geometry.scenario_id == "s3" else "forbidden_area")
    for layer in geometry.layers:
        print(
            "polygon",
            layer.label,
            f"vertices={len(layer.vertices)}",
            f"holes={len(layer.holes)}",
            f"color={layer.color_rgb}",
        )
    if geometry.origin is not None:
        print("origin", geometry.origin)
    print("finish_line", geometry.finish_line.first, geometry.finish_line.second)


def _normalized_ring_v2(
    vertices: Sequence[Point2DV2],
    name: str,
) -> Tuple[Point2DV2, ...]:
    ring = tuple((float(point[0]), float(point[1])) for point in vertices)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError(f"{name} must contain at least three vertices")
    if not all(math.isfinite(value) for point in ring for value in point):
        raise ValueError(f"{name} vertices must be finite")
    return ring


def _canonical_town_v2(name: str) -> str:
    normalized = str(name).strip().lower().split("/")[-1]
    return normalized[:-4] if normalized.endswith("_opt") else normalized


def _positive_float_v2(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float_v2(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _positive_int_v2(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int_v2(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _port_v2(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be in 1..65535")
    return parsed


def _byte_v2(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("alpha must be in 0..255")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = build_parser_v2()
    args = parser.parse_args(argv)
    try:
        config = load_scenario(args.setting)
        _validate_runtime_args_v2(args, config)

        carla_module = None
        world = None
        needs_world = not args.validate_only or config.scenario_id == "s2"
        if needs_world:
            carla_module = _import_carla_v2()
            world = _connect_world_v2(carla_module, config, args)

        geometry = build_visualization_geometry_v2(
            config,
            origin_index=args.origin_index,
            carla_map=world.get_map() if world is not None else None,
        )
        _print_geometry_summary_v2(geometry)
        if args.validate_only:
            print("geometry_valid", True)
            return 0

        assert carla_module is not None and world is not None
        args.road_heading_deg = config.carla.road_heading_deg
        if args.move_spectator:
            _move_spectator_v2(carla_module, world, geometry, args)
        if not args.hold:
            _draw_geometry_once_v2(
                carla_module,
                world,
                geometry,
                args,
                life_time=args.lifetime_s,
            )
            print(f"drawn_for_s {args.lifetime_s:.3f}")
            return 0

        refresh_lifetime = max(0.25, args.refresh_period_s * 2.5)
        print("Holding finite-lived debug lines; press Ctrl+C to stop.")
        try:
            while True:
                _draw_geometry_once_v2(
                    carla_module,
                    world,
                    geometry,
                    args,
                    life_time=refresh_lifetime,
                )
                time.sleep(args.refresh_period_s)
        except KeyboardInterrupt:
            return 0
    except (KeyError, IndexError, ImportError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DRIVABLE_GREEN_V2",
    "FORBIDDEN_RED_V2",
    "PolygonLayerV2",
    "VisualizationGeometryV2",
    "build_parser_v2",
    "build_visualization_geometry_v2",
    "hatch_segments_v2",
    "main",
]
