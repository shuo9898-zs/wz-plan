"""Cycle all exact S2 violation zones and their authored CARLA props.

The tool is intentionally independent from every RL entry point: it creates
no ego vehicle, starts no SUMO process, and never constructs a Gym/PPO
environment.  Each red cross-hatched polygon is built through the same curved
lane-buffer builder used by V2 termination.  The corresponding cone/head
props are destroyed before the next S2 WZ/layout is shown.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenario_catalog import list_setting_ids
from config.scenario_config import load_scenario
from tools import show_traffic_cones as cone_tool
from tools import show_workzone_polygons_v2 as polygon_tool


S2_TOWN_V2 = "Town05"
S2_DEFAULT_CARLA_PORT_V2 = 2020
S2_DEFAULT_DWELL_SECONDS_V2 = 20.0


def ordered_s2_settings_v2() -> tuple[str, ...]:
    """Return and validate WZ1-4 x A/B/C in deterministic display order."""
    settings = tuple(list_setting_ids("s2", runnable_only=True))
    expected = tuple(
        f"s2/wz{wz_index}/{layout}"
        for wz_index in range(1, 5)
        for layout in ("a", "b", "c")
    )
    if settings != expected:
        raise RuntimeError(
            "S2 carousel requires exactly WZ1-4 x A/B/C in catalog order; "
            f"expected={expected}, actual={settings}"
        )
    for setting in settings:
        config = load_scenario(setting)
        if config.carla.town != S2_TOWN_V2:
            raise RuntimeError(
                f"{setting} requires {config.carla.town}, expected {S2_TOWN_V2}"
            )
        if config.workzone.geometry_mode != "forbidden_polygon":
            raise RuntimeError(
                f"{setting} uses {config.workzone.geometry_mode}, expected forbidden_polygon"
            )
        if config.workzone.polygon_config is None:
            raise RuntimeError(f"{setting} has no curved polygon_config")
    return settings


def build_parser_v2() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--carla-port", "--port", type=_port_v2,
        default=S2_DEFAULT_CARLA_PORT_V2,
    )
    parser.add_argument("--timeout", type=_positive_float_v2, default=10.0)
    parser.add_argument(
        "--dwell-seconds",
        type=_positive_float_v2,
        default=S2_DEFAULT_DWELL_SECONDS_V2,
        help="wall-clock display time for each of the 12 settings",
    )
    parser.add_argument(
        "--load-town",
        action="store_true",
        help="load Town05 if the connected server is on another map",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="print the 12-setting order without connecting to CARLA",
    )
    parser.add_argument(
        "--cones-only",
        action="store_true",
        help="omit the authored head marker/sign; ordinary cones are always shown",
    )
    parser.add_argument(
        "--allow-missing-props",
        action="store_true",
        help="continue if CARLA cannot spawn every expected cone/head prop",
    )
    parser.add_argument(
        "--fill-style",
        choices=("outline", "hatch", "crosshatch"),
        default="crosshatch",
    )
    parser.add_argument("--fill-spacing-m", type=_positive_float_v2, default=0.75)
    parser.add_argument("--outline-thickness", type=_positive_float_v2, default=0.14)
    parser.add_argument("--fill-thickness", type=_positive_float_v2, default=0.04)
    parser.add_argument("--z-offset", type=_nonnegative_float_v2, default=0.18)
    parser.add_argument("--cone-z-offset", type=_nonnegative_float_v2, default=0.08)
    parser.add_argument("--alpha", type=_byte_v2, default=170)
    parser.add_argument("--max-fill-segments", type=_positive_int_v2, default=2500)
    parser.add_argument("--show-centerline", action="store_true")
    parser.add_argument("--show-vertices", action="store_true")
    parser.add_argument("--no-move-spectator", action="store_true")
    parser.add_argument("--spectator-height", type=_positive_float_v2, default=65.0)
    return parser


def hold_for_dwell_v2(
    world,
    dwell_seconds: float,
    timeout_seconds: float,
    *,
    sleep_fn: Callable[[float], None] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> int:
    """Keep one setting visible for wall time and advance a synchronous world.

    CARLA DebugHelper lifetimes advance in simulation time.  A previously used
    training world is commonly left synchronous; merely sleeping would leave
    every red polygon on screen.  In that case this standalone viewer becomes
    the sole tick owner and paces exactly enough ticks to expire the drawing.
    Returns the number of ticks issued (zero in asynchronous mode).
    """
    dwell_seconds = float(dwell_seconds)
    timeout_seconds = float(timeout_seconds)
    if not math.isfinite(dwell_seconds) or dwell_seconds <= 0.0:
        raise ValueError("dwell_seconds must be finite and positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    monotonic_fn = time.monotonic if monotonic_fn is None else monotonic_fn

    settings = world.get_settings()
    if not bool(settings.synchronous_mode):
        sleep_fn(dwell_seconds)
        return 0

    fixed_delta = settings.fixed_delta_seconds
    if fixed_delta is None or not math.isfinite(float(fixed_delta)) or fixed_delta <= 0.0:
        raise RuntimeError(
            "Synchronous CARLA visualization requires a positive fixed_delta_seconds"
        )
    tick_count = max(1, int(math.ceil(dwell_seconds / float(fixed_delta))))
    started = monotonic_fn()
    for tick_index in range(tick_count):
        world.tick(timeout_seconds)
        target = started + dwell_seconds * (tick_index + 1) / tick_count
        delay = target - monotonic_fn()
        if delay > 0.0:
            sleep_fn(delay)
    return tick_count


def _expected_prop_count_v2(setting: str, config, cones_only: bool) -> int:
    count = len(config.workzone.traffic_cones)
    if cones_only:
        return count
    visual_props = cone_tool._visual_debug_props(setting)
    if visual_props:
        return count + len(visual_props)
    return count + len(config.workzone.warning_signs)


def _move_spectator_to_zone_v2(carla_module, world, geometry, height_m: float) -> None:
    points = [point for layer in geometry.layers for point in layer.vertices]
    if not points:
        return
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    span = max(max_x - min_x, max_y - min_y)
    surface = polygon_tool._surface_location_v2(
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
                z=surface.z + max(float(height_m), 0.8 * span),
            ),
            carla_module.Rotation(pitch=-82.0, yaw=0.0),
        )
    )


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser_v2()
    args = parser.parse_args(argv)
    try:
        settings = ordered_s2_settings_v2()
        if args.list_only:
            print("\n".join(settings))
            return 0

        configs = [load_scenario(setting) for setting in settings]
        carla_module = polygon_tool._import_carla_v2()
        world = polygon_tool._connect_world_v2(carla_module, configs[0], args)
        clock = world.get_settings()
        print(
            "S2_VIOLATION_CAROUSEL",
            f"town={world.get_map().name.split('/')[-1]}",
            f"port={args.carla_port}",
            f"settings={len(settings)}",
            f"dwell_s={args.dwell_seconds:.1f}",
            f"synchronous={bool(clock.synchronous_mode)}",
            f"fixed_delta_s={clock.fixed_delta_seconds}",
            flush=True,
        )

        for item_index, (setting, config) in enumerate(zip(settings, configs), start=1):
            geometry = polygon_tool.build_visualization_geometry_v2(
                config,
                carla_map=world.get_map(),
            )
            args.road_heading_deg = config.carla.road_heading_deg
            args.show_finish_line = False
            prop_view = cone_tool.WorkZonePropVisualizer(
                setting,
                host=args.host,
                carla_port=args.carla_port,
                timeout=args.timeout,
                z_offset=args.cone_z_offset,
                cones_only=args.cones_only,
                draw_finish_line=False,
                load_town=False,
                move_spectator=False,
            )
            try:
                prop_view.start()
                expected_props = _expected_prop_count_v2(
                    setting, config, args.cones_only
                )
                spawned_props = len(prop_view.actors)
                if spawned_props != expected_props and not args.allow_missing_props:
                    raise RuntimeError(
                        f"{setting} spawned {spawned_props}/{expected_props} expected props; "
                        "use --allow-missing-props only for visual debugging"
                    )
                if not args.no_move_spectator:
                    _move_spectator_to_zone_v2(
                        carla_module,
                        world,
                        geometry,
                        args.spectator_height,
                    )
                polygon_tool._draw_geometry_once_v2(
                    carla_module,
                    world,
                    geometry,
                    args,
                    life_time=args.dwell_seconds,
                )
                print(
                    f"SHOW [{item_index:02d}/{len(settings):02d}] {setting} "
                    f"violation=red_forbidden_area "
                    f"props={spawned_props}/{expected_props} "
                    f"dwell_s={args.dwell_seconds:.1f}",
                    flush=True,
                )
                ticks = hold_for_dwell_v2(
                    world,
                    args.dwell_seconds,
                    args.timeout,
                )
                print(f"DONE {setting} ticks={ticks}", flush=True)
            finally:
                prop_view.close()
        print("S2_VIOLATION_CAROUSEL complete", flush=True)
        return 0
    except KeyboardInterrupt:
        print("S2_VIOLATION_CAROUSEL interrupted", flush=True)
        return 0
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "S2_DEFAULT_CARLA_PORT_V2",
    "S2_DEFAULT_DWELL_SECONDS_V2",
    "S2_TOWN_V2",
    "build_parser_v2",
    "hold_for_dwell_v2",
    "main",
    "ordered_s2_settings_v2",
]
