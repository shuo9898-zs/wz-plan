"""Shared implementation for the six standalone work-zone zone viewers."""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenario_catalog import list_setting_ids
from config.scenario_config import load_scenario
from tools import show_traffic_cones as cone_tool
from tools import show_workzone_polygons_v2 as polygon_tool
from tools.show_s2_violation_zones_v2 import (
    _expected_prop_count_v2,
    _move_spectator_to_zone_v2,
    hold_for_dwell_v2,
)


@dataclass(frozen=True)
class ZoneScenarioSpecV2:
    scenario_id: str
    town: str
    default_carla_port: int
    semantics: str
    origin_specific: bool = False

    @property
    def zone_color_name(self) -> str:
        return "green" if self.semantics == "drivable_union" else "red"


@dataclass(frozen=True)
class ZoneDisplayItemV2:
    setting_id: str
    origin_index: int | None = None

    @property
    def label(self) -> str:
        if self.origin_index is None:
            return self.setting_id
        return f"{self.setting_id}#origin{self.origin_index}"


SCENARIO_SPECS_V2 = {
    "s1": ZoneScenarioSpecV2("s1", "Town02", 2000, "forbidden_area"),
    "s2": ZoneScenarioSpecV2("s2", "Town05", 2020, "forbidden_area"),
    "s3": ZoneScenarioSpecV2(
        "s3", "Town10HD", 2040, "drivable_union", origin_specific=True
    ),
    "s4": ZoneScenarioSpecV2("s4", "Town10HD", 2040, "forbidden_area"),
    "s5": ZoneScenarioSpecV2("s5", "Town05", 2020, "forbidden_area"),
    "s6": ZoneScenarioSpecV2("s6", "Town02", 2000, "forbidden_area"),
}


def scenario_spec_v2(scenario_id: str) -> ZoneScenarioSpecV2:
    normalized = str(scenario_id).strip().lower()
    try:
        return SCENARIO_SPECS_V2[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported scenario zone viewer: {scenario_id!r}") from exc


def scenario_display_items_v2(
    scenario_id: str,
    *,
    origin_index: int | None = None,
) -> tuple[ZoneDisplayItemV2, ...]:
    """Return the complete deterministic setting/origin slideshow."""
    spec = scenario_spec_v2(scenario_id)
    settings = tuple(list_setting_ids(spec.scenario_id, runnable_only=True))
    if not settings:
        raise RuntimeError(f"{spec.scenario_id} has no runnable settings")

    items: list[ZoneDisplayItemV2] = []
    for setting in settings:
        config = load_scenario(setting)
        if config.carla.town != spec.town:
            raise RuntimeError(
                f"{setting} requires {config.carla.town}, expected {spec.town}"
            )
        if spec.origin_specific:
            origins = tuple(config.origin.spawn_points)
            if origin_index is None:
                selected_origins = range(len(origins))
            else:
                if not 0 <= origin_index < len(origins):
                    raise ValueError(
                        f"--origin-index must be in 0..{len(origins) - 1} for {setting}"
                    )
                selected_origins = (origin_index,)
            items.extend(
                ZoneDisplayItemV2(setting, selected) for selected in selected_origins
            )
        else:
            if origin_index is not None:
                raise ValueError("--origin-index is available only for S3")
            items.append(ZoneDisplayItemV2(setting))
    return tuple(items)


def build_parser_for_scenario_v2(scenario_id: str) -> argparse.ArgumentParser:
    spec = scenario_spec_v2(scenario_id)
    if spec.origin_specific:
        semantics = (
            "Green is the allowed drivable union; leaving it is a violation. "
            "Every origin is displayed because the entry geometry is origin-specific."
        )
    else:
        semantics = "Red is the forbidden area; entering/touching it is a violation."
    parser = argparse.ArgumentParser(
        description=(
            f"Show every {spec.scenario_id.upper()} work-zone/layout in {spec.town}. "
            f"{semantics} No ego, SUMO, Gym environment, or RL model is created."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--carla-port", "--port", type=_port_v2,
        default=spec.default_carla_port,
    )
    parser.add_argument("--timeout", type=_positive_float_v2, default=10.0)
    parser.add_argument(
        "--dwell-seconds",
        type=_positive_float_v2,
        default=20.0,
        help="wall-clock display time for each setting/origin view",
    )
    parser.add_argument(
        "--origin-index",
        type=_nonnegative_int_v2,
        default=None,
        help="S3 only: show one origin per setting instead of all origins",
    )
    parser.add_argument(
        "--load-town",
        action="store_true",
        help=f"load {spec.town} if the connected CARLA server uses another map",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="print the complete display order without connecting to CARLA",
    )
    parser.add_argument(
        "--cones-only",
        action="store_true",
        help="omit the authored head marker/sign; ordinary cones remain visible",
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


def main_for_scenario_v2(
    scenario_id: str,
    argv: Sequence[str] | None = None,
) -> int:
    spec = scenario_spec_v2(scenario_id)
    parser = build_parser_for_scenario_v2(spec.scenario_id)
    args = parser.parse_args(argv)
    try:
        items = scenario_display_items_v2(
            spec.scenario_id,
            origin_index=args.origin_index,
        )
        if args.list_only:
            print("\n".join(item.label for item in items))
            return 0

        first_config = load_scenario(items[0].setting_id)
        carla_module = polygon_tool._import_carla_v2()
        world = polygon_tool._connect_world_v2(carla_module, first_config, args)
        clock = world.get_settings()
        print(
            "ZONE_CAROUSEL",
            f"scenario={spec.scenario_id}",
            f"town={world.get_map().name.split('/')[-1]}",
            f"port={args.carla_port}",
            f"views={len(items)}",
            f"dwell_s={args.dwell_seconds:.1f}",
            f"semantics={spec.semantics}",
            f"color={spec.zone_color_name}",
            f"synchronous={bool(clock.synchronous_mode)}",
            f"fixed_delta_s={clock.fixed_delta_seconds}",
            flush=True,
        )

        for item_number, item in enumerate(items, start=1):
            config = load_scenario(item.setting_id)
            geometry = polygon_tool.build_visualization_geometry_v2(
                config,
                origin_index=item.origin_index,
                carla_map=world.get_map(),
            )
            args.road_heading_deg = config.carla.road_heading_deg
            args.show_finish_line = False
            prop_view = cone_tool.WorkZonePropVisualizer(
                item.setting_id,
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
                    item.setting_id,
                    config,
                    args.cones_only,
                )
                spawned_props = len(prop_view.actors)
                if spawned_props != expected_props and not args.allow_missing_props:
                    raise RuntimeError(
                        f"{item.label} spawned {spawned_props}/{expected_props} "
                        "expected props; pass --allow-missing-props only for debugging"
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
                    f"SHOW [{item_number:02d}/{len(items):02d}] {item.label} "
                    f"semantics={spec.semantics} color={spec.zone_color_name} "
                    f"props={spawned_props}/{expected_props} "
                    f"dwell_s={args.dwell_seconds:.1f}",
                    flush=True,
                )
                ticks = hold_for_dwell_v2(
                    world,
                    args.dwell_seconds,
                    args.timeout,
                )
                print(f"DONE {item.label} ticks={ticks}", flush=True)
            finally:
                prop_view.close()
        print(f"ZONE_CAROUSEL complete scenario={spec.scenario_id}", flush=True)
        return 0
    except KeyboardInterrupt:
        print(f"ZONE_CAROUSEL interrupted scenario={spec.scenario_id}", flush=True)
        return 0
    except (ImportError, IndexError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "SCENARIO_SPECS_V2",
    "ZoneDisplayItemV2",
    "ZoneScenarioSpecV2",
    "build_parser_for_scenario_v2",
    "main_for_scenario_v2",
    "scenario_display_items_v2",
    "scenario_spec_v2",
]

