"""Typed runtime configuration loaded from the canonical six-scenario catalog."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config.scenario_catalog import list_setting_ids, resolve_setting, scenario_ids


@dataclass
class WorkZonePolygonConfig:
    head_x: float
    head_y: float
    tail_x: float
    tail_y: float
    head_z: float = 0.0
    tail_z: float = 0.0
    sample_spacing_m: float = 0.5
    half_width_m: Optional[float] = None
    margin_m: float = 0.0
    expected_road_id: Optional[int] = None
    expected_lane_id: Optional[int] = None


@dataclass
class WorkZoneConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    geometry_mode: str = "forbidden_rect"
    boundary_tolerance_m: float = 0.0
    warning_signs: List[Tuple[float, float]] = field(default_factory=list)
    traffic_cones: List[Tuple[float, float]] = field(default_factory=list)
    closed_lanes: List[str] = field(default_factory=list)
    polygon_config: Optional[WorkZonePolygonConfig] = None
    corridor_boundary_points: Optional[List[Tuple[float, float]]] = None
    # S3 keeps the two ordered sides as first-class data.  Index 0 is the
    # entrance and the final index is the exit; both lists run in the ego's
    # direction of travel.  ``corridor_boundary_points`` remains available as
    # the closed-polygon representation used by legacy callers.
    corridor_left_boundary_points: Optional[List[Tuple[float, float]]] = None
    corridor_right_boundary_points: Optional[List[Tuple[float, float]]] = None
    # Optional map-authored continuation from the final cone cross-section to
    # the finish line.  These points are generated once from the intended lane
    # geometry and stored in the scenario file; runtime geometry never guesses
    # a route from the live simulator map.
    exit_left_boundary_points: Optional[List[Tuple[float, float]]] = None
    exit_right_boundary_points: Optional[List[Tuple[float, float]]] = None
    corridor_open_ends: bool = False


@dataclass
class EgoSpawnPointConfig:
    x: float
    y: float
    z: float = 0.5
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class OriginConfig:
    near_dist_m: float
    far_dist_m: float
    eval_distances_m: Tuple[float, ...] = (45.0, 50.0, 55.0)
    explicit_spawn_bounds: Optional[Tuple[float, float, float, float]] = None
    spawn_points: Tuple[EgoSpawnPointConfig, ...] = ()


@dataclass
class FinishLineConfig:
    start: Tuple[float, float]
    end: Tuple[float, float]
    crossing_reference: str = "vehicle_center"
    # Direction in which the ego centre must cross the finite segment.
    # ``None`` preserves the historical per-scenario road heading fallback.
    crossing_heading_deg: Optional[float] = None


@dataclass
class DestinationConfig:
    near_dist_m: float
    far_dist_m: float
    eval_distances_m: Tuple[float, ...] = (15.0, 20.0, 25.0)
    success_dist_m: float = 2.0
    success_heading_deg: float = 30.0
    finish_line: Optional[FinishLineConfig] = None


@dataclass
class EpisodeConfig:
    max_steps: int = 1200
    sim_dt: float = 0.1
    sync_hz: float = 10.0


@dataclass
class SUMOConfig:
    net_file: str
    route_file: str
    bg_routes: Tuple[str, ...]
    # Optional route-specific longitudinal insertion ranges on the first
    # edge, expressed in SUMO lane metres.  Routes omitted from this mapping
    # keep SUMO's normal ``departPos=base`` behaviour.
    bg_route_depart_pos_ranges_m: dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )
    # Optional traffic profiles for individual routes.  Routes listed here
    # are removed from the ordinary background-traffic pool: their initial
    # request count, active-plus-pending cap, and Poisson rate are controlled
    # by the profile instead.  The global fields below continue to describe
    # the shared pool made up of all remaining ``bg_routes``.
    bg_route_traffic_overrides: dict[str, dict[str, float | int]] = field(
        default_factory=dict
    )
    step_length: float = 0.1
    sumo_gui: bool = False
    port: int = 8813
    ego_proxy_id: str = "ego_proxy"
    ped_proxy_prefix: str = "ped_proxy_"
    max_background_vehicles: int = 40
    bg_spawn_rate_veh_s: float = 2.0
    bg_vtype: str = "car"
    # ``poisson`` preserves the ordinary S1/S5 background-flow behaviour.
    # ``platoon`` is used by S6 to create explicit 3--4 vehicle waves with a
    # guaranteed empty interval, instead of filling SUMO's pending queue.
    traffic_pattern: str = "poisson"
    initial_background_vehicles: int = 20
    platoon_size_min: int = 3
    platoon_size_max: int = 4
    platoon_headway_min_s: float = 1.6
    platoon_headway_max_s: float = 2.0
    platoon_gap_min_s: float = 10.0
    platoon_gap_max_s: float = 12.0

    def __post_init__(self) -> None:
        normalized_depart_ranges: dict[str, Tuple[float, float]] = {}
        for route_id, bounds in self.bg_route_depart_pos_ranges_m.items():
            if route_id not in self.bg_routes:
                raise ValueError(
                    "SUMO departure-position route must be present in bg_routes: "
                    f"{route_id}"
                )
            if len(bounds) != 2:
                raise ValueError(
                    f"SUMO departure-position range for {route_id} must have two bounds"
                )
            lower, upper = float(bounds[0]), float(bounds[1])
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(
                    f"SUMO departure-position range for {route_id} must be finite"
                )
            if lower < 0.0 or upper < lower:
                raise ValueError(
                    f"SUMO departure-position range for {route_id} must satisfy "
                    "0 <= min <= max"
                )
            normalized_depart_ranges[str(route_id)] = (lower, upper)
        self.bg_route_depart_pos_ranges_m = normalized_depart_ranges

        required_override_fields = {
            "initial_background_vehicles",
            "max_background_vehicles",
            "bg_spawn_rate_veh_s",
        }
        normalized_traffic_overrides: dict[str, dict[str, float | int]] = {}
        for route_id, raw_profile in self.bg_route_traffic_overrides.items():
            if route_id not in self.bg_routes:
                raise ValueError(
                    "SUMO traffic-override route must be present in bg_routes: "
                    f"{route_id}"
                )
            if set(raw_profile) != required_override_fields:
                raise ValueError(
                    f"SUMO traffic override for {route_id} must contain exactly "
                    f"{sorted(required_override_fields)}"
                )
            initial = raw_profile["initial_background_vehicles"]
            maximum = raw_profile["max_background_vehicles"]
            rate = float(raw_profile["bg_spawn_rate_veh_s"])
            if (
                isinstance(initial, bool)
                or not isinstance(initial, int)
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
            ):
                raise ValueError(
                    f"SUMO traffic override counts for {route_id} must be integers"
                )
            if initial < 0 or maximum < 0 or initial > maximum:
                raise ValueError(
                    f"SUMO traffic override for {route_id} must satisfy "
                    "0 <= initial_background_vehicles <= max_background_vehicles"
                )
            if not math.isfinite(rate) or rate < 0.0:
                raise ValueError(
                    f"SUMO traffic override rate for {route_id} must be finite "
                    "and non-negative"
                )
            normalized_traffic_overrides[str(route_id)] = {
                "initial_background_vehicles": int(initial),
                "max_background_vehicles": int(maximum),
                "bg_spawn_rate_veh_s": rate,
            }
        self.bg_route_traffic_overrides = normalized_traffic_overrides

        numeric_values = {
            "bg_spawn_rate_veh_s": self.bg_spawn_rate_veh_s,
            "platoon_headway_min_s": self.platoon_headway_min_s,
            "platoon_headway_max_s": self.platoon_headway_max_s,
            "platoon_gap_min_s": self.platoon_gap_min_s,
            "platoon_gap_max_s": self.platoon_gap_max_s,
        }
        for field_name, value in numeric_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.traffic_pattern not in {"poisson", "platoon"}:
            raise ValueError(
                "SUMO traffic_pattern must be either 'poisson' or 'platoon'"
            )
        if self.bg_route_traffic_overrides and self.traffic_pattern != "poisson":
            raise ValueError(
                "SUMO route-specific traffic overrides require traffic_pattern='poisson'"
            )
        if self.max_background_vehicles < 0:
            raise ValueError("max_background_vehicles must be non-negative")
        if self.initial_background_vehicles < 0:
            raise ValueError("initial_background_vehicles must be non-negative")
        if self.bg_spawn_rate_veh_s < 0.0:
            raise ValueError("bg_spawn_rate_veh_s must be non-negative")
        if self.platoon_size_min < 1 or self.platoon_size_max < self.platoon_size_min:
            raise ValueError("platoon size bounds must satisfy 1 <= min <= max")
        if (
            self.platoon_headway_min_s < 0.0
            or self.platoon_headway_max_s < self.platoon_headway_min_s
        ):
            raise ValueError("platoon headway bounds must satisfy 0 <= min <= max")
        if (
            self.platoon_gap_min_s < 0.0
            or self.platoon_gap_max_s < self.platoon_gap_min_s
        ):
            raise ValueError("platoon gap bounds must satisfy 0 <= min <= max")
        if self.traffic_pattern == "platoon":
            if self.initial_background_vehicles != 0:
                raise ValueError(
                    "platoon traffic must set initial_background_vehicles=0"
                )
            if self.bg_spawn_rate_veh_s != 0.0:
                raise ValueError("platoon traffic must set bg_spawn_rate_veh_s=0")
            if self.max_background_vehicles < self.platoon_size_min:
                raise ValueError(
                    "platoon max_background_vehicles must be >= platoon_size_min"
                )


@dataclass
class CARLAConfig:
    road_heading_deg: float
    lane_center_offset_m: float = 0.0
    lane_half_width: float = 3.5
    road_half_width: float = 5.0
    host: str = "localhost"
    port: int = 2000
    tm_port: int = 8000
    timeout: float = 10.0
    town: str = "Town02"
    fixed_delta_s: float = 0.1
    lateral_shift: float = -4.0
    server_command: list[str] | None = None
    server_start_timeout_s: float = 120.0
    server_retry_interval_s: float = 3.0
    no_rendering_mode: bool = False


@dataclass
class JaywalkerConfig:
    spawn: Tuple[float, float, float]
    collision_target: Tuple[float, float, float]
    disappear: Tuple[float, float, float]
    trigger_anchor: Tuple[float, float, float]
    yaw_deg: float = 0.0
    speed_mps: float = 2.0
    trigger_distances_m: Tuple[float, ...] = (45.0, 30.0, 15.0)
    disappear_radius_m: float = 0.5
    blueprint: str = "walker.pedestrian.0001"
    verified: bool = False


@dataclass
class ObservationConfig:
    # Observation v3 is a compact current-frame perception contract.  Stable
    # actor IDs preserve slots; raw IDs and temporal frame stacks are not
    # policy inputs.
    history_frames: int = 1
    max_agents: int = 8
    perception_radius_m: float = 50.0
    max_ego_speed_mps: float = 13.89
    max_acceleration_mps2: float = 8.0
    yaw_rate_limit_deg_s: float = 90.0
    relative_speed_scale_mps: float = 30.0
    missing_actor_ttl_steps: int = 2

    def __post_init__(self) -> None:
        if self.history_frames != 1:
            raise ValueError("Observation v3 supports one current perception frame")
        if self.max_agents != 8:
            raise ValueError("Observation v2 requires exactly eight actor slots")
        for name in (
            "perception_radius_m",
            "max_ego_speed_mps",
            "max_acceleration_mps2",
            "yaw_rate_limit_deg_s",
            "relative_speed_scale_mps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.missing_actor_ttl_steps < 0:
            raise ValueError("missing_actor_ttl_steps must be non-negative")


@dataclass
class RewardConfig:
    failure: float = -100.0
    success: float = 100.0
    timeout: float = -20.0
    progress_budget: float = 20.0
    step_cost: float = -0.01


@dataclass
class ScenarioConfig:
    scenario_id: str
    wz_id: str
    layout_id: str
    setting_id: str
    traffic_backend: str
    geometry_profile: str
    reward_profile: str
    workzone: WorkZoneConfig
    origin: OriginConfig
    destination: DestinationConfig
    carla: CARLAConfig
    sumo: SUMOConfig | None = None
    jaywalker: JaywalkerConfig | None = None
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    ego_role: str = "hero"

    @property
    def uses_sumo(self) -> bool:
        return self.traffic_backend == "carla_sumo"


def load_scenario(name: str) -> ScenarioConfig:
    record, data = resolve_setting(name)
    if "corridor" in data:
        return _load_corridor(record, data)
    return _load_classic(record, data)


def _load_classic(record, data: dict) -> ScenarioConfig:
    raw = dict(data["workzone"])
    polygon = raw.pop("polygon_config", None)
    boundary = raw.pop("corridor_boundary_points", None)
    left_boundary = raw.pop("corridor_left_boundary_points", None)
    right_boundary = raw.pop("corridor_right_boundary_points", None)
    raw["warning_signs"] = [tuple(p) for p in raw.get("warning_signs", [])]
    raw["traffic_cones"] = [tuple(p) for p in raw.get("traffic_cones", [])]
    left = [tuple(p) for p in left_boundary] if left_boundary else None
    right = [tuple(p) for p in right_boundary] if right_boundary else None
    if boundary:
        closed_boundary = [tuple(p) for p in boundary]
    elif left and right:
        closed_boundary = left + list(reversed(right))
    else:
        closed_boundary = None
    raw["corridor_boundary_points"] = closed_boundary
    raw["corridor_left_boundary_points"] = left
    raw["corridor_right_boundary_points"] = right
    raw["polygon_config"] = WorkZonePolygonConfig(**polygon) if polygon else None
    workzone = WorkZoneConfig(**raw)
    return _assemble(record, data, workzone)


def _load_corridor(record, data: dict) -> ScenarioConfig:
    coordinates = data.get("coordinates", {})
    scale = float(coordinates.get("units_per_meter", 100.0))
    if scale <= 0:
        raise ValueError("units_per_meter must be positive")
    corridor = data["corridor"]
    raw_left = corridor.get("left_boundary_points")
    raw_right = corridor.get("right_boundary_points")
    raw_exit_left = corridor.get("exit_left_boundary_points")
    raw_exit_right = corridor.get("exit_right_boundary_points")
    raw_boundary = corridor.get("corridor_boundary_points")
    left = (
        [(float(x) / scale, float(y) / scale) for x, y in raw_left]
        if raw_left else None
    )
    right = (
        [(float(x) / scale, float(y) / scale) for x, y in raw_right]
        if raw_right else None
    )
    exit_left = (
        [(float(x) / scale, float(y) / scale) for x, y in raw_exit_left]
        if raw_exit_left else None
    )
    exit_right = (
        [(float(x) / scale, float(y) / scale) for x, y in raw_exit_right]
        if raw_exit_right else None
    )
    if bool(corridor.get("corridor_open_ends", False)) and (
        not left or not right or len(left) < 2 or len(right) < 2
    ):
        raise ValueError(
            "An open-ended corridor requires at least two ordered points on "
            "both left_boundary_points and right_boundary_points"
        )
    if left and right and len(left) != len(right):
        raise ValueError(
            "Corridor left/right boundaries must contain matching cross-sections"
        )
    if (exit_left is None) != (exit_right is None):
        raise ValueError(
            "Exit corridor requires both exit_left_boundary_points and "
            "exit_right_boundary_points"
        )
    if exit_left is not None and (len(exit_left) < 2 or len(exit_right) < 2):
        raise ValueError(
            "Exit corridor boundaries must each contain at least two points"
        )

    boundary = None
    if raw_boundary:
        boundary = [
            (float(x) / scale, float(y) / scale) for x, y in raw_boundary
        ]
    elif left and right:
        boundary = left + list(reversed(right))

    if boundary:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    else:
        x_min = float(corridor["x_min"]) / scale
        x_max = float(corridor["x_max"]) / scale
        y_min = float(corridor["y_min"]) / scale
        y_max = float(corridor["y_max"]) / scale
    workzone = WorkZoneConfig(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        geometry_mode="safe_corridor",
        boundary_tolerance_m=float(corridor.get("boundary_tolerance_m", 0.0)),
        warning_signs=[
            (float(x) / scale, float(y) / scale)
            for x, y in corridor.get("warning_signs", [])
        ],
        traffic_cones=[
            (float(x) / scale, float(y) / scale)
            for x, y in corridor.get("traffic_cones", [])
        ],
        corridor_boundary_points=boundary,
        corridor_left_boundary_points=left,
        corridor_right_boundary_points=right,
        exit_left_boundary_points=exit_left,
        exit_right_boundary_points=exit_right,
        corridor_open_ends=bool(corridor.get("corridor_open_ends", False)),
    )

    spawn = data.get("ego_spawn", {})
    if spawn:
        data = dict(data)
        data["origin"] = dict(data.get("origin", {}))
        data["origin"]["explicit_spawn_bounds"] = [
            float(spawn["x_min"]) / scale,
            float(spawn["x_max"]) / scale,
            float(spawn["y_min"]) / scale,
            float(spawn["y_max"]) / scale,
        ]
    direction = data.get("travel_direction", {"x": 1.0, "y": 0.0})
    data.setdefault("carla", {})["road_heading_deg"] = math.degrees(
        math.atan2(float(direction["y"]), float(direction["x"]))
    )
    return _assemble(record, data, workzone)


def _assemble(record, data: dict, workzone: WorkZoneConfig) -> ScenarioConfig:
    origin_raw = dict(data.get("origin", {}))
    explicit = origin_raw.pop("explicit_spawn_bounds", None)
    spawn_points_raw = origin_raw.pop("spawn_points", [])
    spawn_points = []
    for point in spawn_points_raw:
        location_cm = point["location_cm"]
        rotation = point.get("rotation_deg", {})
        spawn_points.append(EgoSpawnPointConfig(
            x=float(location_cm[0]) / 100.0,
            y=float(location_cm[1]) / 100.0,
            z=float(point.get("z_m", 0.5)),
            pitch_deg=float(rotation.get("pitch", 0.0)),
            roll_deg=float(rotation.get("roll", 0.0)),
            yaw_deg=float(rotation.get("yaw", 0.0)),
        ))
    origin = OriginConfig(
        near_dist_m=float(origin_raw.get("near_dist_m", 45.0)),
        far_dist_m=float(origin_raw.get("far_dist_m", 55.0)),
        eval_distances_m=tuple(origin_raw.get("eval_distances_m", [45.0, 50.0, 55.0])),
        explicit_spawn_bounds=tuple(explicit) if explicit else None,
        spawn_points=tuple(spawn_points),
    )
    dest_raw = data.get("destination", {})
    finish_line = None
    if dest_raw.get("line_cm"):
        line_cm = dest_raw["line_cm"]
        finish_line = FinishLineConfig(
            start=(float(line_cm[0][0]) / 100.0, float(line_cm[0][1]) / 100.0),
            end=(float(line_cm[1][0]) / 100.0, float(line_cm[1][1]) / 100.0),
            crossing_reference=str(dest_raw.get("crossing_reference", "vehicle_center")),
            crossing_heading_deg=(
                float(dest_raw["crossing_heading_deg"])
                if dest_raw.get("crossing_heading_deg") is not None
                else None
            ),
        )
    destination = DestinationConfig(
        near_dist_m=float(dest_raw.get("near_dist_m", 15.0)),
        far_dist_m=float(dest_raw.get("far_dist_m", 25.0)),
        eval_distances_m=tuple(dest_raw.get("eval_distances_m", [15.0, 20.0, 25.0])),
        success_dist_m=float(dest_raw.get("success_dist_m", 2.0)),
        success_heading_deg=float(dest_raw.get("success_heading_deg", 30.0)),
        finish_line=finish_line,
    )
    carla_raw = dict(data.get("carla", {}))
    carla_raw.setdefault("road_heading_deg", 0.0)
    carla_raw["town"] = record.town
    carla_cfg = CARLAConfig(**carla_raw)
    episode = EpisodeConfig(**data.get("episode", {}))
    sumo_cfg = None
    if record.traffic_backend == "carla_sumo":
        sumo_raw = dict(data.get("sumo") or {})
        if not sumo_raw.get("net_file") or not sumo_raw.get("route_file"):
            raise ValueError(f"{record.setting_id} requires SUMO network and route files")
        sumo_raw["bg_routes"] = tuple(sumo_raw.get("bg_routes", []))
        sumo_cfg = SUMOConfig(**sumo_raw)
        if abs(sumo_cfg.step_length - episode.sim_dt) > 1e-9:
            raise ValueError(f"{record.setting_id}: SUMO step length must equal episode sim_dt")
    jaywalker_raw = data.get("jaywalker")
    jaywalker_cfg = None
    if jaywalker_raw:
        jaywalker_raw = dict(jaywalker_raw)
        for key in ("spawn", "collision_target", "disappear", "trigger_anchor"):
            jaywalker_raw[key] = tuple(float(v) for v in jaywalker_raw[key])
        jaywalker_raw["trigger_distances_m"] = tuple(
            float(v) for v in jaywalker_raw.get(
                "trigger_distances_m", (45.0, 30.0, 15.0)
            )
        )
        jaywalker_cfg = JaywalkerConfig(**jaywalker_raw)
    observation_raw = data.get("observation", {})
    reward_raw = data.get("reward", {})
    return ScenarioConfig(
        scenario_id=record.scenario_id,
        wz_id=record.wz_id,
        layout_id=record.layout_id,
        setting_id=record.setting_id,
        traffic_backend=record.traffic_backend,
        geometry_profile=record.geometry_profile,
        reward_profile=record.reward_profile,
        workzone=workzone,
        origin=origin,
        destination=destination,
        carla=carla_cfg,
        sumo=sumo_cfg,
        jaywalker=jaywalker_cfg,
        observation=ObservationConfig(
            history_frames=int(observation_raw.get("history_frames", 1)),
            max_agents=int(observation_raw.get("max_agents", 8)),
            perception_radius_m=float(
                observation_raw.get("perception_radius_m", 50.0)
            ),
            max_ego_speed_mps=float(
                observation_raw.get("max_ego_speed_mps", 13.89)
            ),
            max_acceleration_mps2=float(
                observation_raw.get("max_acceleration_mps2", 8.0)
            ),
            yaw_rate_limit_deg_s=float(
                observation_raw.get("yaw_rate_limit_deg_s", 90.0)
            ),
            relative_speed_scale_mps=float(
                observation_raw.get("relative_speed_scale_mps", 30.0)
            ),
            missing_actor_ttl_steps=int(
                observation_raw.get("missing_actor_ttl_steps", 2)
            ),
        ),
        reward=RewardConfig(
            failure=float(reward_raw.get("failure", -100.0)),
            success=float(reward_raw.get("success", 100.0)),
            timeout=float(reward_raw.get("timeout", -20.0)),
            progress_budget=float(reward_raw.get("progress_budget", 20.0)),
            step_cost=float(reward_raw.get("step_cost", -0.01)),
        ),
        episode=episode,
        ego_role=data.get("ego_role", "hero"),
    )


def list_scenarios(*, include_blocked: bool = False) -> list[str]:
    settings: list[str] = []
    for scenario_id in scenario_ids():
        settings.extend(list_setting_ids(scenario_id, runnable_only=not include_blocked))
    return settings
