"""Translate canonical scenario configuration into PPO V2 geometry.

The adapter deliberately has no import-time dependency on CARLA or Shapely.
Rectangle and S3 geometry can therefore be validated in an ordinary Python
environment.  The exact curved S2 polygon is the one exception: constructing
it requires a live CARLA map and the existing Shapely-backed lane sampler.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from logic.termination_checker_v2 import FinishLineV2, S3DrivableAreaV2

if TYPE_CHECKING:
    from config.scenario_config import ScenarioConfig


Point2DV2 = Tuple[float, float]


@dataclass(frozen=True)
class ForbiddenAreaGeometryV2:
    """A renderable forbidden area plus optional S2 diagnostic metadata."""

    polygon: Tuple[Point2DV2, ...]
    holes: Tuple[Tuple[Point2DV2, ...], ...] = ()
    centerline: Tuple[Point2DV2, ...] = ()
    source: str = "config_rectangle"
    actual_half_width_m: Optional[float] = None

    def __post_init__(self) -> None:
        polygon = _ring_v2(self.polygon, "polygon")
        holes = tuple(
            _ring_v2(ring, f"holes[{index}]")
            for index, ring in enumerate(self.holes)
        )
        centerline = tuple(
            _point_v2(point, f"centerline[{index}]")
            for index, point in enumerate(self.centerline)
        )
        half_width = self.actual_half_width_m
        if half_width is not None:
            half_width = float(half_width)
            if not math.isfinite(half_width) or half_width <= 0.0:
                raise ValueError("actual_half_width_m must be finite and positive")
        object.__setattr__(self, "polygon", polygon)
        object.__setattr__(self, "holes", holes)
        object.__setattr__(self, "centerline", centerline)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "actual_half_width_m", half_width)


def finish_line_from_config_v2(config: "ScenarioConfig") -> FinishLineV2:
    """Return the finite finish line authored for ``config``."""
    finish = config.destination.finish_line
    if finish is None:
        raise ValueError(f"{config.setting_id} has no authored finish line")
    return FinishLineV2(tuple(finish.start), tuple(finish.end))


def build_s3_drivable_area_from_config_v2(
    config: "ScenarioConfig",
    origin_index: int,
) -> S3DrivableAreaV2:
    """Build the selected S3 origin's entry/corridor/exit union."""
    if _scenario_id_v2(config) != "s3":
        raise ValueError("S3 drivable-area geometry may only be built for S3")
    if isinstance(origin_index, bool) or int(origin_index) != origin_index:
        raise ValueError("origin_index must be an integer")
    origin_index = int(origin_index)
    spawn_points = tuple(config.origin.spawn_points)
    if not 0 <= origin_index < len(spawn_points):
        raise IndexError(
            f"origin_index {origin_index} is outside 0..{len(spawn_points) - 1} "
            f"for {config.setting_id}"
        )

    left = config.workzone.corridor_left_boundary_points
    right = config.workzone.corridor_right_boundary_points
    if not left or not right:
        raise ValueError(
            f"{config.setting_id} has no ordered S3 left/right corridor boundaries"
        )
    origin_cfg = spawn_points[origin_index]
    return S3DrivableAreaV2.build(
        origin=(origin_cfg.x, origin_cfg.y),
        origin_heading_deg=origin_cfg.yaw_deg,
        left_boundary=left,
        right_boundary=right,
        finish_line=finish_line_from_config_v2(config),
        exit_left_boundary=config.workzone.exit_left_boundary_points,
        exit_right_boundary=config.workzone.exit_right_boundary_points,
        boundary_tolerance_m=config.workzone.boundary_tolerance_m,
    )


def build_forbidden_area_from_config_v2(
    config: "ScenarioConfig",
    *,
    carla_map: object | None = None,
) -> ForbiddenAreaGeometryV2:
    """Build a forbidden rectangle or the exact lane-following S2 polygon.

    ``carla_map`` is required only for S2.  Imports of CARLA and the existing
    Shapely-backed builder stay inside that branch so every other scenario is
    usable in a lightweight test environment.
    """
    scenario_id = _scenario_id_v2(config)
    if scenario_id == "s3":
        raise ValueError("S3 has a drivable union, not a forbidden-area polygon")
    if scenario_id not in {"s1", "s2", "s4", "s5", "s6"}:
        raise ValueError(f"Unsupported scenario: {scenario_id!r}")

    if scenario_id != "s2":
        workzone = config.workzone
        x_min, x_max = sorted((float(workzone.x_min), float(workzone.x_max)))
        y_min, y_max = sorted((float(workzone.y_min), float(workzone.y_max)))
        return ForbiddenAreaGeometryV2(
            polygon=(
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ),
            source="config_rectangle",
        )

    if carla_map is None:
        raise ValueError(
            "S2 exact curved forbidden geometry requires the loaded CARLA map"
        )
    polygon_config = config.workzone.polygon_config
    if polygon_config is None:
        raise ValueError(f"{config.setting_id} has no S2 polygon_config")

    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "S2 exact curved geometry requires the CARLA PythonAPI in this "
            "Python environment"
        ) from exc
    try:
        from logic.workzone_geometry import build_workzone_polygon
    except ImportError as exc:
        raise RuntimeError(
            "S2 exact curved geometry requires both the CARLA PythonAPI and "
            "Shapely (install Shapely>=1.8,<3)"
        ) from exc

    try:
        result = build_workzone_polygon(
            carla_map,
            carla.Location(
                x=polygon_config.head_x,
                y=polygon_config.head_y,
                z=polygon_config.head_z,
            ),
            carla.Location(
                x=polygon_config.tail_x,
                y=polygon_config.tail_y,
                z=polygon_config.tail_z,
            ),
            sample_spacing_m=polygon_config.sample_spacing_m,
            half_width_m=polygon_config.half_width_m,
            margin_m=polygon_config.margin_m,
            expected_road_id=polygon_config.expected_road_id,
            expected_lane_id=polygon_config.expected_lane_id,
        )
    except ImportError as exc:
        raise RuntimeError(
            "S2 exact curved geometry requires Shapely in the same Python "
            "environment as the CARLA PythonAPI (install Shapely>=1.8,<3)"
        ) from exc

    polygon = result.polygon
    if getattr(polygon, "is_empty", True):
        raise ValueError(f"{config.setting_id} produced an empty S2 polygon")
    if not getattr(polygon, "is_valid", False):
        raise ValueError(f"{config.setting_id} produced an invalid S2 polygon")
    if getattr(polygon, "geom_type", None) != "Polygon":
        raise ValueError(
            f"{config.setting_id} produced {getattr(polygon, 'geom_type', 'unknown')} "
            "instead of one exact S2 Polygon"
        )

    exterior = tuple((float(x), float(y)) for x, y, *_ in polygon.exterior.coords)
    holes = tuple(
        tuple((float(x), float(y)) for x, y, *_ in interior.coords)
        for interior in polygon.interiors
    )
    return ForbiddenAreaGeometryV2(
        polygon=exterior,
        holes=holes,
        centerline=tuple(
            (float(point[0]), float(point[1])) for point in result.centerline_pts
        ),
        source="s2_exact_curved_lane_buffer",
        actual_half_width_m=float(result.actual_half_width_m),
    )


def _scenario_id_v2(config: "ScenarioConfig") -> str:
    return str(config.scenario_id).strip().lower()


def _point_v2(point: Sequence[float], name: str) -> Point2DV2:
    if len(point) != 2:
        raise ValueError(f"{name} must contain exactly x and y")
    normalized = (float(point[0]), float(point[1]))
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _ring_v2(points: Sequence[Sequence[float]], name: str) -> Tuple[Point2DV2, ...]:
    ring = tuple(_point_v2(point, f"{name}[{index}]") for index, point in enumerate(points))
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError(f"{name} must contain at least three distinct vertices")
    if len(set(ring)) < 3:
        raise ValueError(f"{name} must contain at least three distinct vertices")
    twice_area = sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(ring, ring[1:] + ring[:1])
    )
    if abs(twice_area) <= 1e-9:
        raise ValueError(f"{name} must have non-zero area")
    return ring


__all__ = [
    "ForbiddenAreaGeometryV2",
    "Point2DV2",
    "build_forbidden_area_from_config_v2",
    "build_s3_drivable_area_from_config_v2",
    "finish_line_from_config_v2",
]
