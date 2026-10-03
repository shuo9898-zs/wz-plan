"""Scenario-aware terminal judgement for PPO V2.

S2 treats its curved work zone as forbidden.  S3 uses the opposite rule: its
episode-specific wide entry approach, authored corridor, and exit polygon
form one drivable-area union, and leaving that union is the violation.  S4 is
the only jaywalker scenario and has no SUMO collision outcome.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple

from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from logic.reward_v2 import (
    COLLISION_JAYWALKER_V2,
    COLLISION_SUMO_VEHICLE_V2,
    GOAL_REACHED_V2,
    OFF_ROAD_V2,
    RUNNING_V2,
    TIMEOUT_V2,
    WORKZONE_VIOLATION_V2,
)


TERMINATION_CONTRACT_VERSION_V2 = "scenario_geometry_swept_ego_obb_v2"
S3_ENTRY_HALF_WIDTH_M_V2 = 3.0
S3_ENTRY_REAR_MARGIN_M_V2 = 3.0
Point2DV2 = Tuple[float, float]
Footprint2DV2 = Tuple[Point2DV2, ...]
CollisionKindV2 = Literal["jaywalker", "sumo_vehicle", "workzone_object"]


@dataclass(frozen=True)
class TerminalDecisionV2:
    terminated: bool
    reason: str


@dataclass(frozen=True)
class FinishLineV2:
    first: Point2DV2
    second: Point2DV2

    def __post_init__(self) -> None:
        first = _point_v2(self.first, "finish_line.first")
        second = _point_v2(self.second, "finish_line.second")
        if _distance_sq_v2(first, second) <= 1e-12:
            raise ValueError("Finish-line endpoints must be distinct")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)

    @property
    def midpoint(self) -> Point2DV2:
        return (
            0.5 * (self.first[0] + self.second[0]),
            0.5 * (self.first[1] + self.second[1]),
        )

    def crossed_forward(
        self,
        previous: Point2DV2,
        current: Point2DV2,
        *,
        origin: Point2DV2,
        epsilon: float = 1e-9,
    ) -> bool:
        """Require an origin-side to destination-side finite-segment crossing."""
        return self.forward_crossing_point(
            previous,
            current,
            origin=origin,
            epsilon=epsilon,
        ) is not None

    def forward_crossing_point(
        self,
        previous: Point2DV2,
        current: Point2DV2,
        *,
        origin: Point2DV2,
        epsilon: float = 1e-9,
    ) -> Optional[Point2DV2]:
        """Return the finite forward crossing point, or ``None``.

        Exposing the exact crossing lets S3 verify that the trajectory remains
        inside its drivable union up to the finish line while still allowing
        the sampled current position to land just beyond the exit polygon.
        """
        previous = _point_v2(previous, "previous")
        current = _point_v2(current, "current")
        origin = _point_v2(origin, "origin")
        line_x = self.second[0] - self.first[0]
        line_y = self.second[1] - self.first[1]
        normal = (-line_y, line_x)
        midpoint = self.midpoint
        toward_finish = (
            (midpoint[0] - origin[0]) * normal[0]
            + (midpoint[1] - origin[1]) * normal[1]
        )
        if abs(toward_finish) <= epsilon:
            raise ValueError("Origin cannot lie on the finish-line axis")
        if toward_finish < 0.0:
            normal = (-normal[0], -normal[1])

        previous_side = (
            (previous[0] - self.first[0]) * normal[0]
            + (previous[1] - self.first[1]) * normal[1]
        )
        current_side = (
            (current[0] - self.first[0]) * normal[0]
            + (current[1] - self.first[1]) * normal[1]
        )
        if previous_side >= -epsilon or current_side < -epsilon:
            return None
        denominator = previous_side - current_side
        if abs(denominator) <= epsilon:
            return None
        fraction = previous_side / denominator
        crossing = (
            previous[0] + fraction * (current[0] - previous[0]),
            previous[1] + fraction * (current[1] - previous[1]),
        )
        if not _point_on_segment_v2(
            crossing,
            self.first,
            self.second,
            tolerance=1e-7,
        ):
            return None
        return crossing


@dataclass(frozen=True)
class S3DrivableAreaV2:
    """Optional wide entry approach + cone corridor + authored exit polygon.

    ``entry_triangle`` is retained as the serialized/API compatibility name,
    but newly built geometry uses a four-point approach polygon.  Its rear
    cross-section is six metres wide and sits behind the ego spawn, so the
    vehicle starts inside a useful area instead of on a zero-width apex.
    """

    entry_triangle: Tuple[Point2DV2, ...]
    corridor_polygon: Tuple[Point2DV2, ...]
    # Historical straight layouts use a four-point connector.  Curved exits
    # use the explicit lane-following polygon instead.  Exactly one must be
    # present so existing S3 WZ1/WZ2 geometry remains unchanged.
    exit_quadrilateral: Tuple[Point2DV2, ...] = ()
    boundary_tolerance_m: float = 0.1
    exit_corridor_polygon: Tuple[Point2DV2, ...] = ()
    _polygons: Tuple[Tuple[Point2DV2, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _geometry: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entry = tuple(_point_v2(point, "entry_triangle") for point in self.entry_triangle)
        corridor = tuple(
            _point_v2(point, "corridor_polygon") for point in self.corridor_polygon
        )
        exit_quad = tuple(
            _point_v2(point, "exit_quadrilateral")
            for point in self.exit_quadrilateral
        )
        exit_corridor = tuple(
            _point_v2(point, "exit_corridor_polygon")
            for point in self.exit_corridor_polygon
        )
        tolerance = float(self.boundary_tolerance_m)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("boundary_tolerance_m must be finite and non-negative")
        if len(entry) not in (0, 3, 4):
            raise ValueError(
                "entry_triangle compatibility field must be empty or contain "
                "a three-point legacy triangle or four-point approach polygon"
            )
        if len(corridor) < 4:
            raise ValueError("corridor_polygon must contain at least four points")
        if bool(exit_quad) == bool(exit_corridor):
            raise ValueError(
                "Exactly one of exit_quadrilateral or exit_corridor_polygon "
                "must be provided"
            )
        if exit_quad and len(exit_quad) != 4:
            raise ValueError("exit_quadrilateral must contain exactly four points")
        if exit_corridor and len(exit_corridor) < 4:
            raise ValueError(
                "exit_corridor_polygon must contain at least four points"
            )
        exit_polygon = exit_corridor or exit_quad

        if entry:
            _validate_simple_polygon_v2(entry, "entry_approach_polygon")
        _validate_simple_polygon_v2(corridor, "corridor_polygon")
        _validate_simple_polygon_v2(
            exit_polygon,
            "exit_corridor_polygon" if exit_corridor else "exit_quadrilateral",
        )
        if entry:
            _validate_clean_polygon_join_v2(
                entry,
                corridor,
                shared_start=entry[1],
                shared_end=entry[2],
                name="entry/corridor union",
            )
        _validate_clean_polygon_join_v2(
            corridor,
            exit_polygon,
            shared_start=(exit_corridor[0] if exit_corridor else exit_quad[0]),
            shared_end=(exit_corridor[-1] if exit_corridor else exit_quad[1]),
            name="corridor/exit union",
        )
        if entry:
            _validate_disjoint_polygon_interiors_v2(
                entry,
                exit_polygon,
                name="entry/exit union",
            )

        object.__setattr__(self, "entry_triangle", entry)
        object.__setattr__(self, "corridor_polygon", corridor)
        object.__setattr__(self, "exit_quadrilateral", exit_quad)
        object.__setattr__(self, "exit_corridor_polygon", exit_corridor)
        object.__setattr__(self, "boundary_tolerance_m", tolerance)
        polygons = ((entry,) if entry else ()) + (corridor, exit_polygon)
        object.__setattr__(self, "_polygons", polygons)
        geometry = unary_union(tuple(Polygon(polygon) for polygon in polygons))
        if tolerance > 0.0:
            geometry = geometry.buffer(tolerance)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("S3 drivable-area union is empty or invalid")
        object.__setattr__(self, "_geometry", geometry)

    @property
    def exit_polygon(self) -> Tuple[Point2DV2, ...]:
        """The active exit geometry, independent of straight/curved form."""
        return self.exit_corridor_polygon or self.exit_quadrilateral

    @property
    def entry_approach_polygon(self) -> Tuple[Point2DV2, ...]:
        """The active entry polygon under a non-legacy descriptive name."""
        return self.entry_triangle

    @property
    def geometry(self) -> object:
        """Cached Shapely union used for exact ego-footprint containment."""
        return self._geometry

    @classmethod
    def build(
        cls,
        *,
        origin: Point2DV2,
        left_boundary: Sequence[Point2DV2],
        right_boundary: Sequence[Point2DV2],
        finish_line: FinishLineV2,
        exit_left_boundary: Optional[Sequence[Point2DV2]] = None,
        exit_right_boundary: Optional[Sequence[Point2DV2]] = None,
        origin_heading_deg: Optional[float] = None,
        entry_half_width_m: float = S3_ENTRY_HALF_WIDTH_M_V2,
        entry_rear_margin_m: float = S3_ENTRY_REAR_MARGIN_M_V2,
        boundary_tolerance_m: float = 0.1,
    ) -> "S3DrivableAreaV2":
        origin = _point_v2(origin, "origin")
        left = [_point_v2(point, "left_boundary") for point in left_boundary]
        right = [_point_v2(point, "right_boundary") for point in right_boundary]
        if len(left) != len(right) or len(left) < 2:
            raise ValueError(
                "S3 corridor requires matching left/right boundaries with at least two points"
            )
        if not math.isfinite(float(boundary_tolerance_m)) or boundary_tolerance_m < 0.0:
            raise ValueError("boundary_tolerance_m must be finite and non-negative")
        entry_half_width_m = float(entry_half_width_m)
        entry_rear_margin_m = float(entry_rear_margin_m)
        if not math.isfinite(entry_half_width_m) or entry_half_width_m <= 0.0:
            raise ValueError("entry_half_width_m must be finite and positive")
        if not math.isfinite(entry_rear_margin_m) or entry_rear_margin_m <= 0.0:
            raise ValueError("entry_rear_margin_m must be finite and positive")

        # Some authored boundaries may be stored in either direction.  The
        # endpoint pair nearest the selected episode origin is always entry.
        first_distance = _distance_sq_v2(origin, _midpoint_v2(left[0], right[0]))
        last_distance = _distance_sq_v2(origin, _midpoint_v2(left[-1], right[-1]))
        if last_distance < first_distance:
            left.reverse()
            right.reverse()

        corridor = tuple(left + list(reversed(right)))
        # A centre just inside the first corridor cross-section may still have
        # its rear bumper outside.  Keep the same non-overlapping approach pad
        # for every origin less than ``entry_rear_margin_m`` downstream of the
        # entry seam; deeper origins need no extra component.  Outside origins
        # always receive the pad.  Its rear edge is three metres behind the
        # spawn and extends three metres to each side of the ego heading.
        origin_in_corridor = _point_in_polygon_v2(origin, corridor)
        entry_midpoint = _midpoint_v2(left[0], right[0])
        next_midpoint = _midpoint_v2(left[1], right[1])
        entry_direction = (
            next_midpoint[0] - entry_midpoint[0],
            next_midpoint[1] - entry_midpoint[1],
        )
        entry_direction_length = math.hypot(*entry_direction)
        if entry_direction_length <= 1e-9:
            raise ValueError("S3 corridor has a degenerate first longitudinal segment")
        entry_depth_m = (
            (origin[0] - entry_midpoint[0]) * entry_direction[0]
            + (origin[1] - entry_midpoint[1]) * entry_direction[1]
        ) / entry_direction_length
        needs_entry_approach = (
            not origin_in_corridor or entry_depth_m < entry_rear_margin_m
        )
        entry = (
            _build_entry_approach_polygon_v2(
                origin=origin,
                origin_heading_deg=origin_heading_deg,
                entry_first=left[0],
                entry_second=right[0],
                corridor_polygon=corridor,
                half_width_m=entry_half_width_m,
                rear_margin_m=entry_rear_margin_m,
            )
            if needs_entry_approach
            else ()
        )

        if (exit_left_boundary is None) != (exit_right_boundary is None):
            raise ValueError(
                "S3 curved exit requires both left and right boundary points"
            )
        if exit_left_boundary is not None:
            exit_left = [
                _point_v2(point, "exit_left_boundary")
                for point in exit_left_boundary
            ]
            exit_right = [
                _point_v2(point, "exit_right_boundary")
                for point in exit_right_boundary or ()
            ]
            if len(exit_left) < 2 or len(exit_right) < 2:
                raise ValueError(
                    "S3 curved exit boundaries must each contain at least two points"
                )
            if not _same_point_v2(exit_left[0], left[-1]) or not _same_point_v2(
                exit_right[0], right[-1]
            ):
                raise ValueError(
                    "S3 curved exit boundaries must start at the final authored "
                    "left/right corridor points"
                )
            for name, point in (
                ("exit_left_boundary final point", exit_left[-1]),
                ("exit_right_boundary final point", exit_right[-1]),
            ):
                if not _point_on_segment_v2(
                    point,
                    finish_line.first,
                    finish_line.second,
                    tolerance=1e-4,
                ):
                    raise ValueError(f"{name} must lie on the finite finish line")
            return cls(
                entry_triangle=entry,
                corridor_polygon=corridor,
                exit_corridor_polygon=tuple(
                    exit_left + list(reversed(exit_right))
                ),
                boundary_tolerance_m=float(boundary_tolerance_m),
            )

        # There are exactly two ways to connect the corridor cross-section to
        # the finish line.  Distance alone is ambiguous for perpendicular
        # sections (the real s3/wz3/a is an exact tie), so accept only simple,
        # non-degenerate, cleanly joined candidates before using connector
        # length as a deterministic tie-breaker.
        finish_first, finish_second = finish_line.first, finish_line.second
        candidates = (
            (left[-1], right[-1], finish_first, finish_second),
            (left[-1], right[-1], finish_second, finish_first),
        )
        valid: list[tuple[float, S3DrivableAreaV2]] = []
        errors: list[str] = []
        for exit_quad in candidates:
            try:
                area = cls(
                    entry_triangle=entry,
                    corridor_polygon=corridor,
                    exit_quadrilateral=exit_quad,
                    boundary_tolerance_m=float(boundary_tolerance_m),
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            connector_cost = (
                _distance_sq_v2(right[-1], exit_quad[2])
                + _distance_sq_v2(left[-1], exit_quad[3])
            )
            valid.append((connector_cost, area))
        if not valid:
            raise ValueError(
                "S3 exit quadrilateral has no simple, connected endpoint pairing: "
                + "; ".join(errors)
            )
        valid.sort(key=lambda item: item[0])
        return valid[0][1]

    def contains(self, point: Point2DV2) -> bool:
        point = _point_v2(point, "point")
        return any(
            _point_in_or_near_polygon_v2(
                point,
                polygon,
                tolerance=self.boundary_tolerance_m,
            )
            for polygon in self._polygons
        )

    def covers_segment(self, start: Point2DV2, end: Point2DV2) -> bool:
        """Return whether the complete segment stays in the drivable union."""
        start = _point_v2(start, "segment.start")
        end = _point_v2(end, "segment.end")
        parameters = {0.0, 1.0}
        for polygon in self._polygons:
            for edge_start, edge_end in _polygon_edges_v2(polygon):
                parameters.update(
                    _segment_intersection_parameters_v2(
                        start,
                        end,
                        edge_start,
                        edge_end,
                    )
                )
                if self.boundary_tolerance_m > 0.0:
                    parameters.update(
                        _segment_capsule_transition_parameters_v2(
                            start,
                            end,
                            edge_start,
                            edge_end,
                            radius=self.boundary_tolerance_m,
                        )
                    )
        ordered = sorted(parameters)
        probes = list(ordered)
        probes.extend(
            0.5 * (first + second)
            for first, second in zip(ordered, ordered[1:])
            if second - first > 1e-12
        )
        return all(
            self.contains(
                (
                    start[0] + parameter * (end[0] - start[0]),
                    start[1] + parameter * (end[1] - start[1]),
                )
            )
            for parameter in probes
        )

    def covers_footprint(self, footprint: Sequence[Point2DV2]) -> bool:
        """Return whether the complete 2-D ego footprint is drivable."""
        return bool(self._geometry.covers(_footprint_shape_v2(footprint)))


class TerminationCheckerV2:
    """Return exactly one task outcome using scenario-specific geometry."""

    def __init__(
        self,
        *,
        scenario_id: str,
        origin: Point2DV2,
        finish_line: FinishLineV2,
        max_episode_steps: int,
        forbidden_polygon: Optional[Sequence[Point2DV2]] = None,
        s3_drivable_area: Optional[S3DrivableAreaV2] = None,
        boundary_tolerance_m: float = 0.1,
    ) -> None:
        scenario = str(scenario_id).strip().lower()
        if scenario not in {"s1", "s2", "s3", "s4", "s5", "s6"}:
            raise ValueError(f"Unsupported scenario: {scenario_id!r}")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not math.isfinite(float(boundary_tolerance_m)) or boundary_tolerance_m < 0.0:
            raise ValueError("boundary_tolerance_m must be finite and non-negative")
        self.scenario_id = scenario
        self.origin = _point_v2(origin, "origin")
        self.finish_line = finish_line
        self.max_episode_steps = int(max_episode_steps)
        self.boundary_tolerance_m = float(boundary_tolerance_m)
        self.forbidden_polygon = (
            tuple(_point_v2(point, "forbidden_polygon") for point in forbidden_polygon)
            if forbidden_polygon is not None
            else None
        )
        self.s3_drivable_area = s3_drivable_area
        if scenario == "s3" and s3_drivable_area is None:
            raise ValueError("S3 requires s3_drivable_area")
        if scenario != "s3" and s3_drivable_area is not None:
            raise ValueError("s3_drivable_area may only be used by S3")
        if scenario == "s2" and self.forbidden_polygon is None:
            raise ValueError("S2 requires its curved forbidden polygon")
        if self.forbidden_polygon is not None and len(self.forbidden_polygon) < 3:
            raise ValueError("forbidden_polygon requires at least three points")

        self._forbidden_geometry = None
        if self.forbidden_polygon is not None:
            forbidden_geometry = Polygon(self.forbidden_polygon)
            if not forbidden_geometry.is_valid or forbidden_geometry.is_empty:
                raise ValueError("forbidden_polygon must form a valid non-empty polygon")
            if self.boundary_tolerance_m > 0.0:
                forbidden_geometry = forbidden_geometry.buffer(
                    self.boundary_tolerance_m
                )
            self._forbidden_geometry = forbidden_geometry

        self._s3_footprint_drivable_geometry = None
        if self.s3_drivable_area is not None:
            # Success is defined by the vehicle centre crossing the finite
            # finish segment.  Open only that cap in the forward direction so
            # the nose may pass the line before the centre without weakening
            # either side boundary.
            exit_strip = _finish_exit_strip_v2(
                self.finish_line,
                origin=self.origin,
                length_m=100.0,
            )
            self._s3_footprint_drivable_geometry = unary_union(
                (self.s3_drivable_area.geometry, Polygon(exit_strip))
            )

    def judge(
        self,
        *,
        previous_position: Point2DV2,
        current_position: Point2DV2,
        episode_step: int,
        collision: Optional[CollisionKindV2] = None,
        off_road: bool = False,
        ego_footprint: Optional[Sequence[Point2DV2]] = None,
    ) -> TerminalDecisionV2:
        previous = _point_v2(previous_position, "previous_position")
        current = _point_v2(current_position, "current_position")
        footprint = (
            _footprint_v2(ego_footprint, "ego_footprint")
            if ego_footprint is not None
            else None
        )
        if episode_step < 0:
            raise ValueError("episode_step must be non-negative")

        # Safety collisions always beat a simultaneous finish-line crossing.
        if collision is not None:
            if collision == "jaywalker":
                if self.scenario_id != "s4":
                    raise ValueError("Jaywalker collisions are valid only in S4")
                return TerminalDecisionV2(True, COLLISION_JAYWALKER_V2)
            if collision == "sumo_vehicle":
                if self.scenario_id == "s4":
                    raise ValueError("S4 is CARLA-only and has no SUMO vehicles")
                return TerminalDecisionV2(True, COLLISION_SUMO_VEHICLE_V2)
            if collision == "workzone_object":
                return TerminalDecisionV2(True, WORKZONE_VIOLATION_V2)
            raise ValueError(f"Unsupported collision kind: {collision!r}")

        # CARLA's road/lane safety verdict is authoritative in every
        # scenario, including a simultaneous otherwise-valid S3 crossing.
        if off_road:
            return TerminalDecisionV2(True, OFF_ROAD_V2)

        crossing = self.finish_line.forward_crossing_point(
            previous,
            current,
            origin=self.origin,
        )

        # Non-S3 safety geometry precedes success.  In S3 the current sample
        # may legitimately land just beyond the finish edge, so success is
        # allowed first only when the complete pre-crossing sweep is inside
        # the authored drivable union.
        if self.scenario_id == "s3":
            assert self.s3_drivable_area is not None
            if crossing is not None:
                if self.s3_drivable_area.covers_segment(
                    previous, crossing
                ) and (
                    footprint is None
                    or not self._workzone_violated(
                        previous,
                        current,
                        ego_footprint=footprint,
                    )
                ):
                    return TerminalDecisionV2(True, GOAL_REACHED_V2)
                return TerminalDecisionV2(True, WORKZONE_VIOLATION_V2)
        else:
            if self._workzone_violated(
                previous,
                current,
                ego_footprint=footprint,
            ):
                return TerminalDecisionV2(True, WORKZONE_VIOLATION_V2)
            if crossing is not None:
                return TerminalDecisionV2(True, GOAL_REACHED_V2)

        if self._workzone_violated(
            previous,
            current,
            ego_footprint=footprint,
        ):
            return TerminalDecisionV2(True, WORKZONE_VIOLATION_V2)
        if episode_step >= self.max_episode_steps:
            return TerminalDecisionV2(True, TIMEOUT_V2)
        return TerminalDecisionV2(False, RUNNING_V2)

    def _workzone_violated(
        self,
        previous: Point2DV2,
        current: Point2DV2,
        *,
        ego_footprint: Optional[Footprint2DV2] = None,
    ) -> bool:
        if self.scenario_id == "s3":
            assert self.s3_drivable_area is not None
            if ego_footprint is not None:
                assert self._s3_footprint_drivable_geometry is not None
                return not self._s3_footprint_drivable_geometry.covers(
                    _footprint_shape_v2(ego_footprint)
                )
            return not self.s3_drivable_area.contains(current)
        if self.forbidden_polygon is None:
            return False
        if ego_footprint is not None:
            assert self._forbidden_geometry is not None
            if self._forbidden_geometry.intersects(
                _footprint_shape_v2(ego_footprint)
            ):
                return True
        return _point_in_or_near_polygon_v2(
            current,
            self.forbidden_polygon,
            tolerance=self.boundary_tolerance_m,
        ) or _segment_touches_polygon_v2(
            previous,
            current,
            self.forbidden_polygon,
        )


def _point_v2(value: Point2DV2, name: str) -> Point2DV2:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly x and y")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _footprint_v2(
    value: Sequence[Point2DV2],
    name: str,
) -> Footprint2DV2:
    footprint = tuple(
        _point_v2(point, f"{name}[{index}]")
        for index, point in enumerate(value)
    )
    if len(footprint) < 3:
        raise ValueError(f"{name} must contain at least three points")
    _validate_simple_polygon_v2(footprint, name)
    return footprint


def _footprint_shape_v2(footprint: Sequence[Point2DV2]) -> Polygon:
    points = _footprint_v2(footprint, "ego_footprint")
    shape = Polygon(points)
    if shape.is_empty or not shape.is_valid or shape.area <= 1e-9:
        raise ValueError("ego_footprint must form a valid non-empty polygon")
    return shape


def swept_footprint_v2(
    previous: Sequence[Point2DV2],
    current: Sequence[Point2DV2],
) -> Footprint2DV2:
    """Conservative 2-D hull swept by two consecutive ego OBBs."""
    previous_points = _footprint_v2(previous, "previous_footprint")
    current_points = _footprint_v2(current, "current_footprint")
    hull = MultiPoint(previous_points + current_points).convex_hull
    if hull.geom_type != "Polygon" or hull.is_empty or not hull.is_valid:
        raise ValueError("Consecutive ego footprints do not form a valid swept hull")
    return tuple((float(x), float(y)) for x, y in tuple(hull.exterior.coords)[:-1])


def _finish_exit_strip_v2(
    finish_line: FinishLineV2,
    *,
    origin: Point2DV2,
    length_m: float,
) -> Tuple[Point2DV2, ...]:
    """Finite downstream extrusion that opens only the authored finish cap."""
    length_m = float(length_m)
    if not math.isfinite(length_m) or length_m <= 0.0:
        raise ValueError("finish exit-strip length must be finite and positive")
    first, second = finish_line.first, finish_line.second
    line_x, line_y = second[0] - first[0], second[1] - first[1]
    normal = (-line_y, line_x)
    normal_length = math.hypot(normal[0], normal[1])
    midpoint = finish_line.midpoint
    toward_finish = (
        (midpoint[0] - origin[0]) * normal[0]
        + (midpoint[1] - origin[1]) * normal[1]
    )
    if abs(toward_finish) <= 1e-9:
        raise ValueError("Origin cannot lie on the finish-line axis")
    if toward_finish < 0.0:
        normal = (-normal[0], -normal[1])
    forward = (normal[0] / normal_length, normal[1] / normal_length)
    strip = (
        first,
        second,
        (second[0] + length_m * forward[0], second[1] + length_m * forward[1]),
        (first[0] + length_m * forward[0], first[1] + length_m * forward[1]),
    )
    _validate_simple_polygon_v2(strip, "finish_exit_strip")
    return strip


def _midpoint_v2(first: Point2DV2, second: Point2DV2) -> Point2DV2:
    return (0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1]))


def _distance_sq_v2(first: Point2DV2, second: Point2DV2) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _build_entry_approach_polygon_v2(
    *,
    origin: Point2DV2,
    origin_heading_deg: Optional[float],
    entry_first: Point2DV2,
    entry_second: Point2DV2,
    corridor_polygon: Sequence[Point2DV2],
    half_width_m: float,
    rear_margin_m: float,
) -> Tuple[Point2DV2, ...]:
    """Build a clean approach pad joined only at the corridor entry seam."""
    entry_midpoint = _midpoint_v2(entry_first, entry_second)
    if origin_heading_deg is None:
        delta_x = entry_midpoint[0] - origin[0]
        delta_y = entry_midpoint[1] - origin[1]
        if math.hypot(delta_x, delta_y) <= 1e-9:
            raise ValueError(
                "Cannot infer S3 entry heading when origin equals entry midpoint"
            )
        heading_rad = math.atan2(delta_y, delta_x)
    else:
        heading = float(origin_heading_deg)
        if not math.isfinite(heading):
            raise ValueError("origin_heading_deg must be finite")
        heading_rad = math.radians(heading)

    forward = (math.cos(heading_rad), math.sin(heading_rad))
    # CARLA/Unreal uses x-forward, y-right for yaw=0.  Keep the authored
    # corridor's first/second sides aligned with driver-left/driver-right.
    driver_left = (forward[1], -forward[0])
    rear_center = (
        origin[0] - rear_margin_m * forward[0],
        origin[1] - rear_margin_m * forward[1],
    )
    rear_first = (
        rear_center[0] + half_width_m * driver_left[0],
        rear_center[1] + half_width_m * driver_left[1],
    )
    rear_second = (
        rear_center[0] - half_width_m * driver_left[0],
        rear_center[1] - half_width_m * driver_left[1],
    )
    candidates = (
        (rear_first, entry_first, entry_second, rear_second),
        (rear_first, entry_second, entry_first, rear_second),
    )
    valid: list[tuple[float, Tuple[Point2DV2, ...]]] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            _validate_simple_polygon_v2(candidate, "entry_approach_polygon")
            _validate_clean_polygon_join_v2(
                candidate,
                corridor_polygon,
                shared_start=candidate[1],
                shared_end=candidate[2],
                name="entry/corridor union",
            )
            origin_in_corridor = _point_in_polygon_v2(origin, corridor_polygon)
            origin_in_approach = _point_in_polygon_v2(origin, candidate)
            if not origin_in_corridor and not origin_in_approach:
                raise ValueError("entry approach/corridor union misses the ego spawn")
            if not origin_in_corridor:
                spawn_clearance = min(
                    _distance_point_to_segment_v2(origin, start, end)
                    for start, end in _polygon_edges_v2(candidate)
                )
                if spawn_clearance <= 1e-6:
                    raise ValueError("ego spawn remains on the approach boundary")
        except ValueError as error:
            errors.append(str(error))
            continue
        connector_cost = _distance_sq_v2(rear_first, candidate[1]) + _distance_sq_v2(
            rear_second, candidate[2]
        )
        valid.append((connector_cost, candidate))
    if not valid:
        raise ValueError(
            "S3 entry approach has no simple, connected endpoint pairing: "
            + "; ".join(errors)
        )
    valid.sort(key=lambda item: item[0])
    return valid[0][1]


def _cross_v2(first: Point2DV2, second: Point2DV2) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _point_on_segment_v2(
    point: Point2DV2,
    start: Point2DV2,
    end: Point2DV2,
    *,
    tolerance: float,
) -> bool:
    edge = (end[0] - start[0], end[1] - start[1])
    relative = (point[0] - start[0], point[1] - start[1])
    edge_length = math.hypot(edge[0], edge[1])
    if edge_length <= tolerance:
        return math.hypot(relative[0], relative[1]) <= tolerance
    if abs(_cross_v2(edge, relative)) > tolerance * edge_length:
        return False
    dot = relative[0] * edge[0] + relative[1] * edge[1]
    return -tolerance <= dot <= edge_length * edge_length + tolerance


def _point_in_polygon_v2(point: Point2DV2, polygon: Sequence[Point2DV2]) -> bool:
    inside = False
    for start, end in zip(polygon, tuple(polygon[1:]) + (polygon[0],)):
        if _point_on_segment_v2(point, start, end, tolerance=1e-9):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = start[0] + (
                (point[1] - start[1])
                * (end[0] - start[0])
                / (end[1] - start[1])
            )
            if point[0] < crossing_x:
                inside = not inside
    return inside


def _distance_point_to_segment_v2(
    point: Point2DV2,
    start: Point2DV2,
    end: Point2DV2,
) -> float:
    edge_x, edge_y = end[0] - start[0], end[1] - start[1]
    length_sq = edge_x * edge_x + edge_y * edge_y
    if length_sq <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = np_clip_v2(
        ((point[0] - start[0]) * edge_x + (point[1] - start[1]) * edge_y)
        / length_sq,
        0.0,
        1.0,
    )
    closest = (start[0] + fraction * edge_x, start[1] + fraction * edge_y)
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


def _point_in_or_near_polygon_v2(
    point: Point2DV2,
    polygon: Sequence[Point2DV2],
    *,
    tolerance: float,
) -> bool:
    if _point_in_polygon_v2(point, polygon):
        return True
    if tolerance <= 0.0:
        return False
    return any(
        _distance_point_to_segment_v2(point, start, end) <= tolerance
        for start, end in zip(polygon, tuple(polygon[1:]) + (polygon[0],))
    )


def _orientation_v2(a: Point2DV2, b: Point2DV2, c: Point2DV2) -> float:
    return _cross_v2(
        (b[0] - a[0], b[1] - a[1]),
        (c[0] - a[0], c[1] - a[1]),
    )


def _segments_intersect_v2(
    first_start: Point2DV2,
    first_end: Point2DV2,
    second_start: Point2DV2,
    second_end: Point2DV2,
) -> bool:
    o1 = _orientation_v2(first_start, first_end, second_start)
    o2 = _orientation_v2(first_start, first_end, second_end)
    o3 = _orientation_v2(second_start, second_end, first_start)
    o4 = _orientation_v2(second_start, second_end, first_end)
    epsilon = 1e-9
    if o1 * o2 < -epsilon and o3 * o4 < -epsilon:
        return True
    return (
        abs(o1) <= epsilon
        and _point_on_segment_v2(second_start, first_start, first_end, tolerance=epsilon)
    ) or (
        abs(o2) <= epsilon
        and _point_on_segment_v2(second_end, first_start, first_end, tolerance=epsilon)
    ) or (
        abs(o3) <= epsilon
        and _point_on_segment_v2(first_start, second_start, second_end, tolerance=epsilon)
    ) or (
        abs(o4) <= epsilon
        and _point_on_segment_v2(first_end, second_start, second_end, tolerance=epsilon)
    )


def _polygon_edges_v2(
    polygon: Sequence[Point2DV2],
) -> Tuple[Tuple[Point2DV2, Point2DV2], ...]:
    points = tuple(polygon)
    return tuple(zip(points, points[1:] + points[:1]))


def _signed_area_v2(polygon: Sequence[Point2DV2]) -> float:
    return 0.5 * sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in _polygon_edges_v2(polygon)
    )


def _same_point_v2(
    first: Point2DV2,
    second: Point2DV2,
    *,
    tolerance: float = 1e-9,
) -> bool:
    return _distance_sq_v2(first, second) <= tolerance * tolerance


def _edge_matches_v2(
    first_start: Point2DV2,
    first_end: Point2DV2,
    second_start: Point2DV2,
    second_end: Point2DV2,
) -> bool:
    return (
        _same_point_v2(first_start, second_start)
        and _same_point_v2(first_end, second_end)
    ) or (
        _same_point_v2(first_start, second_end)
        and _same_point_v2(first_end, second_start)
    )


def _validate_simple_polygon_v2(
    polygon: Sequence[Point2DV2],
    name: str,
) -> None:
    points = tuple(polygon)
    if len(points) < 3:
        raise ValueError(f"{name} must contain at least three points")
    edges = _polygon_edges_v2(points)
    for index, (start, end) in enumerate(edges):
        if _same_point_v2(start, end):
            raise ValueError(f"{name} has a zero-length edge at vertex {index}")
    if abs(_signed_area_v2(points)) <= 1e-9:
        raise ValueError(f"{name} is degenerate (near-zero signed area)")

    edge_count = len(edges)
    for first_index, first_edge in enumerate(edges):
        for second_index in range(first_index + 1, edge_count):
            # Adjacent polygon edges are supposed to meet at their common
            # endpoint, including the closing first/last pair.
            if second_index == first_index + 1 or (
                first_index == 0 and second_index == edge_count - 1
            ):
                continue
            if _segments_intersect_v2(*first_edge, *edges[second_index]):
                raise ValueError(
                    f"{name} is self-intersecting between edges "
                    f"{first_index} and {second_index}"
                )


def _validate_clean_polygon_join_v2(
    first: Sequence[Point2DV2],
    second: Sequence[Point2DV2],
    *,
    shared_start: Point2DV2,
    shared_end: Point2DV2,
    name: str,
) -> None:
    """Validate an edge-connected union without requiring Shapely.

    The V2 S3 components are authored as three polygons sharing complete
    cross-section edges.  Simple operands plus exactly this clean edge join
    establish a valid, connected union while rejecting overlaps/crossings
    that would make the intended entry/corridor/exit topology ambiguous.
    """
    if _same_point_v2(shared_start, shared_end):
        raise ValueError(f"{name} has a degenerate shared edge")
    first_edges = _polygon_edges_v2(first)
    second_edges = _polygon_edges_v2(second)
    if not any(
        _edge_matches_v2(start, end, shared_start, shared_end)
        for start, end in first_edges
    ) or not any(
        _edge_matches_v2(start, end, shared_start, shared_end)
        for start, end in second_edges
    ):
        raise ValueError(f"{name} is not connected by the expected shared edge")

    for first_edge in first_edges:
        for second_edge in second_edges:
            if _edge_matches_v2(*first_edge, shared_start, shared_end) and _edge_matches_v2(
                *second_edge,
                shared_start,
                shared_end,
            ):
                continue
            if not _segments_intersect_v2(*first_edge, *second_edge):
                continue
            common_shared_endpoint = any(
                _same_point_v2(first_point, shared_point)
                and _same_point_v2(second_point, shared_point)
                for first_point in first_edge
                for second_point in second_edge
                for shared_point in (shared_start, shared_end)
            )
            if not common_shared_endpoint:
                raise ValueError(f"{name} has an intersection away from its shared edge")

            # At a seam endpoint, touching is valid but crossing into the
            # other polygon is not.  Test small interior samples on both
            # incident edges; boundary samples remain allowed.
            for edge, polygon in ((first_edge, second), (second_edge, first)):
                for fraction in (1e-6, 1.0 - 1e-6):
                    sample = (
                        edge[0][0] + fraction * (edge[1][0] - edge[0][0]),
                        edge[0][1] + fraction * (edge[1][1] - edge[0][1]),
                    )
                    if _point_strictly_in_polygon_v2(sample, polygon):
                        raise ValueError(f"{name} overlaps across its shared edge")

    shared_points = (shared_start, shared_end)
    for vertex in first:
        if not any(_same_point_v2(vertex, point) for point in shared_points):
            if _point_strictly_in_polygon_v2(vertex, second):
                raise ValueError(f"{name} has overlapping component interiors")
    for vertex in second:
        if not any(_same_point_v2(vertex, point) for point in shared_points):
            if _point_strictly_in_polygon_v2(vertex, first):
                raise ValueError(f"{name} has overlapping component interiors")


def _point_strictly_in_polygon_v2(
    point: Point2DV2,
    polygon: Sequence[Point2DV2],
) -> bool:
    if any(
        _point_on_segment_v2(point, start, end, tolerance=1e-8)
        for start, end in _polygon_edges_v2(polygon)
    ):
        return False
    return _point_in_polygon_v2(point, polygon)


def _validate_disjoint_polygon_interiors_v2(
    first: Sequence[Point2DV2],
    second: Sequence[Point2DV2],
    *,
    name: str,
) -> None:
    """Reject contact or overlap between non-neighbour S3 components."""
    if any(
        _segments_intersect_v2(*first_edge, *second_edge)
        for first_edge in _polygon_edges_v2(first)
        for second_edge in _polygon_edges_v2(second)
    ) or any(_point_strictly_in_polygon_v2(vertex, second) for vertex in first) or any(
        _point_strictly_in_polygon_v2(vertex, first) for vertex in second
    ):
        raise ValueError(f"{name} has overlapping non-neighbour components")


def _segment_intersection_parameters_v2(
    start: Point2DV2,
    end: Point2DV2,
    edge_start: Point2DV2,
    edge_end: Point2DV2,
) -> set[float]:
    """Return trajectory parameters at intersections, including overlaps."""
    direction = (end[0] - start[0], end[1] - start[1])
    edge_direction = (
        edge_end[0] - edge_start[0],
        edge_end[1] - edge_start[1],
    )
    direction_length_sq = direction[0] ** 2 + direction[1] ** 2
    if direction_length_sq <= 1e-18:
        return (
            {0.0}
            if _point_on_segment_v2(start, edge_start, edge_end, tolerance=1e-9)
            else set()
        )
    offset = (edge_start[0] - start[0], edge_start[1] - start[1])
    denominator = _cross_v2(direction, edge_direction)
    epsilon = 1e-9
    if abs(denominator) > epsilon:
        parameter = _cross_v2(offset, edge_direction) / denominator
        edge_parameter = _cross_v2(offset, direction) / denominator
        if -epsilon <= parameter <= 1.0 + epsilon and -epsilon <= edge_parameter <= 1.0 + epsilon:
            return {np_clip_v2(parameter, 0.0, 1.0)}
        return set()
    if abs(_cross_v2(offset, direction)) > epsilon:
        return set()

    projections = sorted(
        (
            ((point[0] - start[0]) * direction[0] + (point[1] - start[1]) * direction[1])
            / direction_length_sq
            for point in (edge_start, edge_end)
        )
    )
    overlap_start = max(0.0, projections[0])
    overlap_end = min(1.0, projections[1])
    if overlap_start > overlap_end + epsilon:
        return set()
    return {
        np_clip_v2(overlap_start, 0.0, 1.0),
        np_clip_v2(overlap_end, 0.0, 1.0),
    }


def _segment_capsule_transition_parameters_v2(
    start: Point2DV2,
    end: Point2DV2,
    edge_start: Point2DV2,
    edge_end: Point2DV2,
    *,
    radius: float,
) -> set[float]:
    """Return all points where a trajectory enters/exits an edge capsule."""
    if radius <= 0.0:
        return set()
    direction = (end[0] - start[0], end[1] - start[1])
    direction_length_sq = direction[0] ** 2 + direction[1] ** 2
    if direction_length_sq <= 1e-18:
        return set()
    epsilon = 1e-9
    result: set[float] = set()

    # Rounded ends of the capsule.
    for centre in (edge_start, edge_end):
        offset = (start[0] - centre[0], start[1] - centre[1])
        linear = 2.0 * (offset[0] * direction[0] + offset[1] * direction[1])
        constant = offset[0] ** 2 + offset[1] ** 2 - radius ** 2
        discriminant = linear ** 2 - 4.0 * direction_length_sq * constant
        if discriminant < -epsilon:
            continue
        root = math.sqrt(max(0.0, discriminant))
        for parameter in (
            (-linear - root) / (2.0 * direction_length_sq),
            (-linear + root) / (2.0 * direction_length_sq),
        ):
            if -epsilon <= parameter <= 1.0 + epsilon:
                result.add(np_clip_v2(parameter, 0.0, 1.0))

    # Parallel sides of the capsule.  The projection guard keeps only the
    # finite edge strip; its rounded ends were handled above.
    edge = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
    edge_length = math.hypot(*edge)
    if edge_length <= epsilon:
        return result
    signed_start = _cross_v2(edge, (start[0] - edge_start[0], start[1] - edge_start[1]))
    signed_delta = _cross_v2(edge, direction)
    edge_length_sq = edge_length * edge_length
    if abs(signed_delta) > epsilon:
        for target in (-radius * edge_length, radius * edge_length):
            parameter = (target - signed_start) / signed_delta
            if not -epsilon <= parameter <= 1.0 + epsilon:
                continue
            point = (
                start[0] + parameter * direction[0],
                start[1] + parameter * direction[1],
            )
            projection = (
                (point[0] - edge_start[0]) * edge[0]
                + (point[1] - edge_start[1]) * edge[1]
            ) / edge_length_sq
            if -epsilon <= projection <= 1.0 + epsilon:
                result.add(np_clip_v2(parameter, 0.0, 1.0))
    return result


def _segment_touches_polygon_v2(
    start: Point2DV2,
    end: Point2DV2,
    polygon: Sequence[Point2DV2],
) -> bool:
    return any(
        _segments_intersect_v2(start, end, edge_start, edge_end)
        for edge_start, edge_end in zip(
            polygon,
            tuple(polygon[1:]) + (polygon[0],),
        )
    )


def np_clip_v2(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


__all__ = [
    "FinishLineV2",
    "Footprint2DV2",
    "S3DrivableAreaV2",
    "TERMINATION_CONTRACT_VERSION_V2",
    "TerminalDecisionV2",
    "TerminationCheckerV2",
    "swept_footprint_v2",
]
