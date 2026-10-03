"""
EpisodeTerminationChecker
=========================
Evaluates all episode-ending conditions each step.

Terminated (failure):
    - CARLA collision sensor fired  (pedestrians, static objects)
    - Ego's 2-D oriented bbox overlaps a physics-disabled background vehicle
      bbox  (CARLA sensor cannot reliably detect kinematic actors)
    - Ego footprint violates the authored forbidden area / safe corridor

Terminated + truncated (absorbing timeout; no value bootstrap):
    - Step count ≥ max_steps  (timeout)

Success:
    - Manual finish line: vehicle-centre path crosses the finite line segment
    - Legacy fallback: destination radius + heading (only for old configs)

ForbiddenAreaMonitor is no longer a separate class; the AABB logic is
inlined here to reduce total file count.
"""
from __future__ import annotations

import math
import threading
from typing import Any, Callable

import carla

from config.scenario_config import ScenarioConfig

_COLLISION_REASON_PRIORITY = {
    "collision_walker": 0,
    "collision_sumo_vehicle": 1,
    "collision_carla_vehicle": 2,
    "collision_static": 3,
    "collision_other": 4,
    "collision_unknown": 5,
}


class EpisodeTerminationChecker:

    def __init__(self, config: ScenarioConfig, carla_map: carla.Map | None = None) -> None:
        self._cfg             = config
        self._wz              = config.workzone
        self._collision_flag  = False
        self._collision_reason = "collision_unknown"
        self._collision_actor_id: int | None = None
        self._collision_actor_type = ""
        self._collision_actor_role = ""
        self._collision_detector = ""
        self._collision_owner = ""
        self._collision_counterpart = ""
        self._collision_impulse_magnitude: float | None = None
        self._collision_frame: int | None = None
        self._collision_event_count = 0
        self._collision_sort_key: tuple | None = None
        self._collision_lock = threading.Lock()
        self._carla_map = carla_map
        self._collision_sensor: carla.Actor | None = None
        self._step            = 0
        self._previous_ego_xy: tuple[float, float] | None = None
        self._actor_register: Callable[[carla.Actor, str], None] | None = None
        self._actor_unregister: Callable[[int], None] | None = None
        # Optional Shapely polygon for curved work-zone intrusion detection.
        # Set via set_forbidden_polygon() after CARLA map is ready.
        self._forbidden_polygon: Any | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def attach_collision_sensor(
        self,
        world: carla.World,
        ego: carla.Actor,
        actor_register: Callable[[carla.Actor, str], None] | None = None,
        actor_unregister: Callable[[int], None] | None = None,
    ) -> None:
        """Attach a CARLA collision sensor (detects pedestrians / static objects)."""
        bp = world.get_blueprint_library().find("sensor.other.collision")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "episode_sensor")
        self._actor_register = actor_register
        self._actor_unregister = actor_unregister
        self._collision_sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=ego
        )
        if self._actor_register:
            self._actor_register(self._collision_sensor, "collision_sensor")
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, event: Any) -> None:
        other = getattr(event, "other_actor", None)
        type_id = getattr(other, "type_id", "") if other is not None else ""
        actor_id = getattr(other, "id", None) if other is not None else None
        try:
            role_name = other.attributes.get("role_name", "") if other is not None else ""
        except (AttributeError, RuntimeError):
            role_name = ""
        owner = "carla"
        if role_name == "sumo_background":
            reason = "collision_sumo_vehicle"
            owner = "sumo"
            counterpart = "vehicle"
        elif type_id.startswith("walker."):
            reason = "collision_walker"
            counterpart = "walker"
        elif type_id.startswith("vehicle."):
            reason = "collision_carla_vehicle"
            counterpart = "vehicle"
        elif type_id.startswith(("static.", "traffic.")):
            reason = "collision_static"
            counterpart = "static"
        elif type_id:
            reason = "collision_other"
            owner = "unknown"
            counterpart = "other"
        else:
            reason = "collision_unknown"
            owner = "unknown"
            counterpart = "unknown"

        frame_raw = getattr(event, "frame", None)
        frame = int(frame_raw) if isinstance(frame_raw, (int, float)) else None
        impulse_magnitude = _collision_impulse_magnitude(event)
        actor_sort_id = int(actor_id) if isinstance(actor_id, int) else 2**63 - 1
        # CARLA may emit several callbacks in one simulation frame.  Prefer the
        # earliest frame; within one frame use a stable semantic/actor ordering
        # so the reported primary collision does not depend on callback order.
        sort_key = (
            frame is None,
            frame if frame is not None else 2**63 - 1,
            _COLLISION_REASON_PRIORITY[reason],
            actor_sort_id,
            str(type_id),
            str(role_name),
        )

        lock = getattr(self, "_collision_lock", None)
        if lock is None:
            # Compatibility for lightweight tests/diagnostics that construct
            # the checker with __new__ rather than calling __init__.
            lock = threading.Lock()
            self._collision_lock = lock
        with lock:
            self._collision_event_count = getattr(self, "_collision_event_count", 0) + 1
            current_key = getattr(self, "_collision_sort_key", None)
            if current_key is not None and current_key <= sort_key:
                return
            self._collision_flag = True
            self._collision_sort_key = sort_key
            self._collision_reason = reason
            self._collision_actor_id = actor_id
            self._collision_actor_type = type_id
            self._collision_actor_role = role_name
            self._collision_detector = "carla_sensor"
            self._collision_owner = owner
            self._collision_counterpart = counterpart
            self._collision_impulse_magnitude = impulse_magnitude
            self._collision_frame = frame

    def reset(self) -> None:
        self._collision_flag = False
        self._collision_reason = "collision_unknown"
        self._collision_actor_id = None
        self._collision_actor_type = ""
        self._collision_actor_role = ""
        self._collision_detector = ""
        self._collision_owner = ""
        self._collision_counterpart = ""
        self._collision_impulse_magnitude = None
        self._collision_frame = None
        self._collision_event_count = 0
        self._collision_sort_key = None
        self._step           = 0
        self._previous_ego_xy = None

    def set_forbidden_polygon(self, polygon: Any) -> None:
        """Inject a Shapely polygon built from the curved work-zone geometry."""
        self._forbidden_polygon = polygon

    def destroy(self) -> None:
        if self._collision_sensor is not None:
            actor_id = self._collision_sensor.id
            try:
                if self._collision_sensor.is_alive:
                    self._collision_sensor.destroy()
            except Exception:
                return
            if self._actor_unregister:
                self._actor_unregister(actor_id)
            self._collision_sensor = None

    # ------------------------------------------------------------------ #
    #  Per-step check                                                      #
    # ------------------------------------------------------------------ #

    def tick(
        self,
        ego: carla.Actor,
        dest: carla.Transform,
        bg_actor_map: dict | None = None,
    ) -> tuple[bool, bool, bool, dict]:
        """
        Parameters
        ----------
        ego          : CARLA Ego actor
        dest         : destination Transform
        bg_actor_map : {sumo_id: carla.Actor}  physics-disabled background
                       vehicles that the sensor cannot detect

        Returns
        -------
        (terminated, truncated, success, info)
        """
        self._step += 1
        info: dict = {"step": self._step}

        # 1. CARLA sensor: pedestrian / static-object collision
        if self._collision_flag:
            with self._collision_lock:
                info["reason"] = self._collision_reason
                info["collision_actor_id"] = self._collision_actor_id
                info["collision_actor_type"] = self._collision_actor_type or "unknown"
                info["collision_actor_role"] = self._collision_actor_role or "(none)"
                info["collision_detector"] = self._collision_detector
                info["collision_owner"] = self._collision_owner
                info["collision_counterpart"] = self._collision_counterpart
                info["collision_event_count"] = self._collision_event_count
                if self._collision_frame is not None:
                    info["collision_frame"] = self._collision_frame
                if self._collision_impulse_magnitude is not None:
                    info["collision_impulse_magnitude"] = self._collision_impulse_magnitude
            return True, False, False, info

        # 2. Geometry check: ego footprint overlaps a physics-disabled mirror.
        bg_hit = self._bg_collision(ego, bg_actor_map) if bg_actor_map else None
        if bg_hit is not None:
            sumo_id, actor_id, distance_m = bg_hit
            info["reason"] = "collision_sumo_vehicle_proximity"
            info["collision_sumo_id"] = sumo_id
            info["collision_actor_id"] = actor_id
            info["collision_distance_m"] = distance_m
            info["collision_detector"] = "obb_overlap"
            info["collision_owner"] = "sumo"
            info["collision_counterpart"] = "vehicle"
            info["collision_overlap_method"] = "2d_obb_sat"
            return True, False, False, info

        # 3/4. Ordinary scenarios always use CARLA lane validity before their
        # work-zone geometry.  An open-ended S3 corridor instead has three
        # phases: CARLA owns the approach/exit, while the manually annotated
        # corridor owns the section between its entrance and exit cross-lines.
        corridor_active = self._open_corridor_active(ego)
        if corridor_active is not True and self._is_offroad(ego):
            info["reason"] = "offroad"
            return True, False, False, info

        if corridor_active is not False and self._wz_intrusion(ego):
            info["reason"] = (
                "corridor_departure"
                if getattr(self._wz, "geometry_mode", "forbidden_rect") == "safe_corridor"
                else "workzone_intrusion"
            )
            return True, False, False, info

        # 5. Manual finite finish-line crossing.
        if self._cfg.destination.finish_line is not None:
            crossed = self._finish_line_crossed(ego)
            if crossed:
                info["reason"] = "finish_line_crossed"
                info["crossing_reference"] = self._cfg.destination.finish_line.crossing_reference
                return True, False, True, info

        # 6. Timeout
        if self._step >= self._cfg.episode.max_steps:
            info["reason"] = "timeout"
            # Mark the cap as absorbing as well as time-limited.  SB3's vector
            # wrapper only bootstraps truncated-only endings; both flags keep
            # the explicit timeout penalty as the final return target.
            return True, True, False, info

        # 7. Goal observation/progress distance. Manual finish-line settings
        # use the midpoint only as an observation and reward reference.
        ego_loc = ego.get_location()
        dist    = math.sqrt(
            (ego_loc.x - dest.location.x) ** 2 +
            (ego_loc.y - dest.location.y) ** 2
        )
        info["dist_to_goal"] = dist

        if self._cfg.destination.finish_line is None and dist < self._cfg.destination.success_dist_m:
            ego_yaw = ego.get_transform().rotation.yaw
            hdg_err = abs(((ego_yaw - dest.rotation.yaw + 180.0) % 360.0) - 180.0)
            if hdg_err < self._cfg.destination.success_heading_deg:
                info["reason"]        = "goal_reached"
                info["heading_error"] = hdg_err
                return True, False, True, info

        return False, False, False, info

    def _open_corridor_active(self, ego: carla.Actor) -> bool | None:
        """Return whether an open S3 corridor owns drivable-area checking.

        ``True`` means the ego centre lies between the entrance and exit
        cross-sections, so CARLA off-road checking is skipped and only the
        annotated side boundaries are enforced. ``False`` means approach or
        exit, where CARLA remains authoritative and corridor containment is
        skipped. ``None`` preserves the legacy checking order for every
        closed corridor and non-S3 geometry.
        """
        wz = self._wz
        if (
            getattr(wz, "geometry_mode", "forbidden_rect") != "safe_corridor"
            or not getattr(wz, "corridor_open_ends", False)
        ):
            return None
        left = getattr(wz, "corridor_left_boundary_points", None)
        right = getattr(wz, "corridor_right_boundary_points", None)
        if not left or not right:
            raise ValueError("Open-ended safe corridor is missing side boundaries")
        location = ego.get_location()
        return _open_corridor_phase((float(location.x), float(location.y)), left, right) == "inside"

    def _finish_line_crossed(self, ego: carla.Actor) -> bool:
        """True when the ego-centre trajectory crosses the configured segment."""
        line = self._cfg.destination.finish_line
        if line is None:
            return False
        if line.crossing_reference != "vehicle_center":
            raise ValueError(
                f"Unsupported finish-line crossing_reference: {line.crossing_reference}"
            )

        location = ego.get_location()
        current = (float(location.x), float(location.y))
        previous = self._previous_ego_xy
        self._previous_ego_xy = current

        crossing_heading = getattr(line, "crossing_heading_deg", None)
        if crossing_heading is None:
            crossing_heading = self._cfg.carla.road_heading_deg

        # S3 destination lines are deliberately independent of the corridor's
        # initial heading (WZ3 starts southbound but finishes eastbound).  The
        # explicit crossing heading selects the upstream side; the geometric
        # intersection below still requires the ego-centre path to touch the
        # authored finite segment.
        heading = math.radians(float(crossing_heading))
        return _crosses_directed_segment(
            previous=previous,
            current=current,
            start=line.start,
            end=line.end,
            forward_direction=(math.cos(heading), math.sin(heading)),
        )

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _bg_collision(self, ego: carla.Actor,
                       bg_map: dict[str, carla.Actor]) -> tuple[str, int, float] | None:
        """Return the closest SUMO mirror whose 2-D OBB overlaps Ego's OBB."""
        ego_loc = ego.get_location()
        ego_footprint = _actor_footprint_xy(ego)
        overlaps: list[tuple[float, str, int]] = []
        for sumo_id, actor in bg_map.items():
            try:
                if not actor.is_alive:
                    continue
                actor_footprint = _actor_footprint_xy(actor)
                if not _convex_polygons_overlap(ego_footprint, actor_footprint):
                    continue
                actor_loc = actor.get_location()
                actor_id = int(actor.id)
            except RuntimeError:
                # A mirror can be destroyed after get_actor_map() returns its
                # snapshot.  Ignore only that stale actor; ego access and
                # malformed bbox data still propagate to the fault boundary.
                continue
            distance_m = math.hypot(
                float(ego_loc.x) - float(actor_loc.x),
                float(ego_loc.y) - float(actor_loc.y),
            )
            overlaps.append((distance_m, str(sumo_id), actor_id))
        if not overlaps:
            return None
        distance_m, sumo_id, actor_id = min(overlaps)
        return sumo_id, actor_id, distance_m

    def _is_offroad(self, ego: carla.Actor) -> bool:
        if self._carla_map is None:
            return False
        try:
            waypoint = self._carla_map.get_waypoint(
                ego.get_location(),
                project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            return waypoint is None
        except Exception:
            # A map-query transport fault is handled by the environment's
            # CARLA fault boundary; it is not silently labeled as off-road.
            raise

    def _wz_intrusion(self, ego: carla.Actor) -> bool:
        """
        Return True if Ego has violated the work-zone constraint.

        Behaviour depends on ``self._wz.geometry_mode``:
          - "forbidden_rect"    : True when Ego OBB overlaps the axis-aligned
            rectangle (Separating Axis Theorem, 4 axes).
          - "forbidden_polygon" : True when Ego footprint intersects the curved
            Shapely polygon (built from head/tail cones).
          - "safe_corridor"     : True when Ego footprint is NOT fully inside
            the corridor rectangle (i.e. Ego left the safe drivable area).
        """
        mode = getattr(self._wz, "geometry_mode", "forbidden_rect")

        if mode == "safe_corridor":
            return self._corridor_violation(ego)

        if mode == "forbidden_polygon" and self._forbidden_polygon is not None:
            from logic.workzone_geometry import intersects_actor
            return intersects_actor(self._forbidden_polygon, ego)

        # Legacy AABB-OBB SAT (4 axes) for straight-lane scenarios
        t   = ego.get_transform()
        ext = ego.bounding_box.extent
        ex, ey  = ext.x, ext.y
        yaw_rad = math.radians(t.rotation.yaw)
        c_y, s_y = math.cos(yaw_rad), math.sin(yaw_rad)

        wz   = self._wz
        wz_cx = (wz.x_min + wz.x_max) / 2.0
        wz_cy = (wz.y_min + wz.y_max) / 2.0
        wz_ex = (wz.x_max - wz.x_min) / 2.0
        wz_ey = (wz.y_max - wz.y_min) / 2.0

        dx = t.location.x - wz_cx   # centre-to-centre offset
        dy = t.location.y - wz_cy

        # Axis 1 — World X
        if abs(dx) > wz_ex + abs(ex * c_y) + abs(ey * s_y):
            return False
        # Axis 2 — World Y
        if abs(dy) > wz_ey + abs(ex * s_y) + abs(ey * c_y):
            return False
        # Axis 3 — Ego forward (c_y, s_y)
        if abs(dx * c_y + dy * s_y) > abs(wz_ex * c_y) + abs(wz_ey * s_y) + ex:
            return False
        # Axis 4 — Ego sideways (−s_y, c_y)
        if abs(-dx * s_y + dy * c_y) > abs(wz_ex * s_y) + abs(wz_ey * c_y) + ey:
            return False

        return True   # no separating axis → overlap

    def _corridor_violation(self, ego: carla.Actor) -> bool:
        """True when Ego is not fully inside the safe corridor.

        If the corridor is defined by curved boundary points, builds a Shapely
        polygon and requires the ego's 2-D footprint to be fully covered by it.
        Otherwise uses the axis-aligned rectangle (expanded by
        ``boundary_tolerance_m``) and requires every bbox corner inside it.
        This is the S3 "drivable corridor" rule: leaving the corridor is a
        work-zone violation.
        """
        wz = self._wz
        boundary_pts = getattr(wz, "corridor_boundary_points", None)
        left = getattr(wz, "corridor_left_boundary_points", None)
        right = getattr(wz, "corridor_right_boundary_points", None)

        if boundary_pts or (left and right):
            try:
                from shapely.geometry import MultiPoint, Polygon
                if getattr(wz, "corridor_open_ends", False) and left and right:
                    # Only the side boundaries are walls.  Extend both ends by
                    # more than the ego footprint so a centre just inside a
                    # cross-section is not rejected because its bumper crosses
                    # the artificial polygon end-cap.
                    verts = ego.bounding_box.get_world_vertices(ego.get_transform())
                    footprint = MultiPoint([(v.x, v.y) for v in verts]).convex_hull
                    min_x, min_y, max_x, max_y = footprint.bounds
                    extension_m = math.hypot(max_x - min_x, max_y - min_y) + 1.0
                    polygon_points = _open_corridor_polygon_points(
                        left, right, extension_m=extension_m
                    )
                else:
                    polygon_points = boundary_pts
                    verts = ego.bounding_box.get_world_vertices(ego.get_transform())
                    footprint = MultiPoint([(v.x, v.y) for v in verts]).convex_hull

                poly = Polygon(polygon_points)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                tolerance = float(getattr(wz, "boundary_tolerance_m", 0.0))
                if tolerance:
                    poly = poly.buffer(tolerance)
                return not poly.covers(footprint)
            except ImportError:
                # Fall back to AABB containment if shapely is unavailable.
                pass

        tol = getattr(wz, "boundary_tolerance_m", 0.0)
        x_lo, x_hi = wz.x_min - tol, wz.x_max + tol
        y_lo, y_hi = wz.y_min - tol, wz.y_max + tol

        verts = ego.bounding_box.get_world_vertices(ego.get_transform())
        for v in verts:
            if not (x_lo <= v.x <= x_hi and y_lo <= v.y <= y_hi):
                return True
        return False


def _collision_impulse_magnitude(event: Any) -> float | None:
    impulse = getattr(event, "normal_impulse", None)
    if impulse is None:
        return None
    try:
        x = float(impulse.x)
        y = float(impulse.y)
        z = float(impulse.z)
    except (AttributeError, TypeError, ValueError):
        return None
    magnitude = math.sqrt(x * x + y * y + z * z)
    return magnitude if math.isfinite(magnitude) else None


def _actor_footprint_xy(actor: carla.Actor) -> list[tuple[float, float]]:
    """Return the convex XY footprint of an actor's world-oriented bbox."""
    transform = actor.get_transform()
    vertices = actor.bounding_box.get_world_vertices(transform)
    points = [(float(vertex.x), float(vertex.y)) for vertex in vertices]
    return _convex_hull(points)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew monotone-chain hull; CARLA's 8 bbox vertices duplicate in XY."""
    ordered = sorted(set(points))
    if len(ordered) < 3:
        raise ValueError("Actor bounding box must produce at least three XY points")

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("Actor bounding box XY footprint is degenerate")
    return hull


def _convex_polygons_overlap(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
    *,
    tolerance: float = 1e-7,
) -> bool:
    """Separating-axis test; touching footprint boundaries count as contact."""
    for polygon in (first, second):
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            edge_x = end[0] - start[0]
            edge_y = end[1] - start[1]
            axis_x, axis_y = -edge_y, edge_x
            axis_length = math.hypot(axis_x, axis_y)
            if axis_length <= 1e-12:
                continue
            axis_x /= axis_length
            axis_y /= axis_length
            first_projection = [x * axis_x + y * axis_y for x, y in first]
            second_projection = [x * axis_x + y * axis_y for x, y in second]
            if (
                max(first_projection) < min(second_projection) - tolerance
                or max(second_projection) < min(first_projection) - tolerance
            ):
                return False
    return True


def _open_corridor_phase(
    point: tuple[float, float],
    left_boundary: list[tuple[float, float]],
    right_boundary: list[tuple[float, float]],
) -> str:
    """Classify a point as ``before``, ``inside`` or ``after`` an open corridor.

    The entrance/exit are the cross-sections joining the first/last boundary
    pairs.  Their forward normals come from the adjacent centreline samples,
    so the calculation uses only vector dot products and is independent of
    CARLA's left-handed world-coordinate convention.
    """
    if len(left_boundary) < 2 or len(right_boundary) < 2:
        raise ValueError("Open corridor sides must each contain at least two points")
    if len(left_boundary) != len(right_boundary):
        raise ValueError("Open corridor sides must contain matching cross-sections")

    count = len(left_boundary)
    centres = [
        (
            (float(left_boundary[i][0]) + float(right_boundary[i][0])) * 0.5,
            (float(left_boundary[i][1]) + float(right_boundary[i][1])) * 0.5,
        )
        for i in range(count)
    ]

    def dot_from(origin: tuple[float, float], direction: tuple[float, float]) -> float:
        return (
            (float(point[0]) - origin[0]) * direction[0]
            + (float(point[1]) - origin[1]) * direction[1]
        )

    entry_direction = _unit_direction(centres[0], centres[1])
    exit_direction = _unit_direction(centres[-2], centres[-1])
    if dot_from(centres[0], entry_direction) < 0.0:
        return "before"
    if dot_from(centres[-1], exit_direction) > 0.0:
        return "after"
    return "inside"


def _open_corridor_polygon_points(
    left_boundary: list[tuple[float, float]],
    right_boundary: list[tuple[float, float]],
    *,
    extension_m: float,
) -> list[tuple[float, float]]:
    """Return a side-wall polygon whose entrance and exit are extended open."""
    if extension_m <= 0.0:
        raise ValueError("extension_m must be positive")
    if len(left_boundary) < 2 or len(right_boundary) < 2:
        raise ValueError("Open corridor sides must each contain at least two points")
    if len(left_boundary) != len(right_boundary):
        raise ValueError("Open corridor sides must contain matching cross-sections")
    count = len(left_boundary)
    left = [(float(x), float(y)) for x, y in left_boundary[:count]]
    right = [(float(x), float(y)) for x, y in right_boundary[:count]]
    left_entry_dir = _unit_direction(left[0], left[1])
    right_entry_dir = _unit_direction(right[0], right[1])
    left_exit_dir = _unit_direction(left[-2], left[-1])
    right_exit_dir = _unit_direction(right[-2], right[-1])

    def shifted(p: tuple[float, float], d: tuple[float, float], amount: float):
        return (p[0] + d[0] * amount, p[1] + d[1] * amount)

    extended_left = [
        shifted(left[0], left_entry_dir, -extension_m),
        *left,
        shifted(left[-1], left_exit_dir, extension_m),
    ]
    extended_right = [
        shifted(right[0], right_entry_dir, -extension_m),
        *right,
        shifted(right[-1], right_exit_dir, extension_m),
    ]
    return extended_left + list(reversed(extended_right))


def _unit_direction(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("Adjacent corridor cross-sections must have distinct centres")
    return dx / length, dy / length


def _crosses_directed_finish_line(
    *,
    previous: tuple[float, float] | None,
    current: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    heading_deg: float,
) -> bool:
    """Backward-compatible heading wrapper for finite directed crossing."""
    heading = math.radians(heading_deg)
    return _crosses_directed_segment(
        previous=previous,
        current=current,
        start=start,
        end=end,
        forward_direction=(math.cos(heading), math.sin(heading)),
    )


def _crosses_directed_segment(
    *,
    previous: tuple[float, float] | None,
    current: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    forward_direction: tuple[float, float],
) -> bool:
    """Return whether the ego-centre motion crosses a finite line forward.

    Both constraints are geometric: the previous-to-current motion segment
    must intersect the authored finish segment, and it must move from the
    upstream half-plane to the downstream half-plane selected by
    ``forward_direction``.  A first observation never counts as a crossing.
    """
    x1, y1 = start
    x2, y2 = end
    seg_x, seg_y = x2 - x1, y2 - y1
    seg_len_sq = seg_x * seg_x + seg_y * seg_y
    if seg_len_sq <= 1e-12:
        raise ValueError("Destination finish line must have two different endpoints")

    fwd_x, fwd_y = float(forward_direction[0]), float(forward_direction[1])
    fwd_length = math.hypot(fwd_x, fwd_y)
    if not math.isfinite(fwd_length) or fwd_length <= 1e-12:
        raise ValueError("Finish-line forward direction must be non-zero and finite")
    fwd_x /= fwd_length
    fwd_y /= fwd_length

    # Use the finish segment's own normal so the upstream/downstream plane is
    # exactly the authored segment's supporting line.  The explicit heading
    # only chooses which of the two normals is forward.
    normal_x, normal_y = -seg_y, seg_x
    if normal_x * fwd_x + normal_y * fwd_y < 0.0:
        normal_x, normal_y = -normal_x, -normal_y
    alignment = normal_x * fwd_x + normal_y * fwd_y
    if alignment <= 1e-12:
        raise ValueError("Finish-line crossing heading must not be parallel to the line")

    def signed_side(point: tuple[float, float]) -> float:
        return (point[0] - x1) * normal_x + (point[1] - y1) * normal_y

    if previous is None:
        return False

    current_side = signed_side(current)
    previous_side = signed_side(previous)
    if not (previous_side < 0.0 <= current_side):
        return False

    motion_x = current[0] - previous[0]
    motion_y = current[1] - previous[1]
    denominator = _cross_2d(motion_x, motion_y, seg_x, seg_y)
    if abs(denominator) <= 1e-12:
        return False

    offset_x = x1 - previous[0]
    offset_y = y1 - previous[1]
    motion_fraction = _cross_2d(offset_x, offset_y, seg_x, seg_y) / denominator
    finish_fraction = _cross_2d(offset_x, offset_y, motion_x, motion_y) / denominator
    tolerance = 1e-9
    return (
        -tolerance <= motion_fraction <= 1.0 + tolerance
        and -tolerance <= finish_fraction <= 1.0 + tolerance
    )


def _cross_2d(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx
