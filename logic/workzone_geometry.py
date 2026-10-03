"""
workzone_geometry.py
====================
Build a curved forbidden polygon for a work-zone from head/tail cone
CARLA locations, using waypoint sampling along the driving lane.

Public API
----------
build_workzone_polygon(carla_map, head_location, tail_location, ...)
    -> WorkZonePolygon  (namedtuple with .polygon, .centerline_pts, ...)

actor_footprint_polygon(actor)
    -> shapely.geometry.Polygon  (2-D XY footprint from CARLA bbox vertices)

intersects_actor(workzone_polygon, actor)
    -> bool
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple, Optional

import carla

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import LineString, MultiPoint
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False
    logger.warning(
        "Shapely is not installed — curved work-zone polygon support is disabled. "
        "Install with:  pip install 'Shapely>=1.8,<3'"
    )


class WorkZoneGeometryError(RuntimeError):
    """Raised when waypoints cannot be connected or the polygon is degenerate."""


class WorkZonePolygon(NamedTuple):
    """Result bundle returned by build_workzone_polygon()."""
    polygon: object                # shapely.geometry.Polygon
    centerline_pts: list           # [(x, y), ...] CARLA world coordinates
    start_wp: object               # carla.Waypoint matched to head cone
    end_wp: object                 # carla.Waypoint matched to tail cone
    road_id: int
    lane_id: int
    actual_half_width_m: float


# --------------------------------------------------------------------------- #
#  Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _require_shapely() -> None:
    if not _SHAPELY_AVAILABLE:
        raise ImportError(
            "Shapely is required for curved work-zone polygons. "
            "Install with:  pip install 'Shapely>=1.8,<3'"
        )


def _dist2d(a: object, b: object) -> float:
    """Euclidean distance between two carla.Waypoint transforms (XY only)."""
    ax, ay = a.transform.location.x, a.transform.location.y
    bx, by = b.transform.location.x, b.transform.location.y
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _get_waypoint_on_lane(
    carla_map: carla.Map,
    location: carla.Location,
    expected_road_id: Optional[int],
    expected_lane_id: Optional[int],
) -> carla.Waypoint:
    """
    Snap location to the nearest driving waypoint, with optional road/lane
    validation.

    Cones are often placed at lane edges, which can cause get_waypoint() to
    snap to a neighbouring lane.  When expected_road_id / expected_lane_id
    are provided, left and right neighbours are also checked.
    """
    wp = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        raise WorkZoneGeometryError(
            f"No driving waypoint found near "
            f"({location.x:.2f}, {location.y:.2f}, {location.z:.2f})"
        )

    if expected_road_id is None and expected_lane_id is None:
        logger.debug(
            "Waypoint matched (%.2f, %.2f) → road_id=%d lane_id=%d",
            location.x, location.y, wp.road_id, wp.lane_id,
        )
        return wp

    def _matches(w: carla.Waypoint) -> bool:
        road_ok = (expected_road_id is None or w.road_id == expected_road_id)
        lane_ok = (expected_lane_id is None or w.lane_id == expected_lane_id)
        return road_ok and lane_ok

    if _matches(wp):
        return wp

    # Check adjacent driving lanes
    for neighbour in (wp.get_left_lane(), wp.get_right_lane()):
        if (neighbour is not None
                and neighbour.lane_type == carla.LaneType.Driving
                and _matches(neighbour)):
            logger.debug(
                "Adjusted to neighbour lane: road_id=%d lane_id=%d",
                neighbour.road_id, neighbour.lane_id,
            )
            return neighbour

    raise WorkZoneGeometryError(
        f"Cannot find waypoint at ({location.x:.2f}, {location.y:.2f}) "
        f"matching road_id={expected_road_id} lane_id={expected_lane_id}. "
        f"Best match: road_id={wp.road_id} lane_id={wp.lane_id}."
    )


def _sample_waypoints(
    start_wp: carla.Waypoint,
    end_wp: carla.Waypoint,
    spacing_m: float,
    max_dist_m: float,
    preferred_road_id: int,
    preferred_lane_id: int,
    use_next: bool,
) -> Optional[list]:
    """
    Greedy waypoint chain from start_wp toward end_wp.

    use_next=True  → calls waypoint.next(spacing_m)
    use_next=False → calls waypoint.previous(spacing_m)

    Returns [start_wp, ..., end_wp] on success, None if unreachable.
    """
    pts = [start_wp]
    visited: set[tuple[float, float]] = {
        (round(start_wp.transform.location.x, 2),
         round(start_wp.transform.location.y, 2))
    }
    max_steps = int(max_dist_m / spacing_m) + 10
    current = start_wp

    for _ in range(max_steps):
        if _dist2d(current, end_wp) <= spacing_m * 1.5:
            pts.append(end_wp)
            return pts

        candidates = current.next(spacing_m) if use_next else current.previous(spacing_m)
        if not candidates:
            return None

        # Prefer same road/lane; break ties by proximity to end_wp
        def _priority(w: carla.Waypoint) -> tuple:
            same = (w.road_id == preferred_road_id and w.lane_id == preferred_lane_id)
            return (0 if same else 1, _dist2d(w, end_wp))

        candidates.sort(key=_priority)

        advanced = False
        for cand in candidates:
            key = (round(cand.transform.location.x, 2),
                   round(cand.transform.location.y, 2))
            if key in visited:
                continue
            visited.add(key)
            pts.append(cand)
            current = cand
            advanced = True
            break

        if not advanced:
            return None

    return None  # max_steps exceeded without reaching end_wp


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def build_workzone_polygon(
    carla_map: carla.Map,
    head_location: carla.Location,
    tail_location: carla.Location,
    sample_spacing_m: float = 0.5,
    half_width_m: Optional[float] = None,
    margin_m: float = 0.0,
    expected_road_id: Optional[int] = None,
    expected_lane_id: Optional[int] = None,
) -> WorkZonePolygon:
    """
    Build a curved forbidden Shapely polygon from head/tail cone locations.

    Parameters
    ----------
    carla_map         : carla.Map  (must already be loaded)
    head_location     : CARLA Location of the upstream (head) cone
    tail_location     : CARLA Location of the downstream (tail) cone
    sample_spacing_m  : waypoint spacing along the lane (m); default 0.5
    half_width_m      : half-width of the forbidden zone (m).
                        If None, uses the waypoint's lane_width / 2.
    margin_m          : extra margin added to half_width_m (m); default 0.0
    expected_road_id  : if provided, verifies the matched waypoint road
    expected_lane_id  : if provided, verifies the matched waypoint lane

    Returns
    -------
    WorkZonePolygon namedtuple

    Raises
    ------
    WorkZoneGeometryError : anchors unreachable, or resulting polygon degenerate
    ImportError           : Shapely not installed
    """
    _require_shapely()

    start_wp = _get_waypoint_on_lane(
        carla_map, head_location, expected_road_id, expected_lane_id,
    )
    end_wp = _get_waypoint_on_lane(
        carla_map, tail_location, expected_road_id, expected_lane_id,
    )

    road_id = start_wp.road_id
    lane_id = start_wp.lane_id
    est_dist = _dist2d(start_wp, end_wp)
    max_dist = est_dist * 3.0 + 20.0   # generous budget for curves

    # Try all four combinations of (anchor order) × (next/previous)
    wps: Optional[list] = None
    for (a, b) in ((start_wp, end_wp), (end_wp, start_wp)):
        for use_next in (True, False):
            wps = _sample_waypoints(a, b, sample_spacing_m, max_dist, road_id, lane_id, use_next)
            if wps is not None:
                if a is end_wp:          # anchors were swapped
                    wps = list(reversed(wps))
                break
        if wps is not None:
            break

    if wps is None or len(wps) < 2:
        raise WorkZoneGeometryError(
            f"Cannot connect work-zone anchors: "
            f"head=({head_location.x:.2f}, {head_location.y:.2f}) "
            f"[road={road_id} lane={lane_id}] ↔ "
            f"tail=({tail_location.x:.2f}, {tail_location.y:.2f}) "
            f"[road={end_wp.road_id} lane={end_wp.lane_id}]. "
            f"Tried next() and previous() up to {max_dist:.1f} m."
        )

    actual_half_width = (
        half_width_m if half_width_m is not None else start_wp.lane_width / 2.0
    ) + margin_m

    centerline_pts = [
        (wp.transform.location.x, wp.transform.location.y) for wp in wps
    ]

    polygon = LineString(centerline_pts).buffer(
        actual_half_width, cap_style=2, join_style=1
    )
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise WorkZoneGeometryError(
            f"Buffered polygon is empty (half_width={actual_half_width:.2f} m, "
            f"waypoints={len(wps)}). Verify spacing_m and lane geometry."
        )

    logger.info(
        "Work-zone polygon built: road_id=%d lane_id=%d waypoints=%d "
        "half_width=%.2f m bounds=(%.2f, %.2f, %.2f, %.2f)",
        road_id, lane_id, len(wps), actual_half_width, *polygon.bounds,
    )

    return WorkZonePolygon(
        polygon=polygon,
        centerline_pts=centerline_pts,
        start_wp=start_wp,
        end_wp=end_wp,
        road_id=road_id,
        lane_id=lane_id,
        actual_half_width_m=actual_half_width,
    )


def actor_footprint_polygon(actor: carla.Actor) -> object:
    """
    Build a 2-D Shapely Polygon from the actor's world-space bounding box.

    Uses all 8 corners of the OBB projected to XY; the convex hull gives
    the correct footprint for any vehicle orientation and handles non-centred
    bounding boxes (bbox offset relative to actor origin).
    """
    _require_shapely()
    verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    pts_2d = [(v.x, v.y) for v in verts]
    return MultiPoint(pts_2d).convex_hull


def intersects_actor(workzone_polygon: object, actor: carla.Actor) -> bool:
    """Return True if the work-zone polygon overlaps the actor's 2D footprint."""
    return workzone_polygon.intersects(actor_footprint_polygon(actor))
