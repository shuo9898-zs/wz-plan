"""Fixed-scale ego and object-perception features shared by all scenarios."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


Point2D = Tuple[float, float]

OBSERVATION_CONTRACT_VERSION = "ego_objects_no_route_v3"
EGO_STATE_DIM = 7
FREE_SPACE_RAY_ANGLES_DEG = (
    -135.0,
    -90.0,
    -45.0,
    0.0,
    45.0,
    90.0,
    135.0,
    180.0,
)
FREE_SPACE_DIM = len(FREE_SPACE_RAY_ANGLES_DEG)
MAX_DYNAMIC_AGENTS = 8
DYNAMIC_AGENT_DIM = 8
OBSERVATION_DIM = EGO_STATE_DIM + MAX_DYNAMIC_AGENTS * DYNAMIC_AGENT_DIM


def encode_ego_state(
    *,
    ego_yaw_deg: float,
    ego_speed_mps: float,
    acceleration_mps2: float,
    yaw_rate_deg_s: float,
    previous_speed_action: float,
    previous_heading_action: float,
    base_heading_deg: float,
    max_ego_speed_mps: float,
    max_acceleration_mps2: float,
    yaw_rate_limit_deg_s: float,
) -> np.ndarray:
    """Return the seven owner-approved ego features; no OD/route is exposed."""
    if (
        max_ego_speed_mps <= 0.0
        or max_acceleration_mps2 <= 0.0
        or yaw_rate_limit_deg_s <= 0.0
    ):
        raise ValueError("Observation normalization scales must be positive")
    base_error = math.radians(
        (float(base_heading_deg) - float(ego_yaw_deg) + 180.0) % 360.0
        - 180.0
    )
    return np.asarray(
        [
            np.clip(float(ego_speed_mps) / max_ego_speed_mps, 0.0, 1.0),
            np.clip(float(acceleration_mps2) / max_acceleration_mps2, -1.0, 1.0),
            np.clip(float(yaw_rate_deg_s) / yaw_rate_limit_deg_s, -1.0, 1.0),
            np.clip(float(previous_speed_action), -1.0, 1.0),
            np.clip(float(previous_heading_action), -1.0, 1.0),
            math.sin(base_error),
            math.cos(base_error),
        ],
        dtype=np.float32,
    )


class WorkzoneFreeSpaceEncoder:
    """Cache static WZ geometry and vectorize its per-frame ray intersections."""

    def __init__(
        self,
        *,
        radius_m: float,
        forbidden_boundary: Sequence[Point2D] | None = None,
        safe_left_boundary: Sequence[Point2D] | None = None,
        safe_right_boundary: Sequence[Point2D] | None = None,
    ) -> None:
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("radius_m must be finite and positive")
        has_forbidden = forbidden_boundary is not None
        has_safe = safe_left_boundary is not None or safe_right_boundary is not None
        if has_forbidden and has_safe:
            raise ValueError("Provide either forbidden geometry or safe geometry")
        self.radius_m = float(radius_m)
        self._safe = bool(has_safe)
        self._left: list[Point2D] | None = None
        self._right: list[Point2D] | None = None
        self._entry_origin: Point2D | None = None
        self._entry_direction: Point2D | None = None
        self._exit_origin: Point2D | None = None
        self._exit_direction: Point2D | None = None

        if has_safe:
            if not safe_left_boundary or not safe_right_boundary:
                raise ValueError("Safe geometry requires both side boundaries")
            left = _points(safe_left_boundary)
            right = _points(safe_right_boundary)
            if len(left) != len(right) or len(left) < 2:
                raise ValueError(
                    "Safe geometry requires matching sides with at least two points"
                )
            self._left, self._right = left, right
            self._polygon = left + list(reversed(right))
            centres = [
                ((l[0] + r[0]) * 0.5, (l[1] + r[1]) * 0.5)
                for l, r in zip(left, right)
            ]
            self._entry_origin = centres[0]
            self._entry_direction = _unit_direction(centres[0], centres[1])
            self._exit_origin = centres[-1]
            self._exit_direction = _unit_direction(centres[-2], centres[-1])
            # Entrance and exit are open. Only side walls restrict legal space.
            segments = list(zip(left, left[1:])) + list(zip(right, right[1:]))
        elif has_forbidden:
            polygon = _points(forbidden_boundary or ())
            if len(polygon) < 3:
                raise ValueError("Forbidden geometry requires at least three points")
            self._polygon = polygon
            segments = list(zip(polygon, polygon[1:] + polygon[:1]))
        else:
            self._polygon = []
            segments = []

        self._segment_starts = np.asarray(
            [segment[0] for segment in segments], dtype=np.float64
        ).reshape(-1, 2)
        self._segment_vectors = np.asarray(
            [
                (end[0] - start[0], end[1] - start[1])
                for start, end in segments
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        polygon_segments = list(
            zip(self._polygon, self._polygon[1:] + self._polygon[:1])
        )
        self._polygon_starts = np.asarray(
            [segment[0] for segment in polygon_segments], dtype=np.float64
        ).reshape(-1, 2)
        self._polygon_vectors = np.asarray(
            [
                (end[0] - start[0], end[1] - start[1])
                for start, end in polygon_segments
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        self._ray_angles_rad = np.radians(
            np.asarray(FREE_SPACE_RAY_ANGLES_DEG, dtype=np.float64)
        )

    def encode(
        self, *, ego_x: float, ego_y: float, ego_yaw_deg: float
    ) -> np.ndarray:
        """Return normalized WZ-constrained free distance on eight rays."""
        if self._segment_starts.shape[0] == 0:
            return np.ones(FREE_SPACE_DIM, dtype=np.float32)
        origin = (float(ego_x), float(ego_y))
        corridor_phase: str | None = None
        if self._safe:
            assert self._left is not None and self._right is not None
            corridor_phase = self._corridor_phase(origin)
            if (
                corridor_phase == "inside"
                and not self._point_in_polygon(origin)
            ):
                return np.zeros(FREE_SPACE_DIM, dtype=np.float32)
        elif self._point_in_polygon(origin):
            return np.zeros(FREE_SPACE_DIM, dtype=np.float32)

        angles = self._ray_angles_rad + math.radians(float(ego_yaw_deg))
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        offset = self._segment_starts - np.asarray(origin, dtype=np.float64)
        segment = self._segment_vectors

        # Shapes: directions=(R,2), segments=(S,2), results=(R,S).
        denominator = (
            directions[:, 0, None] * segment[None, :, 1]
            - directions[:, 1, None] * segment[None, :, 0]
        )
        parallel = np.abs(denominator) <= 1e-10
        safe_denominator = np.where(parallel, 1.0, denominator)
        ray_distance = (
            offset[None, :, 0] * segment[None, :, 1]
            - offset[None, :, 1] * segment[None, :, 0]
        ) / safe_denominator
        segment_fraction = (
            offset[None, :, 0] * directions[:, 1, None]
            - offset[None, :, 1] * directions[:, 0, None]
        ) / safe_denominator
        valid = (
            ~parallel
            & (ray_distance >= -1e-9)
            & (segment_fraction >= -1e-9)
            & (segment_fraction <= 1.0 + 1e-9)
        )
        candidates = np.where(valid, np.maximum(0.0, ray_distance), np.inf)

        # Preserve the scalar implementation's collinear-boundary behaviour.
        collinear = parallel & (
            np.abs(
                offset[None, :, 0] * directions[:, 1, None]
                - offset[None, :, 1] * directions[:, 0, None]
            )
            <= 1e-9
        )
        start_projection = (
            offset[None, :, 0] * directions[:, 0, None]
            + offset[None, :, 1] * directions[:, 1, None]
        )
        end_offset = offset + segment
        end_projection = (
            end_offset[None, :, 0] * directions[:, 0, None]
            + end_offset[None, :, 1] * directions[:, 1, None]
        )
        projection_low = np.minimum(start_projection, end_projection)
        projection_high = np.maximum(start_projection, end_projection)
        collinear_valid = collinear & (projection_high >= -1e-9)
        collinear_distance = np.maximum(0.0, projection_low)
        candidates = np.minimum(
            candidates,
            np.where(collinear_valid, collinear_distance, np.inf),
        )

        if self._safe and corridor_phase != "inside":
            # A side-wall intersection can enter the legal corridor from its
            # open approach/exit rather than leave legal space.  Inspect the
            # state immediately after each ordered hit and keep only the first
            # transition into illegal space.  There are only 2*(N-1) S3 side
            # segments, so this exact topology check remains inexpensive.
            clearance = np.full(FREE_SPACE_DIM, self.radius_m, dtype=np.float64)
            for ray_index, direction in enumerate(directions):
                for distance in np.sort(candidates[ray_index]):
                    if not np.isfinite(distance) or distance > self.radius_m:
                        break
                    probe_distance = float(distance) + 1e-5
                    probe = (
                        origin[0] + float(direction[0]) * probe_distance,
                        origin[1] + float(direction[1]) * probe_distance,
                    )
                    phase = self._corridor_phase(probe)
                    legal = phase != "inside" or self._point_in_polygon(probe)
                    if not legal:
                        clearance[ray_index] = max(0.0, float(distance))
                        break
        else:
            clearance = np.min(candidates, axis=1)
        clearance = np.where(np.isfinite(clearance), clearance, self.radius_m)
        return np.clip(clearance / self.radius_m, 0.0, 1.0).astype(np.float32)

    def _corridor_phase(self, point: Point2D) -> str:
        assert self._entry_origin is not None
        assert self._entry_direction is not None
        assert self._exit_origin is not None
        assert self._exit_direction is not None
        entry_dot = (
            (point[0] - self._entry_origin[0]) * self._entry_direction[0]
            + (point[1] - self._entry_origin[1]) * self._entry_direction[1]
        )
        if entry_dot < 0.0:
            return "before"
        exit_dot = (
            (point[0] - self._exit_origin[0]) * self._exit_direction[0]
            + (point[1] - self._exit_origin[1]) * self._exit_direction[1]
        )
        return "after" if exit_dot > 0.0 else "inside"

    def _point_in_polygon(self, point: Point2D) -> bool:
        """Boundary-inclusive vectorized even/odd containment."""
        if self._polygon_starts.shape[0] == 0:
            return False
        px, py = point
        start = self._polygon_starts
        edge = self._polygon_vectors
        rel_x = px - start[:, 0]
        rel_y = py - start[:, 1]
        edge_len_sq = edge[:, 0] ** 2 + edge[:, 1] ** 2
        cross = edge[:, 0] * rel_y - edge[:, 1] * rel_x
        nonzero = edge_len_sq > 0.0
        fraction = np.divide(
            rel_x * edge[:, 0] + rel_y * edge[:, 1],
            edge_len_sq,
            out=np.zeros_like(edge_len_sq),
            where=nonzero,
        )
        on_boundary = (
            nonzero
            & (np.abs(cross) <= 1e-8)
            & (fraction >= -1e-9)
            & (fraction <= 1.0 + 1e-9)
        )
        if bool(np.any(on_boundary)):
            return True
        end_y = start[:, 1] + edge[:, 1]
        crosses = (start[:, 1] > py) != (end_y > py)
        safe_dy = np.where(crosses, edge[:, 1], 1.0)
        x_cross = start[:, 0] + (py - start[:, 1]) * edge[:, 0] / safe_dy
        return bool(np.count_nonzero(crosses & (px < x_cross)) % 2)


def encode_legal_free_space(
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw_deg: float,
    radius_m: float,
    forbidden_boundary: Sequence[Point2D] | None = None,
    safe_left_boundary: Sequence[Point2D] | None = None,
    safe_right_boundary: Sequence[Point2D] | None = None,
) -> np.ndarray:
    """One-shot convenience wrapper; production reuses the cached encoder."""
    return WorkzoneFreeSpaceEncoder(
        radius_m=radius_m,
        forbidden_boundary=forbidden_boundary,
        safe_left_boundary=safe_left_boundary,
        safe_right_boundary=safe_right_boundary,
    ).encode(ego_x=ego_x, ego_y=ego_y, ego_yaw_deg=ego_yaw_deg)


def _points(points: Sequence[Point2D]) -> list[Point2D]:
    result = [(float(x), float(y)) for x, y in points]
    if len(result) >= 2 and result[0] == result[-1]:
        result.pop()
    return result


def _cross(first: Point2D, second: Point2D) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _ray_segment_distance(
    origin: Point2D,
    direction: Point2D,
    start: Point2D,
    end: Point2D,
) -> float | None:
    segment = (end[0] - start[0], end[1] - start[1])
    offset = (start[0] - origin[0], start[1] - origin[1])
    denominator = _cross(direction, segment)
    if abs(denominator) <= 1e-10:
        if abs(_cross(offset, direction)) > 1e-9:
            return None
        # Collinear: the ray first reaches whichever endpoint is closest in
        # its non-negative direction.
        projections = [
            (point[0] - origin[0]) * direction[0]
            + (point[1] - origin[1]) * direction[1]
            for point in (start, end)
        ]
        non_negative = [value for value in projections if value >= -1e-9]
        return max(0.0, min(non_negative)) if non_negative else None
    ray_distance = _cross(offset, segment) / denominator
    segment_fraction = _cross(offset, direction) / denominator
    if ray_distance < -1e-9 or not -1e-9 <= segment_fraction <= 1.0 + 1e-9:
        return None
    return max(0.0, ray_distance)


def _point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
    """Boundary-inclusive even/odd containment without a Shapely dependency."""
    px, py = point
    inside = False
    for start, end in zip(polygon, list(polygon[1:]) + [polygon[0]]):
        ax, ay = start
        bx, by = end
        edge = (bx - ax, by - ay)
        rel = (px - ax, py - ay)
        edge_len_sq = edge[0] * edge[0] + edge[1] * edge[1]
        if edge_len_sq > 0.0 and abs(_cross(edge, rel)) <= 1e-8:
            fraction = (rel[0] * edge[0] + rel[1] * edge[1]) / edge_len_sq
            if -1e-9 <= fraction <= 1.0 + 1e-9:
                return True
        crosses = (ay > py) != (by > py)
        if crosses:
            x_cross = ax + (py - ay) * (bx - ax) / (by - ay)
            if px < x_cross:
                inside = not inside
    return inside


def _unit_direction(start: Point2D, end: Point2D) -> Point2D:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("Corridor centre samples must be distinct")
    return dx / length, dy / length


def _corridor_phase(
    point: Point2D,
    left: Sequence[Point2D],
    right: Sequence[Point2D],
) -> str:
    centres = [
        ((l[0] + r[0]) * 0.5, (l[1] + r[1]) * 0.5)
        for l, r in zip(left, right)
    ]

    def unit(start: Point2D, end: Point2D) -> Point2D:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            raise ValueError("Corridor centre samples must be distinct")
        return dx / length, dy / length

    entry_direction = unit(centres[0], centres[1])
    exit_direction = unit(centres[-2], centres[-1])
    entry_dot = (
        (point[0] - centres[0][0]) * entry_direction[0]
        + (point[1] - centres[0][1]) * entry_direction[1]
    )
    if entry_dot < 0.0:
        return "before"
    exit_dot = (
        (point[0] - centres[-1][0]) * exit_direction[0]
        + (point[1] - centres[-1][1]) * exit_direction[1]
    )
    return "after" if exit_dot > 0.0 else "inside"
