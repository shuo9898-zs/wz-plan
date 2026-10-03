"""Local heading reference for an ordered, open-ended work-zone corridor.

The corridor boundaries are authored as matching cross-sections ordered in
the ego vehicle's direction of travel.  This module converts those pairs to a
piecewise-linear centerline once per episode, then provides a small look-ahead
heading reference without consulting CARLA's road graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class _CenterlineSegment:
    start: Point2D
    dx: float
    dy: float
    length: float
    length_sq: float
    start_distance: float


class CorridorHeadingTracker:
    """Track the nearest centerline segment and return its look-ahead heading.

    A short local segment window prevents the reference from jumping between
    unrelated parts of a curved or self-near corridor.  At 10 Hz an ego cannot
    normally traverse the window in one step; the first query after every
    reset searches the entire centerline.
    """

    def __init__(
        self,
        left_boundary: Sequence[Point2D],
        right_boundary: Sequence[Point2D],
        *,
        lookahead_m: float = 4.0,
    ) -> None:
        if len(left_boundary) != len(right_boundary) or len(left_boundary) < 2:
            raise ValueError(
                "Corridor heading requires matching left/right boundaries "
                "with at least two points"
            )
        if not math.isfinite(lookahead_m) or lookahead_m <= 0.0:
            raise ValueError("Corridor heading lookahead_m must be positive")

        centerline: list[Point2D] = []
        for left, right in zip(left_boundary, right_boundary):
            lx, ly = float(left[0]), float(left[1])
            rx, ry = float(right[0]), float(right[1])
            if not all(math.isfinite(v) for v in (lx, ly, rx, ry)):
                raise ValueError("Corridor boundaries must contain finite points")
            point = ((lx + rx) * 0.5, (ly + ry) * 0.5)
            # Ignore duplicate adjacent cross-sections instead of constructing
            # a zero-length segment whose heading would be undefined.
            if not centerline or math.dist(centerline[-1], point) > 1e-9:
                centerline.append(point)
        if len(centerline) < 2:
            raise ValueError("Corridor centerline must contain a non-zero segment")

        segments: list[_CenterlineSegment] = []
        distance = 0.0
        for start, end in zip(centerline, centerline[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length_sq = dx * dx + dy * dy
            length = math.sqrt(length_sq)
            segments.append(
                _CenterlineSegment(
                    start=start,
                    dx=dx,
                    dy=dy,
                    length=length,
                    length_sq=length_sq,
                    start_distance=distance,
                )
            )
            distance += length

        self._centerline = tuple(centerline)
        self._segments = tuple(segments)
        self._total_length = distance
        self._lookahead_m = float(lookahead_m)
        self.reset()

    @property
    def centerline(self) -> tuple[Point2D, ...]:
        """Precomputed centerline, exposed read-only for diagnostics/tests."""
        return self._centerline

    def reset(self) -> None:
        """Forget segment/angle continuity from the previous episode."""
        self._last_segment_index: int | None = None
        self._previous_heading_deg: float | None = None

    def heading_at(self, x: float, y: float) -> float:
        """Return a continuous CARLA yaw reference at ``(x, y)`` in degrees."""
        x = float(x)
        y = float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Ego position must be finite")

        segment_index, segment_fraction = self._nearest_projection(x, y)
        self._last_segment_index = segment_index
        segment = self._segments[segment_index]
        progress = segment.start_distance + segment_fraction * segment.length

        ahead = self._point_at_distance(
            min(self._total_length, progress + self._lookahead_m)
        )
        # Aim from the *actual ego position* to the look-ahead point.  Using
        # the projected centreline point here would provide only a tangent and
        # would never correct accumulated lateral drift in a narrow curve.
        # This pure-pursuit-style chord equals the tangent when centred, while
        # naturally steering back toward the corridor when displaced.
        dx = ahead[0] - x
        dy = ahead[1] - y

        raw_fraction = (
            (x - segment.start[0]) * segment.dx
            + (y - segment.start[1]) * segment.dy
        ) / segment.length_sq
        beyond_open_exit = (
            segment_index == len(self._segments) - 1 and raw_fraction >= 1.0
        )
        if beyond_open_exit or dx * dx + dy * dy <= 1e-12:
            # At or beyond the open exit, keep extending its authored tangent
            # instead of steering back toward the final centreline point.
            dx = segment.dx
            dy = segment.dy

        raw_heading = math.degrees(math.atan2(dy, dx))
        if self._previous_heading_deg is None:
            heading = raw_heading
        else:
            delta = (raw_heading - self._previous_heading_deg + 180.0) % 360.0 - 180.0
            heading = self._previous_heading_deg + delta
        self._previous_heading_deg = heading
        return heading

    def _nearest_projection(self, x: float, y: float) -> tuple[int, float]:
        if self._last_segment_index is None:
            indices = range(len(self._segments))
        else:
            # Two segments of backward tolerance allow small reverse motion;
            # four forward segments comfortably exceed one 10-Hz motion step.
            start = max(0, self._last_segment_index - 2)
            stop = min(len(self._segments), self._last_segment_index + 5)
            indices = range(start, stop)

        best_index = 0
        best_fraction = 0.0
        best_distance_sq = math.inf
        for index in indices:
            segment = self._segments[index]
            rel_x = x - segment.start[0]
            rel_y = y - segment.start[1]
            fraction = (rel_x * segment.dx + rel_y * segment.dy) / segment.length_sq
            fraction = min(1.0, max(0.0, fraction))
            nearest_x = segment.start[0] + fraction * segment.dx
            nearest_y = segment.start[1] + fraction * segment.dy
            distance_sq = (x - nearest_x) ** 2 + (y - nearest_y) ** 2
            if distance_sq < best_distance_sq:
                best_index = index
                best_fraction = fraction
                best_distance_sq = distance_sq
        return best_index, best_fraction

    def _point_at_distance(self, distance: float) -> Point2D:
        distance = min(self._total_length, max(0.0, distance))
        for index, segment in enumerate(self._segments):
            end_distance = segment.start_distance + segment.length
            if distance <= end_distance or index == len(self._segments) - 1:
                fraction = (distance - segment.start_distance) / segment.length
                fraction = min(1.0, max(0.0, fraction))
                return (
                    segment.start[0] + fraction * segment.dx,
                    segment.start[1] + fraction * segment.dy,
                )
        return self._centerline[-1]


class CorridorProgressTracker:
    """Project positions onto ordered centreline arc length.

    Progress is measured in metres from the authored entrance.  It is
    intentionally unbounded on the first and last segment tangents, giving
    negative progress on the approach and values greater than ``total_length``
    after the exit.  This keeps the dense forward reward continuous through
    all three open-corridor phases.
    """

    def __init__(
        self,
        left_boundary: Sequence[Point2D],
        right_boundary: Sequence[Point2D],
    ) -> None:
        heading_geometry = CorridorHeadingTracker(
            left_boundary, right_boundary, lookahead_m=4.0
        )
        self._segments = heading_geometry._segments
        self._total_length = heading_geometry._total_length
        self.reset()

    @property
    def total_length(self) -> float:
        return self._total_length

    def reset(self) -> None:
        self._last_segment_index: int | None = None

    def progress_at(self, x: float, y: float) -> float:
        x = float(x)
        y = float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Ego position must be finite")

        if self._last_segment_index is None:
            indices = range(len(self._segments))
        else:
            start = max(0, self._last_segment_index - 2)
            stop = min(len(self._segments), self._last_segment_index + 5)
            indices = range(start, stop)

        best_index = 0
        best_fraction = 0.0
        best_distance_sq = math.inf
        for index in indices:
            segment = self._segments[index]
            raw_fraction = (
                (x - segment.start[0]) * segment.dx
                + (y - segment.start[1]) * segment.dy
            ) / segment.length_sq
            fraction = raw_fraction
            if index != 0:
                fraction = max(0.0, fraction)
            if index != len(self._segments) - 1:
                fraction = min(1.0, fraction)
            nearest_x = segment.start[0] + fraction * segment.dx
            nearest_y = segment.start[1] + fraction * segment.dy
            distance_sq = (x - nearest_x) ** 2 + (y - nearest_y) ** 2
            if distance_sq < best_distance_sq:
                best_index = index
                best_fraction = fraction
                best_distance_sq = distance_sq

        self._last_segment_index = best_index
        segment = self._segments[best_index]
        return segment.start_distance + best_fraction * segment.length
