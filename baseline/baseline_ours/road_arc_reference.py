"""Static CARLA-world road-arc references used by baseline_ours."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple


Point2D = Tuple[float, float]
REFERENCE_SCHEMA_VERSION = 1
REFERENCE_PATH = Path(__file__).with_name("road_arc_references_v1.json")


@dataclass(frozen=True)
class ArcProjection:
    point: Point2D
    distance_along_m: float
    distance_to_reference_m: float


class RoadArcReference:
    """Project CARLA XY positions onto one ordered road polyline."""

    def __init__(self, points: Sequence[Point2D]) -> None:
        cleaned: list[Point2D] = []
        for index, raw in enumerate(points):
            if len(raw) != 2:
                raise ValueError(f"reference point {index} must contain x and y")
            point = (float(raw[0]), float(raw[1]))
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"reference point {index} must be finite")
            if not cleaned or math.dist(cleaned[-1], point) > 1e-8:
                cleaned.append(point)
        if len(cleaned) < 2:
            raise ValueError("road-arc reference requires at least two distinct points")

        cumulative = [0.0]
        for first, second in zip(cleaned, cleaned[1:]):
            cumulative.append(cumulative[-1] + math.dist(first, second))
        if cumulative[-1] <= 1e-6:
            raise ValueError("road-arc reference has zero length")
        self.points = tuple(cleaned)
        self.cumulative_lengths_m = tuple(cumulative)
        self.length_m = float(cumulative[-1])

    def project(self, position: Point2D) -> ArcProjection:
        px, py = float(position[0]), float(position[1])
        if not math.isfinite(px) or not math.isfinite(py):
            raise ValueError("projection position must be finite")
        best: tuple[float, float, Point2D] | None = None
        for index, (first, second) in enumerate(zip(self.points, self.points[1:])):
            dx, dy = second[0] - first[0], second[1] - first[1]
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                continue
            fraction = max(
                0.0,
                min(1.0, ((px - first[0]) * dx + (py - first[1]) * dy) / length_sq),
            )
            projected = (first[0] + fraction * dx, first[1] + fraction * dy)
            distance_sq = (px - projected[0]) ** 2 + (py - projected[1]) ** 2
            segment_length = math.sqrt(length_sq)
            distance_along = self.cumulative_lengths_m[index] + fraction * segment_length
            candidate = (distance_sq, distance_along, projected)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("road-arc reference contains no projectable segment")
        return ArcProjection(
            point=best[2],
            distance_along_m=float(best[1]),
            distance_to_reference_m=math.sqrt(best[0]),
        )


@lru_cache(maxsize=1)
def _reference_document() -> dict:
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"missing road-arc reference file: {REFERENCE_PATH}")
    document = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if int(document.get("schema_version", -1)) != REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported road-arc reference schema")
    if document.get("coordinate_system") != "carla_world_xy_metres":
        raise ValueError("road-arc references must use CARLA world XY metres")
    return document


def reference_for_setting(setting_id: str) -> RoadArcReference:
    """Load the static reference bound to a stable runtime setting ID."""
    key = str(setting_id).replace("\\", "/").lower()
    document = _reference_document()
    try:
        reference_id = str(document["settings"][key])
        raw = document["references"][reference_id]["points_xy_m"]
    except KeyError as error:
        raise KeyError(f"no road-arc reference for setting {key!r}") from error
    return RoadArcReference(raw)


__all__ = [
    "ArcProjection",
    "REFERENCE_PATH",
    "REFERENCE_SCHEMA_VERSION",
    "RoadArcReference",
    "reference_for_setting",
]
