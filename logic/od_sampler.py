"""Manual-only origin/destination definitions.

Origins are absolute CARLA transforms authored for the materialized setting.
They may be shared at Work-zone level or overridden by an individual a/b/c
layout. Destinations are finite finish-line segments with the same override
semantics. No distance-based geometry, CARLA waypoint projection, predefined
map spawn point, or yaw perturbation is used here.
"""
from __future__ import annotations

import random

import carla

from config.scenario_config import ScenarioConfig


class OriginDestinationSampler:
    """Select manual O and expose the configured finish-line D."""

    def __init__(self, config: ScenarioConfig) -> None:
        self._cfg = config

    def sample_origin(self, mode: str = "train") -> carla.Transform:
        """Uniformly select one absolute spawn transform for this setting."""
        del mode  # Manual candidates are used identically in train/demo/eval.
        points = self._cfg.origin.spawn_points
        if not points:
            raise ValueError(
                f"{self._cfg.setting_id} has no manual origin.spawn_points; "
                "automatic origin sampling has been removed"
            )
        point = random.choice(points)
        return carla.Transform(
            carla.Location(x=point.x, y=point.y, z=point.z),
            carla.Rotation(
                pitch=point.pitch_deg,
                roll=point.roll_deg,
                yaw=point.yaw_deg,
            ),
        )

    def sample_destination(self, mode: str = "train") -> carla.Transform:
        """Return the finish-line midpoint for observations/progress reward.

        Episode success is evaluated against the full finite line segment by
        ``EpisodeTerminationChecker``; this midpoint is not a radius target.
        """
        del mode
        line = self._cfg.destination.finish_line
        if line is None:
            raise ValueError(
                f"{self._cfg.setting_id} has no manual destination.line_cm; "
                "automatic destination sampling has been removed"
            )
        x = (line.start[0] + line.end[0]) / 2.0
        y = (line.start[1] + line.end[1]) / 2.0
        return carla.Transform(
            carla.Location(x=x, y=y, z=0.5),
            carla.Rotation(yaw=self._cfg.carla.road_heading_deg),
        )
