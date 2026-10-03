"""CARLA-facing adapter for the pure PPO V2 termination contract.

The collision sensor, background-vehicle OBB test, and CARLA road query are
reused from the existing environment integration.  Only the task judgement
and reason vocabulary are replaced by :mod:`logic.termination_checker_v2`.
"""
from __future__ import annotations

import math
from typing import Any

from config.scenario_config import ScenarioConfig
from logic.scenario_geometry_adapter_v2 import (
    build_forbidden_area_from_config_v2,
    finish_line_from_config_v2,
)
from logic.termination_checker import EpisodeTerminationChecker, _actor_footprint_xy
from logic.termination_checker_v2 import (
    Footprint2DV2,
    S3DrivableAreaV2,
    TerminationCheckerV2,
    swept_footprint_v2,
)
from logic.reward_v2 import (
    GOAL_REACHED_V2,
    RUNNING_V2,
    TIMEOUT_V2,
)


class EpisodeTerminationCheckerV2(EpisodeTerminationChecker):
    """Keep the simulator hooks while delegating outcomes to V2 geometry."""

    def __init__(self, config: ScenarioConfig, carla_map: Any | None = None) -> None:
        super().__init__(config, carla_map=carla_map)
        self._core_v2: TerminationCheckerV2 | None = None
        self._origin_v2: tuple[float, float] | None = None
        self._origin_heading_v2: float | None = None
        self._previous_ego_footprint_v2: Footprint2DV2 | None = None

    def attach_collision_sensor(self, world: Any, ego: Any, **kwargs: Any) -> None:
        super().attach_collision_sensor(world, ego, **kwargs)
        location = ego.get_location()
        self._origin_v2 = (float(location.x), float(location.y))
        self._origin_heading_v2 = float(ego.get_transform().rotation.yaw)
        self._previous_ego_xy = self._origin_v2
        self._previous_ego_footprint_v2 = tuple(_actor_footprint_xy(ego))
        self._rebuild_core_v2()

    def set_forbidden_polygon(self, polygon: Any) -> None:
        super().set_forbidden_polygon(polygon)
        self._rebuild_core_v2()

    def _rebuild_core_v2(self) -> None:
        if self._origin_v2 is None:
            return
        scenario = str(self._cfg.scenario_id).lower()
        finish = finish_line_from_config_v2(self._cfg)
        forbidden = None
        drivable = None
        if scenario == "s3":
            left = self._wz.corridor_left_boundary_points
            right = self._wz.corridor_right_boundary_points
            if not left or not right:
                raise ValueError(f"{self._cfg.setting_id} has no S3 corridor boundaries")
            drivable = S3DrivableAreaV2.build(
                origin=self._origin_v2,
                origin_heading_deg=self._origin_heading_v2,
                left_boundary=left,
                right_boundary=right,
                finish_line=finish,
                exit_left_boundary=self._wz.exit_left_boundary_points,
                exit_right_boundary=self._wz.exit_right_boundary_points,
                boundary_tolerance_m=self._wz.boundary_tolerance_m,
            )
        elif scenario == "s2":
            if self._forbidden_polygon is None:
                return
            forbidden = tuple(
                (float(point[0]), float(point[1]))
                for point in self._forbidden_polygon.exterior.coords
            )
        else:
            forbidden = build_forbidden_area_from_config_v2(self._cfg).polygon
        self._core_v2 = TerminationCheckerV2(
            scenario_id=scenario,
            origin=self._origin_v2,
            finish_line=finish,
            max_episode_steps=self._cfg.episode.max_steps,
            forbidden_polygon=forbidden,
            s3_drivable_area=drivable,
            boundary_tolerance_m=self._wz.boundary_tolerance_m,
        )

    def tick(
        self,
        ego: Any,
        dest: Any,
        bg_actor_map: dict | None = None,
    ) -> tuple[bool, bool, bool, dict]:
        if self._core_v2 is None:
            self._rebuild_core_v2()
        if self._core_v2 is None:
            raise RuntimeError("PPO V2 termination geometry is not initialized")

        self._step += 1
        location = ego.get_location()
        current = (float(location.x), float(location.y))
        previous = self._previous_ego_xy or self._origin_v2 or current
        current_footprint = tuple(_actor_footprint_xy(ego))
        previous_footprint = self._previous_ego_footprint_v2 or current_footprint
        swept_footprint = swept_footprint_v2(
            previous_footprint,
            current_footprint,
        )
        scenario = str(self._cfg.scenario_id).lower()
        collision = None
        collision_info: dict[str, Any] = {}

        if self._collision_flag:
            with self._collision_lock:
                reason = self._collision_reason
                if reason == "collision_walker":
                    collision = "jaywalker"
                elif reason in {"collision_sumo_vehicle", "collision_carla_vehicle"}:
                    collision = "sumo_vehicle" if scenario != "s4" else "workzone_object"
                else:
                    collision = "workzone_object"
                collision_info = {
                    "collision_actor_id": self._collision_actor_id,
                    "collision_actor_type": self._collision_actor_type or "unknown",
                    "collision_actor_role": self._collision_actor_role or "(none)",
                    "collision_detector": self._collision_detector,
                    "collision_owner": self._collision_owner,
                    "collision_counterpart": self._collision_counterpart,
                    "collision_event_count": self._collision_event_count,
                    "collision_legacy_reason": reason,
                }
                if self._collision_frame is not None:
                    collision_info["collision_frame"] = self._collision_frame
                if self._collision_impulse_magnitude is not None:
                    collision_info["collision_impulse_magnitude"] = (
                        self._collision_impulse_magnitude
                    )
        elif bg_actor_map:
            hit = self._bg_collision(ego, bg_actor_map)
            if hit is not None:
                sumo_id, actor_id, distance_m = hit
                collision = "sumo_vehicle"
                collision_info = {
                    "collision_sumo_id": sumo_id,
                    "collision_actor_id": actor_id,
                    "collision_distance_m": distance_m,
                    "collision_detector": "obb_overlap",
                    "collision_owner": "sumo",
                    "collision_counterpart": "vehicle",
                    "collision_overlap_method": "2d_obb_sat",
                }

        # S3's authored union is the road authority inside the work-zone; CARLA
        # lane metadata does not describe this temporary drivable corridor.
        off_road = False if scenario == "s3" else self._is_offroad(ego)
        decision = self._core_v2.judge(
            previous_position=previous,
            current_position=current,
            episode_step=self._step,
            collision=collision,
            off_road=off_road,
            ego_footprint=swept_footprint,
        )
        self._previous_ego_xy = current
        self._previous_ego_footprint_v2 = current_footprint

        dist = math.hypot(
            current[0] - float(dest.location.x),
            current[1] - float(dest.location.y),
        )
        info: dict[str, Any] = {
            "step": self._step,
            "reason": decision.reason,
            "dist_to_goal": dist,
            "workzone_geometry_reference": "swept_ego_obb",
        }
        info.update(collision_info)
        if not decision.terminated or decision.reason == RUNNING_V2:
            return False, False, False, info
        success = decision.reason == GOAL_REACHED_V2
        timeout = decision.reason == TIMEOUT_V2
        return True, timeout, success, info


__all__ = ["EpisodeTerminationCheckerV2"]
