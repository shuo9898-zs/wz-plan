"""Center-point ego judgement for the controlled PPO V2 ablation.

This module deliberately reuses the current simulator hooks, collision
handling, scenario geometry, finish lines, and reason vocabulary.  The only
experimental change is the work-zone geometry reference: the pure V2 core is
called without an ego footprint, so it judges the current ego centre and the
segment swept by consecutive ego-centre samples.
"""
from __future__ import annotations

import math
from typing import Any

from logic.episode_termination_v2 import EpisodeTerminationCheckerV2
from logic.reward_v2 import GOAL_REACHED_V2, RUNNING_V2, TIMEOUT_V2


CENTER_EGO_JUDGEMENT_V2 = "ego_center_point_segment"
CENTER_TERMINATION_CONTRACT_VERSION_V2 = "scenario_geometry_center_trajectory_v2"


class CenterPointEpisodeTerminationCheckerV2(EpisodeTerminationCheckerV2):
    """Current V2 task geometry with the previous centre-point ego semantics."""

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
        scenario = str(self._cfg.scenario_id).lower()
        collision = None
        collision_info: dict[str, Any] = {}

        if self._collision_flag:
            with self._collision_lock:
                reason = self._collision_reason
                if reason == "collision_walker":
                    collision = "jaywalker"
                elif reason in {"collision_sumo_vehicle", "collision_carla_vehicle"}:
                    collision = (
                        "sumo_vehicle" if scenario != "s4" else "workzone_object"
                    )
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
            # This intentionally remains the existing background-vehicle OBB
            # collision detector.  The ablation changes only the ego reference
            # used by authored work-zone geometry.
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

        # S3's authored union remains the road authority inside the work-zone;
        # CARLA lane metadata does not describe this temporary corridor.
        off_road = False if scenario == "s3" else self._is_offroad(ego)
        decision = self._core_v2.judge(
            previous_position=previous,
            current_position=current,
            episode_step=self._step,
            collision=collision,
            off_road=off_road,
            ego_footprint=None,
        )
        self._previous_ego_xy = current

        dist = math.hypot(
            current[0] - float(dest.location.x),
            current[1] - float(dest.location.y),
        )
        info: dict[str, Any] = {
            "step": self._step,
            "reason": decision.reason,
            "dist_to_goal": dist,
            "workzone_geometry_reference": CENTER_EGO_JUDGEMENT_V2,
        }
        info.update(collision_info)
        if not decision.terminated or decision.reason == RUNNING_V2:
            return False, False, False, info
        success = decision.reason == GOAL_REACHED_V2
        timeout = decision.reason == TIMEOUT_V2
        return True, timeout, success, info


__all__ = [
    "CENTER_EGO_JUDGEMENT_V2",
    "CENTER_TERMINATION_CONTRACT_VERSION_V2",
    "CenterPointEpisodeTerminationCheckerV2",
]
