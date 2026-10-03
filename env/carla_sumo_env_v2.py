"""PPO V2 specialization of the existing CARLA-SUMO engine.

Simulator lifecycle, scenario rotation, SUMO synchronization, logging, and
fault recovery stay in :class:`env.carla_sumo_env.CarlaSumoEnv`.  This class
replaces only observation, reward, and terminal judgement.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from env.sumo_runtime_v2 import (
    MIN_SUMO_VERSION_V2,
    SUMO_RUNTIME_V2,
    parse_sumo_server_version_v2,
    pinned_sumo_binary_path_v2,
    prefer_sumo_python_tools_v2,
)

# The shared engine imports ``traci`` at module import time.  Prefer the V2
# package tools first so its Python client matches the pinned SUMO executable.
prefer_sumo_python_tools_v2()

import env.carla_sumo_env as legacy_env
from env.carla_sumo_env import CarlaSumoEnv
from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_DIM_V2,
    EgoSampleV2,
    LaneSegmentSampleV2,
    ObjectSampleV2,
    ObservationEncoderV2,
)
from logic.episode_termination_v2 import EpisodeTerminationCheckerV2
from logic.reward_v2 import RUNNING_V2, RewardCalculatorV2, RewardConfigV2
from logic.scenario_geometry_adapter_v2 import finish_line_from_config_v2
from sync.background_traffic_v2 import BackgroundTrafficSynchronizerV2


_OBS_DIM = DEFAULT_OBSERVATION_DIM_V2
_LANE_SAMPLE_SPACING_M_V2 = 5.0
_LANE_GRID_CELL_SIZE_M_V2 = 25.0


def _build_lane_spatial_grid_v2(
    samples: tuple[LaneSegmentSampleV2, ...],
) -> tuple[
    dict[tuple[int, int], tuple[LaneSegmentSampleV2, ...]],
    float,
]:
    """Index fixed-map lane midpoints once; validate identities once too."""
    mutable: dict[tuple[int, int], list[LaneSegmentSampleV2]] = {}
    seen_ids: set[str] = set()
    max_half_length_m = 0.0
    for sample in samples:
        segment_id = str(sample.segment_id)
        if not segment_id:
            raise ValueError("Lane segment IDs must be non-empty")
        if segment_id in seen_ids:
            raise ValueError(f"Duplicate lane segment identity: {segment_id}")
        seen_ids.add(segment_id)
        numeric = (
            sample.x,
            sample.y,
            sample.axis_heading_deg,
            sample.length_m,
            sample.width_m,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            continue
        if sample.length_m <= 0.0 or sample.width_m <= 0.0:
            continue
        key = (
            math.floor(float(sample.x) / _LANE_GRID_CELL_SIZE_M_V2),
            math.floor(float(sample.y) / _LANE_GRID_CELL_SIZE_M_V2),
        )
        mutable.setdefault(key, []).append(sample)
        max_half_length_m = max(
            max_half_length_m,
            0.5 * float(sample.length_m),
        )
    return (
        {key: tuple(value) for key, value in mutable.items()},
        max_half_length_m,
    )


def _query_lane_spatial_grid_v2(
    grid: dict[tuple[int, int], tuple[LaneSegmentSampleV2, ...]],
    *,
    x: float,
    y: float,
    radius_m: float,
    max_half_length_m: float,
) -> tuple[LaneSegmentSampleV2, ...]:
    """Return a conservative local candidate set for exact downstream filtering."""
    extent = float(radius_m) + float(max_half_length_m)
    min_x = math.floor((float(x) - extent) / _LANE_GRID_CELL_SIZE_M_V2)
    max_x = math.floor((float(x) + extent) / _LANE_GRID_CELL_SIZE_M_V2)
    min_y = math.floor((float(y) - extent) / _LANE_GRID_CELL_SIZE_M_V2)
    max_y = math.floor((float(y) + extent) / _LANE_GRID_CELL_SIZE_M_V2)
    candidates: list[LaneSegmentSampleV2] = []
    for cell_x in range(min_x, max_x + 1):
        for cell_y in range(min_y, max_y + 1):
            candidates.extend(grid.get((cell_x, cell_y), ()))
    return tuple(candidates)


class _ResetCompatibleObservationEncoderV2(ObservationEncoderV2):
    """Accept the legacy lifecycle's early reset, then bind the real ego."""

    def __init__(self) -> None:
        super().__init__()
        self.needs_initial_ego = True

    def reset(
        self,
        initial_ego: EgoSampleV2 | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if initial_ego is None:
            self.needs_initial_ego = True
            return
        super().reset(initial_ego, seed=seed)
        self.needs_initial_ego = False


class CarlaSumoEnvV2(CarlaSumoEnv):
    """Existing simulator engine with the official PPO V2 task contract."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reward_v2: RewardCalculatorV2 | None = None
        self._last_observation = np.zeros(_OBS_DIM, dtype=np.float32)
        self._observation_actor_ids_v2: dict[str, tuple[str | None, ...]] = {
            "workzone": (),
            "other_agents": (),
            "lanes": (),
        }
        self._lane_segments_v2_cache: tuple[LaneSegmentSampleV2, ...] = ()
        self._lane_segments_v2_map_name: str | None = None
        self._lane_segments_v2_grid: dict[
            tuple[int, int], tuple[LaneSegmentSampleV2, ...]
        ] = {}
        self._lane_segments_v2_max_half_length_m = 0.0

    def _new_agent_encoder(self) -> _ResetCompatibleObservationEncoderV2:
        return _ResetCompatibleObservationEncoderV2()

    def _launch_sumo_process(self) -> None:
        """Launch the project-pinned SUMO instead of a machine-wide binary."""
        with pinned_sumo_binary_path_v2():
            super()._launch_sumo_process()

    def _connect_traci(self) -> None:
        """Connect and fail fast if the server is not the pinned V2 release."""
        super()._connect_traci()
        if self._conn is None:
            raise legacy_env.SumoRuntimeFault("PPO V2 TraCI connection is missing")
        try:
            api_version, description = self._conn.getVersion()
            server_version = parse_sumo_server_version_v2(description)
        except Exception as exc:
            self._record_sumo_failure("v2_version_handshake", exc)
            self._terminate_sumo(reason="v2_version_handshake_failed")
            raise legacy_env.SumoRuntimeFault(
                f"Could not verify the PPO V2 SUMO server: {exc}"
            ) from exc
        if server_version < MIN_SUMO_VERSION_V2:
            error = RuntimeError(
                f"PPO V2 requires SUMO >= {'.'.join(map(str, MIN_SUMO_VERSION_V2))}; "
                f"connected to {description}"
            )
            self._record_sumo_failure("v2_version_mismatch", error)
            self._terminate_sumo(reason="v2_version_mismatch")
            raise legacy_env.SumoRuntimeFault(str(error))
        self._monitor.record_event(
            "sumo_runtime_v2_verified",
            api_version=int(api_version),
            server_version=".".join(map(str, server_version)),
            server_description=description,
            sumo_binary=str(SUMO_RUNTIME_V2.sumo_binary),
            traci_client=str(getattr(legacy_env.traci, "__file__", "unknown")),
        )

    def _resilient_build_sync_components(self) -> None:
        """Build the shared synchronizers with V2's bounded traffic queue."""
        if self.cfg.sumo is None:
            raise legacy_env.SumoRuntimeFault(
                f"{self.cfg.setting_id} is CARLA-only"
            )
        if not self._world or not self._conn:
            raise legacy_env.SumoRuntimeFault(
                "Cannot build synchronizers without CARLA world and TraCI connection"
            )
        net_file = self.cfg.sumo.net_file
        if not legacy_env.os.path.isabs(net_file):
            net_file = legacy_env.os.path.join(legacy_env._PROJECT_ROOT, net_file)
        net_offset = legacy_env.CarlaSumoCoordinateBridge.net_offset_from_net(
            net_file
        )
        self._bridge = legacy_env.CarlaSumoCoordinateBridge(
            net_offset=net_offset,
            lateral_shift=self.cfg.carla.lateral_shift,
        )
        self._ego_proxy = legacy_env.EgoProxySynchronizer(
            self._bridge,
            self._conn,
            self.cfg.sumo.ego_proxy_id,
        )
        self._bg_traffic = BackgroundTrafficSynchronizerV2(
            self._bridge,
            self._world,
            self._conn,
            worker_id=self._worker_id,
            route_depart_pos_ranges_m=(
                self.cfg.sumo.bg_route_depart_pos_ranges_m
            ),
            route_traffic_overrides=(
                self.cfg.sumo.bg_route_traffic_overrides
            ),
            actor_register=self._track_episode_actor,
            actor_unregister=self._untrack_episode_actor,
            spawn_attempt_callback=self._record_mirror_spawn_attempt,
        )
        self._monitor.record_event(
            "sync_components_built",
            net_file=net_file,
            net_offset=net_offset,
            traffic_capacity="active_plus_pending_v2",
            route_depart_pos_ranges_m=(
                self.cfg.sumo.bg_route_depart_pos_ranges_m
            ),
            route_traffic_overrides=(
                self.cfg.sumo.bg_route_traffic_overrides
            ),
        )

    def _reset_once(self, mode: str) -> np.ndarray:
        self._reward_v2 = None
        # The base lifecycle deliberately owns sensor attachment and geometry
        # injection.  Swap only its checker factory for the duration of reset.
        previous_checker = legacy_env.EpisodeTerminationChecker
        legacy_env.EpisodeTerminationChecker = EpisodeTerminationCheckerV2
        try:
            return super()._reset_once(mode)
        finally:
            legacy_env.EpisodeTerminationChecker = previous_checker

    def _build_observation(self) -> np.ndarray:
        if self._ego is None:
            return np.zeros(_OBS_DIM, dtype=np.float32)
        transform = self._ego.get_transform()
        velocity = self._ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        ego = EgoSampleV2(
            x=float(transform.location.x),
            y=float(transform.location.y),
            heading_deg=float(transform.rotation.yaw),
            speed_mps=float(speed),
        )
        encoder = self._agent_encoder
        if encoder.needs_initial_ego:
            encoder.reset(
                ego,
                seed=(self._worker_id * 1_000_003 + self._episode_id),
            )

        workzone_elements = [
            ObjectSampleV2(f"cone:{index}", "traffic_cone", float(x), float(y))
            for index, (x, y) in enumerate(self.cfg.workzone.traffic_cones)
        ]
        workzone_elements.extend(
            ObjectSampleV2(f"sign:{index}", "warning_sign", float(x), float(y))
            for index, (x, y) in enumerate(self.cfg.workzone.warning_signs)
        )
        other_agents: list[ObjectSampleV2] = []
        for sumo_id, actor in self.get_background_actor_map().items():
            try:
                if not actor.is_alive:
                    continue
                location = actor.get_location()
                other_agents.append(ObjectSampleV2(
                    f"sumo:{sumo_id}", "sumo_vehicle",
                    float(location.x), float(location.y),
                ))
            except Exception:
                continue
        for actor in self._pedestrians:
            try:
                if not actor.is_alive:
                    continue
                location = actor.get_location()
                other_agents.append(ObjectSampleV2(
                    f"ped:{actor.id}", "pedestrian",
                    float(location.x), float(location.y),
                ))
            except Exception:
                continue

        encoded = encoder.encode(
            ego,
            workzone_elements=workzone_elements,
            other_agents=other_agents,
            lane_segments=self._nearby_lane_segments_v2(
                ego,
                radius_m=encoder.spec.perception_radius_m,
            ),
        )
        observation = encoded.values
        self._observation_actor_ids_v2 = encoded.observation_ids
        if observation.shape != (_OBS_DIM,) or not np.all(np.isfinite(observation)):
            raise RuntimeError(
                f"Invalid PPO V2 observation shape/content: {observation.shape}"
            )
        return observation

    def _lane_segments_v2(self) -> tuple[LaneSegmentSampleV2, ...]:
        """Cache direction-free Driving-lane center segments for this Town."""
        if self._cmap is None:
            return ()
        map_name = str(getattr(self._cmap, "name", self.cfg.carla.town))
        if self._lane_segments_v2_cache and self._lane_segments_v2_map_name == map_name:
            return self._lane_segments_v2_cache

        samples: list[LaneSegmentSampleV2] = []
        waypoints = self._cmap.generate_waypoints(_LANE_SAMPLE_SPACING_M_V2)
        for index, waypoint in enumerate(waypoints):
            if waypoint.lane_type != legacy_env.carla.LaneType.Driving:
                continue
            transform = waypoint.transform
            location = transform.location
            segment_id = (
                f"road:{int(waypoint.road_id)}:section:"
                f"{int(getattr(waypoint, 'section_id', 0))}:lane:"
                f"{int(waypoint.lane_id)}:s:{float(getattr(waypoint, 's', index)):.3f}"
            )
            samples.append(LaneSegmentSampleV2(
                segment_id=segment_id,
                x=float(location.x),
                y=float(location.y),
                axis_heading_deg=float(transform.rotation.yaw),
                length_m=_LANE_SAMPLE_SPACING_M_V2,
                width_m=float(waypoint.lane_width),
            ))
        if not samples:
            raise RuntimeError(f"CARLA map {map_name!r} has no Driving-lane segments")
        self._lane_segments_v2_cache = tuple(samples)
        self._lane_segments_v2_map_name = map_name
        (
            self._lane_segments_v2_grid,
            self._lane_segments_v2_max_half_length_m,
        ) = _build_lane_spatial_grid_v2(self._lane_segments_v2_cache)
        self._monitor.record_event(
            "lane_segments_v2_cached",
            map=map_name,
            count=len(samples),
            spacing_m=_LANE_SAMPLE_SPACING_M_V2,
            grid_cells=len(self._lane_segments_v2_grid),
            grid_cell_size_m=_LANE_GRID_CELL_SIZE_M_V2,
        )
        return self._lane_segments_v2_cache

    def _nearby_lane_segments_v2(
        self,
        ego: EgoSampleV2,
        *,
        radius_m: float,
    ) -> tuple[LaneSegmentSampleV2, ...]:
        """Query only fixed-map grid cells that can intersect the ego radius."""
        self._lane_segments_v2()
        return _query_lane_spatial_grid_v2(
            self._lane_segments_v2_grid,
            x=float(ego.x),
            y=float(ego.y),
            radius_m=float(radius_m),
            max_half_length_m=self._lane_segments_v2_max_half_length_m,
        )

    def _initialize_reward_progress(self) -> None:
        if self._ego is None or self._destination_transform is None:
            raise RuntimeError("Cannot initialize PPO V2 reward before ego/destination")
        transform = self._ego.get_transform()
        location = transform.location
        finish = finish_line_from_config_v2(self.cfg)
        self._reward_v2 = RewardCalculatorV2(
            origin=(float(location.x), float(location.y)),
            finish_line=(finish.first, finish.second),
            config=RewardConfigV2(
                progress_budget=50.0,
                timeout_penalty=-50.0,
                goal_reward=100.0,
                failure_penalty=-100.0,
                average_speed_threshold_mps=5.0,
                average_speed_bonus=20.0,
            ),
        )
        self._reward_progress_source = "baseline_fixed_od_high_water_r1"

    def _compute_reward(
        self,
        terminated: bool,
        truncated: bool,
        success: bool,
        info: dict[str, Any],
    ) -> float:
        del success
        if self._reward_v2 is None or self._ego is None:
            raise RuntimeError("PPO V2 reward is not initialized")
        transform = self._ego.get_transform()
        location = transform.location
        velocity = self._ego.get_velocity()
        speed_mps = math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )
        reason = info.get("reason") if (terminated or truncated) else RUNNING_V2
        result = self._reward_v2.step(
            (float(location.x), float(location.y)),
            speed_mps=float(speed_mps),
            terminal_reason=reason,
        )
        progress_reward = result.reward if result.reason == RUNNING_V2 else 0.0
        info["reward_progress_source"] = "baseline_fixed_od_high_water_r1"
        info["reward_progress_bonus"] = float(progress_reward)
        info["reward_progress_fraction"] = float(result.best_progress)
        info["reward_normalized_progress"] = float(result.normalized_progress)
        info["reward_cumulative_progress"] = float(
            result.cumulative_progress_reward
        )
        info["reward_step_cost"] = 0.0
        info["reward_average_speed_mps"] = float(result.average_speed_mps)
        info["reward_speed_bonus"] = float(result.speed_bonus)
        info["reward_terminal_base"] = float(result.terminal_base_reward)
        info["reward_total"] = float(result.reward)
        return float(result.reward)

    @property
    def observation_actor_ids(self) -> dict[str, tuple[str | None, ...]]:
        """Raw slot IDs for logs; learned V2 features ignore pseudonym bits."""
        return self._observation_actor_ids_v2


__all__ = ["CarlaSumoEnvV2", "_OBS_DIM"]
