"""Visible, isolated PPO debug run for S3/WZ1/origin2 only.

This entry point deliberately does not edit the S3 configuration or reuse the
three-server trainer.  It cycles layouts A/B/C, binds owner-authored origin 2,
collects exactly the requested number of *valid* transitions, and performs one
PPO update after every short rollout.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from baseline.PPO.callbacks import format_live_update
from baseline.PPO.rollout_plan import describe_plan, plan_single
from baseline.PPO.runtime_v2 import (
    V2_DEFAULT_PPO_EPOCHS,
    build_or_load_ppo,
    configure_initial_config,
    save_ppo,
)
from baseline.PPO.valid_rollout import collect_valid_rollouts
from config.scenario_catalog import list_setting_ids
from config.scenario_config import EgoSpawnPointConfig, load_scenario
from config.scenario_selector import CoverageSelector
from tools.preflight import validate_scenario
from tools.show_workzone_polygons_v2 import (
    DRIVABLE_GREEN_V2,
    FINISH_YELLOW_V2,
    ORIGIN_CYAN_V2,
    VisualizationGeometryV2,
    build_visualization_geometry_v2,
    hatch_segments_v2,
)


SETTINGS_V2 = tuple(f"s3/wz1/{layout}" for layout in ("a", "b", "c"))
ORIGIN_INDEX_V2 = 2
DEFAULT_TOTAL_VALID_STEPS_V2 = 10_000
DEFAULT_ROLLOUT_STEPS_V2 = 1_000
OVERLAY_LIFETIME_S_V2 = 1.5
OVERLAY_REFRESH_PERIOD_S_V2 = 1.0
OVERLAY_CLOSE_TIMEOUT_S_V2 = 5.0


class FixedOriginCoverageSelectorV2(CoverageSelector):
    """Cycle the three layouts while reporting one fixed origin in ``info``."""

    def __init__(
        self,
        settings: Sequence[str],
        *,
        origin_index: int = ORIGIN_INDEX_V2,
    ) -> None:
        super().__init__(settings, repeats=1)
        if origin_index < 0:
            raise ValueError("origin_index must be non-negative")
        self.origin_index = int(origin_index)

    def on_episode_end(self, info: dict) -> None:
        info.setdefault("origin_index", self.origin_index)
        info.setdefault(
            "ticket_id",
            f"{info.get('setting_id', 'unknown')}#origin{self.origin_index}",
        )
        super().on_episode_end(info)


class FixedOriginBinderV2:
    """Bind one authored origin after layout selection and before reset."""

    def __init__(
        self,
        settings: Sequence[str],
        *,
        origin_index: int = ORIGIN_INDEX_V2,
        heading_mode: str = "authored",
    ) -> None:
        if heading_mode not in {"authored", "entry-midpoint"}:
            raise ValueError("heading_mode must be authored or entry-midpoint")
        self.origin_index = int(origin_index)
        self.heading_mode = heading_mode
        self._points: dict[str, EgoSpawnPointConfig] = {}
        self._entry_midpoints: dict[str, tuple[float, float]] = {}
        for setting in settings:
            cfg = load_scenario(setting)
            if not 0 <= self.origin_index < len(cfg.origin.spawn_points):
                raise ValueError(
                    f"{setting} has no origin index {self.origin_index}"
                )
            left = cfg.workzone.corridor_left_boundary_points or []
            right = cfg.workzone.corridor_right_boundary_points or []
            if not left or not right:
                raise ValueError(f"{setting} has no S3 corridor entrance")
            self._points[setting] = cfg.origin.spawn_points[self.origin_index]
            self._entry_midpoints[setting] = (
                0.5 * (float(left[0][0]) + float(right[0][0])),
                0.5 * (float(left[0][1]) + float(right[0][1])),
            )
        self.gym_env: Any | None = None

    def attach(self, gym_env: Any) -> None:
        self.gym_env = gym_env

    def resolved_point(self, setting_id: str) -> EgoSpawnPointConfig:
        point = self._points[setting_id]
        if self.heading_mode == "authored":
            return point
        entry_x, entry_y = self._entry_midpoints[setting_id]
        heading = math.degrees(math.atan2(entry_y - point.y, entry_x - point.x))
        return replace(point, yaw_deg=float(heading))

    def bind(self, setting_id: str) -> None:
        if self.gym_env is None:
            raise RuntimeError("FixedOriginBinderV2 is not attached")
        cfg = self.gym_env.engine.cfg
        if cfg.setting_id != setting_id:
            raise RuntimeError(
                f"Engine/config mismatch: {cfg.setting_id} != {setting_id}"
            )
        cfg.origin.spawn_points = (self.resolved_point(setting_id),)


class VisibleExplorationWrapperV2(gym.Wrapper):
    """Chase the ego and draw the exact S3 drivable union in native CARLA."""

    def __init__(
        self,
        env: gym.Env,
        *,
        geometries: Mapping[str, VisualizationGeometryV2],
        total_valid_steps: int,
        follow_camera: bool,
        realtime: bool,
        terminal_pause_seconds: float,
    ) -> None:
        super().__init__(env)
        self._geometries = dict(geometries)
        self._follow_camera_enabled = bool(follow_camera)
        self._realtime = bool(realtime)
        self._terminal_pause_seconds = max(0.0, float(terminal_pause_seconds))
        # CARLA 0.9.15 has no delete/clear API for DebugHelper primitives.
        # Use a short finite lifetime and refresh only once per simulated
        # second (ten steps at 10 Hz).  close() restores async mode through the
        # wrapped env and waits for the final primitives to expire.
        control_dt_s = float(self.env.engine.cfg.episode.sim_dt)
        self._overlay_lifetime_s = OVERLAY_LIFETIME_S_V2
        self._overlay_refresh_interval_steps = max(
            1,
            int(round(OVERLAY_REFRESH_PERIOD_S_V2 / control_dt_s)),
        )
        self._overlay_refresh_countdown = 0
        self._overlay_key: tuple[int, tuple] | None = None
        self._camera_warning_printed = False

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # A reset may consume many simulation ticks, so refresh immediately
        # instead of waiting for the previous episode's countdown.
        self._overlay_refresh_countdown = 0
        self._draw_overlay_if_needed()
        self._follow_camera()
        return result

    def step(self, action):
        started = time.perf_counter()
        result = self.env.step(action)
        self._draw_overlay_if_needed()
        self._follow_camera()
        terminated = bool(result[2])
        truncated = bool(result[3])
        if terminated or truncated:
            if self._terminal_pause_seconds:
                time.sleep(self._terminal_pause_seconds)
        if self._realtime:
            target = float(self.env.engine.cfg.episode.sim_dt)
            remaining = target - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        return result

    def close(self) -> None:
        world = getattr(self.env.engine, "world", None)
        overlay_was_drawn = self._overlay_key is not None
        try:
            super().close()
        finally:
            if overlay_was_drawn and world is not None:
                _wait_for_debug_overlay_expiry_v2(
                    world,
                    life_time_s=self._overlay_lifetime_s,
                    wall_timeout_s=OVERLAY_CLOSE_TIMEOUT_S_V2,
                )

    def _follow_camera(self) -> None:
        if not self._follow_camera_enabled:
            return
        try:
            import carla

            ego = self.env.engine.ego
            world = self.env.engine.world
            if ego is None or world is None:
                return
            transform = ego.get_transform()
            forward = transform.get_forward_vector()
            location = carla.Location(
                x=float(transform.location.x - 8.0 * forward.x),
                y=float(transform.location.y - 8.0 * forward.y),
                z=float(transform.location.z + 4.0),
            )
            world.get_spectator().set_transform(
                carla.Transform(
                    location,
                    carla.Rotation(
                        pitch=-18.0,
                        yaw=float(transform.rotation.yaw),
                        roll=0.0,
                    ),
                )
            )
        except Exception as error:
            if not self._camera_warning_printed:
                print(f"WARN spectator follow disabled after error: {error}", flush=True)
                self._camera_warning_printed = True
            self._follow_camera_enabled = False

    def _draw_overlay_if_needed(self) -> None:
        try:
            import carla

            world = self.env.engine.world
            cfg = self.env.engine.cfg
            if world is None or cfg.setting_id not in self._geometries:
                return
            geometry = self._geometries[cfg.setting_id]
            vertices_key = tuple(
                (layer.label, layer.vertices) for layer in geometry.layers
            )
            key = (
                id(world),
                vertices_key,
                geometry.finish_line.first,
                geometry.finish_line.second,
                geometry.origin,
            )
            if self._overlay_key == key and self._overlay_refresh_countdown > 0:
                self._overlay_refresh_countdown -= 1
                return
            carla_map = world.get_map()

            def location(point: tuple[float, float], z_offset: float = 0.18):
                query = carla.Location(x=float(point[0]), y=float(point[1]), z=1.0)
                try:
                    waypoint = carla_map.get_waypoint(
                        query,
                        project_to_road=True,
                        lane_type=carla.LaneType.Any,
                    )
                except TypeError:
                    waypoint = carla_map.get_waypoint(query, project_to_road=True)
                ground_z = waypoint.transform.location.z if waypoint is not None else 0.0
                return carla.Location(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=float(ground_z + z_offset),
                )

            green = carla.Color(*DRIVABLE_GREEN_V2)
            for layer in geometry.layers:
                for start, end in zip(
                    layer.vertices, layer.vertices[1:] + layer.vertices[:1]
                ):
                    world.debug.draw_line(
                        location(start), location(end), thickness=0.10,
                        color=green, life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
                for start, end in hatch_segments_v2(
                    layer.vertices, 2.0, False, holes=layer.holes
                ):
                    world.debug.draw_line(
                        location(start, 0.19), location(end, 0.19), thickness=0.025,
                        color=green, life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
            yellow = carla.Color(*FINISH_YELLOW_V2)
            world.debug.draw_line(
                location(geometry.finish_line.first, 0.22),
                location(geometry.finish_line.second, 0.22),
                thickness=0.18,
                color=yellow,
                life_time=self._overlay_lifetime_s,
                persistent_lines=False,
            )
            if geometry.origin is not None:
                cyan = carla.Color(*ORIGIN_CYAN_V2)
                world.debug.draw_point(
                    location(geometry.origin, 0.30), size=0.22, color=cyan,
                    life_time=self._overlay_lifetime_s, persistent_lines=False,
                )
                world.debug.draw_string(
                    location(geometry.origin, 0.55), "S3 WZ1 origin2",
                    draw_shadow=True, color=cyan,
                    life_time=self._overlay_lifetime_s, persistent_lines=False,
                )
            self._overlay_key = key
            self._overlay_refresh_countdown = (
                self._overlay_refresh_interval_steps - 1
            )
        except Exception as error:
            print(f"WARN drivable overlay could not be drawn: {error}", flush=True)


class ExplorationTraceCallbackV2(BaseCallback):
    """Write every valid step and print compact episode/rollout diagnostics."""

    FIELDNAMES = (
        "valid_step",
        "policy_update",
        "setting_id",
        "origin_index",
        "episode_step",
        "ego_x",
        "ego_y",
        "displacement_from_spawn_m",
        "planar_step_distance_m",
        "path_distance_m",
        "planar_speed_mps",
        "planar_average_speed_mps",
        "ego_speed_mps",
        "reward_average_3d_speed_mps",
        "action_target_speed_norm",
        "action_target_yaw_rate_norm",
        "target_speed_mps",
        "target_yaw_rate_deg_s",
        "steer",
        "throttle",
        "brake",
        "reward",
        "episode_return",
        "normalized_progress",
        "done",
        "reason",
    )

    def __init__(
        self,
        *,
        trace_path: Path,
        spawn_xy_by_setting: Mapping[str, tuple[float, float]],
        live_every: int,
        control_dt_s: float,
    ) -> None:
        super().__init__()
        self.trace_path = Path(trace_path)
        self.spawn_xy_by_setting = dict(spawn_xy_by_setting)
        self.live_every = max(0, int(live_every))
        self.control_dt_s = float(control_dt_s)
        if not math.isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and positive")
        self.outcomes: Counter[tuple[str, str]] = Counter()
        self.infrastructure_faults: Counter[str] = Counter()
        self.policy_update = 0
        self._file = None
        self._writer = None
        self._episode_return = 0.0
        self._path_distance = 0.0
        self._previous_xy: tuple[float, float] | None = None

    def _on_training_start(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.trace_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        actions = self.locals.get("clipped_actions", self.locals.get("actions", []))
        if not infos:
            return True
        info = infos[0]
        reward = float(rewards[0]) if len(rewards) else 0.0
        done = bool(dones[0]) if len(dones) else False
        action = np.asarray(actions[0], dtype=np.float64).reshape(-1)
        setting = str(info.get("setting_id", "unknown"))
        episode_step = int(info.get("episode_step", 0))
        x = _finite_or_nan(info.get("ego_x"))
        y = _finite_or_nan(info.get("ego_y"))
        current_xy = (x, y)
        if episode_step <= 1 or self._previous_xy is None:
            origin = self.spawn_xy_by_setting.get(setting, current_xy)
            step_distance = math.hypot(x - origin[0], y - origin[1])
            self._path_distance = step_distance
            self._episode_return = 0.0
        else:
            step_distance = math.hypot(
                x - self._previous_xy[0], y - self._previous_xy[1]
            )
            self._path_distance += step_distance
        self._previous_xy = current_xy
        self._episode_return += reward
        origin = self.spawn_xy_by_setting.get(setting, current_xy)
        displacement = math.hypot(x - origin[0], y - origin[1])
        row = {
            "valid_step": int(self.model.num_timesteps),
            "policy_update": self.policy_update,
            "setting_id": setting,
            "origin_index": int(info.get("origin_index", ORIGIN_INDEX_V2)),
            "episode_step": episode_step,
            "ego_x": x,
            "ego_y": y,
            "displacement_from_spawn_m": displacement,
            "planar_step_distance_m": step_distance,
            "path_distance_m": self._path_distance,
            "planar_speed_mps": step_distance / self.control_dt_s,
            "planar_average_speed_mps": (
                self._path_distance / max(1, episode_step) / self.control_dt_s
            ),
            "ego_speed_mps": _finite_or_nan(info.get("ego_speed_mps")),
            "reward_average_3d_speed_mps": _finite_or_nan(
                info.get("reward_average_speed_mps")
            ),
            "action_target_speed_norm": float(action[0]) if action.size > 0 else math.nan,
            "action_target_yaw_rate_norm": float(action[1]) if action.size > 1 else math.nan,
            "target_speed_mps": _finite_or_nan(
                info.get("control_policy_target_speed_mps")
            ),
            "target_yaw_rate_deg_s": _finite_or_nan(
                info.get("control_target_yaw_rate_deg_s")
            ),
            "steer": _finite_or_nan(info.get("control_steer")),
            "throttle": _finite_or_nan(info.get("control_throttle")),
            "brake": _finite_or_nan(info.get("control_brake")),
            "reward": reward,
            "episode_return": self._episode_return,
            "normalized_progress": _finite_or_nan(
                info.get("reward_normalized_progress")
            ),
            "done": done,
            "reason": str(info.get("reason", "running")),
        }
        if self._writer is not None:
            self._writer.writerow(row)
        if self.live_every and (
            done or episode_step == 1 or self.model.num_timesteps % self.live_every == 0
        ):
            print(
                f"{format_live_update(info, reward, done)} "
                f"origin=2 displacement={displacement:.3f}m "
                f"path={self._path_distance:.3f}m "
                f"planar_avg={row['planar_average_speed_mps']:.3f}m/s "
                f"reward_avg3d={row['reward_average_3d_speed_mps']:.3f}m/s "
                f"policy_update={self.policy_update}",
                flush=True,
            )
            if self._file is not None:
                self._file.flush()
        if done:
            reason = str(info.get("reason", "unknown"))
            self.outcomes[(setting, reason)] += 1
            self._previous_xy = None
        return True

    def record_infrastructure_fault(self, info: dict) -> None:
        reason = str(info.get("reason", "unknown"))
        self.infrastructure_faults[reason] += 1
        print(
            f"INFRA_RETRY setting={info.get('setting_id', 'unknown')} "
            f"origin=2 reason={reason}",
            flush=True,
        )

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def _wait_for_debug_overlay_expiry_v2(
    world: Any,
    *,
    life_time_s: float,
    wall_timeout_s: float,
) -> bool:
    """Wait for finite-lived server primitives after env.close enables async."""
    try:
        start = float(world.get_snapshot().timestamp.elapsed_seconds)
        target = start + float(life_time_s) + 0.1
        deadline = time.monotonic() + float(wall_timeout_s)
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            snapshot = world.wait_for_tick(min(0.5, remaining))
            if float(snapshot.timestamp.elapsed_seconds) >= target:
                return True
    except Exception as error:
        print(f"WARN debug overlay expiry wait failed: {error}", flush=True)
    return False


def _finite_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _close_debug_resources_v2(
    *,
    callback: Any,
    training_started: bool,
    vec_env: Any | None,
    visualizer: Any | None,
) -> None:
    """Close every owned resource even when an earlier close hook fails."""
    try:
        if training_started:
            callback.on_training_end()
    finally:
        try:
            if vec_env is not None:
                vec_env.close()
        finally:
            if visualizer is not None:
                visualizer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-valid-steps", type=int, default=DEFAULT_TOTAL_VALID_STEPS_V2)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS_V2)
    parser.add_argument("--ppo-epochs", type=int, default=V2_DEFAULT_PPO_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--live-log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--carla-port", type=int, default=2040)
    parser.add_argument("--tm-port", type=int, default=8040)
    parser.add_argument("--sumo-port", type=int, default=8853)
    parser.add_argument(
        "--origin-heading",
        choices=("authored", "entry-midpoint"),
        default="authored",
        help=(
            "authored reproduces yaw=0; entry-midpoint changes only the runtime "
            "spawn yaw to about 18.52 degrees without editing JSON"
        ),
    )
    parser.add_argument(
        "--show-traffic-cones",
        dest="show_traffic_cones",
        action="store_true",
        default=False,
        help=(
            "Spawn physical CARLA cone props for viewing; disabled by default "
            "so the debug dynamics match the previous no-cone training run"
        ),
    )
    parser.add_argument(
        "--no-traffic-cones",
        dest="show_traffic_cones",
        action="store_false",
    )
    parser.add_argument("--cones-only", action="store_true")
    parser.add_argument(
        "--follow-camera", dest="follow_camera", action="store_true", default=True
    )
    parser.add_argument(
        "--no-follow-camera", dest="follow_camera", action="store_false"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Throttle simulator steps to at most 10 Hz wall-clock for easier viewing",
    )
    parser.add_argument("--terminal-pause-seconds", type=float, default=0.25)
    parser.add_argument("--model", help="Optional compatible V2 PPO checkpoint")
    parser.add_argument(
        "--run-dir", default="runs/s3_wz1_origin2_debug_v2"
    )
    parser.add_argument("--save-model", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_valid_steps < 1 or args.rollout_steps < 1:
        raise ValueError("step counts must be positive")
    if args.total_valid_steps % args.rollout_steps != 0:
        raise ValueError("--total-valid-steps must be divisible by --rollout-steps")
    if args.terminal_pause_seconds < 0.0:
        raise ValueError("--terminal-pause-seconds must be non-negative")

    preflight = validate_scenario("s3")
    if not preflight.ok:
        for error in preflight.errors:
            print(f"ERROR {error}")
        return 2
    runnable = set(list_setting_ids("s3", runnable_only=True))
    missing = [setting for setting in SETTINGS_V2 if setting not in runnable]
    if missing:
        print("ERROR missing runnable settings", missing)
        return 2

    configs = [load_scenario(setting) for setting in SETTINGS_V2]
    plan = plan_single(
        [cfg.episode.max_steps for cfg in configs],
        repeats=1,
        n_steps_override=args.rollout_steps,
    )
    selector = FixedOriginCoverageSelectorV2(SETTINGS_V2)
    binder = FixedOriginBinderV2(
        SETTINGS_V2,
        origin_index=ORIGIN_INDEX_V2,
        heading_mode=args.origin_heading,
    )
    geometries = {
        setting: build_visualization_geometry_v2(
            load_scenario(setting), origin_index=ORIGIN_INDEX_V2
        )
        for setting in SETTINGS_V2
    }
    spawn_xy = {
        setting: (
            float(binder.resolved_point(setting).x),
            float(binder.resolved_point(setting).y),
        )
        for setting in SETTINGS_V2
    }
    initial = configs[0]
    configure_initial_config(
        initial,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        sumo_port=args.sumo_port,
        no_rendering=False,
    )

    run_dir = Path(args.run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    trace_path = run_dir / f"exploration_{timestamp}.csv"
    visualizer = None
    if args.show_traffic_cones:
        from tools.show_traffic_cones import WorkZonePropVisualizer

        visualizer = WorkZonePropVisualizer(
            SETTINGS_V2[0],
            host=initial.carla.host,
            carla_port=args.carla_port,
            cones_only=args.cones_only,
            draw_finish_line=False,
            load_town=True,
            move_spectator=False,
        )

    def episode_setup(setting_id: str) -> None:
        binder.bind(setting_id)
        if visualizer is not None:
            visualizer.switch(setting_id)

    def make_env():
        from env.gym_wrapper_v2 import CarlaSumoGymEnv

        base_env = CarlaSumoGymEnv(
            scenario=SETTINGS_V2[0],
            config=initial,
            worker_id=0,
            no_rendering_mode=False,
            scenario_selector=selector,
            episode_setup_callback=episode_setup,
        )
        binder.attach(base_env)
        return VisibleExplorationWrapperV2(
            base_env,
            geometries=geometries,
            total_valid_steps=args.total_valid_steps,
            follow_camera=args.follow_camera,
            realtime=args.realtime,
            terminal_pause_seconds=args.terminal_pause_seconds,
        )

    print("debug_scope settings=", SETTINGS_V2, "origin_index=2", flush=True)
    print("rollout_plan", describe_plan(plan), flush=True)
    print(
        f"training valid_steps={args.total_valid_steps} "
        f"short_rollouts={args.total_valid_steps // args.rollout_steps} "
        f"steps_per_rollout={args.rollout_steps}",
        flush=True,
    )
    for setting in SETTINGS_V2:
        point = binder.resolved_point(setting)
        print(
            f"spawn {setting} origin2=({point.x:.3f},{point.y:.3f}) "
            f"yaw={point.yaw_deg:.6f} mode={args.origin_heading}",
            flush=True,
        )
    print(f"trace_csv={trace_path}", flush=True)

    vec_env = None
    callback = ExplorationTraceCallbackV2(
        trace_path=trace_path,
        spawn_xy_by_setting=spawn_xy,
        live_every=args.live_log_every,
        control_dt_s=float(initial.episode.sim_dt),
    )
    training_started = False
    try:
        vec_env = DummyVecEnv([make_env])
        model = build_or_load_ppo(
            vec_env,
            plan,
            model_path=args.model,
            ppo_epochs=args.ppo_epochs,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            target_kl=args.target_kl,
        )
        print(
            f"compute PPO_device={model.device} CARLA_rendering=enabled "
            f"camera_follow={args.follow_camera}",
            flush=True,
        )
        training_end_step, callback = model._setup_learn(
            args.total_valid_steps,
            callback,
            reset_num_timesteps=(args.model is None),
            tb_log_name="s3_wz1_origin2_debug_v2",
        )
        callback.on_training_start(locals(), globals())
        training_started = True
        rollout_count = args.total_valid_steps // args.rollout_steps
        for update_index in range(1, rollout_count + 1):
            callback.policy_update = update_index - 1
            print(
                f"ROLLOUT {update_index:02d}/{rollout_count:02d} START "
                f"valid_target={args.rollout_steps} policy_version={update_index - 1}",
                flush=True,
            )
            complete = collect_valid_rollouts(
                model,
                vec_env,
                callback,
                model.rollout_buffer,
                n_rollout_steps=args.rollout_steps,
            )
            if (
                not complete
                or not model.rollout_buffer.full
                or int(model.rollout_buffer.pos) != args.rollout_steps
            ):
                raise RuntimeError(f"rollout {update_index} did not fill")
            model._update_current_progress_remaining(
                model.num_timesteps, training_end_step
            )
            epochs_before = int(model._n_updates)
            model.train()
            callback.policy_update = update_index
            model.logger.dump(step=model.num_timesteps)
            print(
                f"PPO_UPDATE {update_index:02d}/{rollout_count:02d} END "
                f"valid_steps={model.num_timesteps} "
                f"optimizer_epochs={int(model._n_updates) - epochs_before}",
                flush=True,
            )
        if args.save_model:
            save_ppo(model, run_dir / "model")
    finally:
        _close_debug_resources_v2(
            callback=callback,
            training_started=training_started,
            vec_env=vec_env,
            visualizer=visualizer,
        )

    print("coverage", selector.completed_counts, flush=True)
    print("outcomes", dict(callback.outcomes), flush=True)
    print("infrastructure_faults", dict(callback.infrastructure_faults), flush=True)
    print(f"trace_csv={trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
