"""Visible PPO debug run over every S3 setting and authored origin.

The episode order is fixed and repeats as
``WZ1 -> WZ2 -> WZ3``, ``a -> b -> c``, ``origin 0 -> 1 -> 2``.
The native CARLA spectator is never moved by this entry point, so the user can
inspect the complete work-zone and SUMO traffic from a manually chosen view.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

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
from baseline.PPO.train_s3_wz1_origin2_debug_v2 import (
    OVERLAY_CLOSE_TIMEOUT_S_V2,
    OVERLAY_LIFETIME_S_V2,
    OVERLAY_REFRESH_PERIOD_S_V2,
    _close_debug_resources_v2,
    _wait_for_debug_overlay_expiry_v2,
)
from baseline.PPO.valid_rollout import collect_valid_rollouts
from baseline.PPO.validate_allSwithoneExe import (
    EpisodeTicket,
    OrderedOriginSelector,
    OriginBinder,
)
from config.scenario_catalog import list_setting_ids
from config.scenario_config import load_scenario
from tools.preflight import validate_scenario
from tools.show_workzone_polygons_v2 import (
    DRIVABLE_GREEN_V2,
    FINISH_YELLOW_V2,
    ORIGIN_CYAN_V2,
    VisualizationGeometryV2,
    build_visualization_geometry_v2,
    hatch_segments_v2,
)


SETTINGS_V2 = tuple(
    f"s3/wz{workzone}/{layout}"
    for workzone in range(1, 4)
    for layout in ("a", "b", "c")
)
ORIGINS_PER_SETTING_V2 = 3
TICKET_IDS_V2 = tuple(
    f"{setting}#origin{origin_index}"
    for setting in SETTINGS_V2
    for origin_index in range(ORIGINS_PER_SETTING_V2)
)
DEFAULT_TOTAL_VALID_STEPS_V2 = 10_000
DEFAULT_ROLLOUT_STEPS_V2 = 1_000


class S3AllDebugWrapperV2(gym.Wrapper):
    """Draw the active origin-specific union without moving the spectator."""

    def __init__(
        self,
        env: gym.Env,
        *,
        selector: OrderedOriginSelector,
        geometries_by_ticket: Mapping[str, VisualizationGeometryV2],
    ) -> None:
        super().__init__(env)
        self._selector = selector
        self._geometries_by_ticket = dict(geometries_by_ticket)
        control_dt_s = float(self.env.engine.cfg.episode.sim_dt)
        self._overlay_lifetime_s = OVERLAY_LIFETIME_S_V2
        self._overlay_refresh_interval_steps = max(
            1,
            int(round(OVERLAY_REFRESH_PERIOD_S_V2 / control_dt_s)),
        )
        self._overlay_refresh_countdown = 0
        self._overlay_key: tuple[Any, ...] | None = None
        self._active_ticket: EpisodeTicket | None = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._active_ticket = self._selector.current_ticket
        self._overlay_refresh_countdown = 0
        self._draw_overlay_if_needed()
        tagged = dict(info)
        self._tag_info(tagged)
        return observation, tagged

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._draw_overlay_if_needed()
        tagged = dict(info)
        self._tag_info(tagged)
        return observation, reward, terminated, truncated, tagged

    def close(self) -> None:
        engine = getattr(self.env, "engine", None)
        world = getattr(engine, "world", None)
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

    def _tag_info(self, info: dict[str, Any]) -> None:
        if self._active_ticket is None:
            return
        info.setdefault("origin_index", self._active_ticket.origin_index)
        info.setdefault("ticket_id", self._active_ticket.ticket_id)

    def _draw_overlay_if_needed(self) -> None:
        try:
            import carla

            ticket = self._active_ticket
            world = self.env.engine.world
            if ticket is None or world is None:
                return
            geometry = self._geometries_by_ticket[ticket.ticket_id]
            vertices_key = tuple(
                (layer.label, layer.vertices, layer.holes)
                for layer in geometry.layers
            )
            key = (
                id(world),
                ticket.ticket_id,
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
                query = carla.Location(
                    x=float(point[0]), y=float(point[1]), z=1.0
                )
                try:
                    waypoint = carla_map.get_waypoint(
                        query,
                        project_to_road=True,
                        lane_type=carla.LaneType.Any,
                    )
                except TypeError:
                    waypoint = carla_map.get_waypoint(
                        query, project_to_road=True
                    )
                ground_z = (
                    waypoint.transform.location.z
                    if waypoint is not None
                    else 0.0
                )
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
                        location(start),
                        location(end),
                        thickness=0.10,
                        color=green,
                        life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
                for start, end in hatch_segments_v2(
                    layer.vertices, 2.0, False, holes=layer.holes
                ):
                    world.debug.draw_line(
                        location(start, 0.19),
                        location(end, 0.19),
                        thickness=0.025,
                        color=green,
                        life_time=self._overlay_lifetime_s,
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
                label = (
                    f"{ticket.setting_id.upper()} origin{ticket.origin_index}"
                )
                world.debug.draw_point(
                    location(geometry.origin, 0.30),
                    size=0.22,
                    color=cyan,
                    life_time=self._overlay_lifetime_s,
                    persistent_lines=False,
                )
                world.debug.draw_string(
                    location(geometry.origin, 0.55),
                    label,
                    draw_shadow=True,
                    color=cyan,
                    life_time=self._overlay_lifetime_s,
                    persistent_lines=False,
                )
            self._overlay_key = key
            self._overlay_refresh_countdown = (
                self._overlay_refresh_interval_steps - 1
            )
        except Exception as error:
            print(
                f"WARN S3 debug overlay could not be drawn: {error}",
                flush=True,
            )


class S3AllExplorationTraceCallbackV2(BaseCallback):
    """Write ticket-aware exploration telemetry for all 27 combinations."""

    FIELDNAMES = (
        "valid_step",
        "policy_update",
        "ticket_id",
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
        spawn_xy_by_ticket: Mapping[str, tuple[float, float]],
        live_every: int,
        control_dt_s: float,
    ) -> None:
        super().__init__()
        self.trace_path = Path(trace_path)
        self.spawn_xy_by_ticket = dict(spawn_xy_by_ticket)
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
        self._writer = csv.DictWriter(
            self._file, fieldnames=self.FIELDNAMES
        )
        self._writer.writeheader()
        self._file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        actions = self.locals.get(
            "clipped_actions", self.locals.get("actions", [])
        )
        if not infos:
            return True
        info = infos[0]
        reward = float(rewards[0]) if len(rewards) else 0.0
        done = bool(dones[0]) if len(dones) else False
        action = np.asarray(actions[0], dtype=np.float64).reshape(-1)
        setting = str(info.get("setting_id", "unknown"))
        origin_index = int(info.get("origin_index", -1))
        ticket_id = str(
            info.get("ticket_id", f"{setting}#origin{origin_index}")
        )
        episode_step = int(info.get("episode_step", 0))
        x = _finite_or_nan(info.get("ego_x"))
        y = _finite_or_nan(info.get("ego_y"))
        current_xy = (x, y)
        origin = self.spawn_xy_by_ticket.get(ticket_id, current_xy)
        if episode_step <= 1 or self._previous_xy is None:
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
        displacement = math.hypot(x - origin[0], y - origin[1])
        row = {
            "valid_step": int(self.model.num_timesteps),
            "policy_update": self.policy_update,
            "ticket_id": ticket_id,
            "setting_id": setting,
            "origin_index": origin_index,
            "episode_step": episode_step,
            "ego_x": x,
            "ego_y": y,
            "displacement_from_spawn_m": displacement,
            "planar_step_distance_m": step_distance,
            "path_distance_m": self._path_distance,
            "planar_speed_mps": step_distance / self.control_dt_s,
            "planar_average_speed_mps": (
                self._path_distance
                / max(1, episode_step)
                / self.control_dt_s
            ),
            "ego_speed_mps": _finite_or_nan(info.get("ego_speed_mps")),
            "reward_average_3d_speed_mps": _finite_or_nan(
                info.get("reward_average_speed_mps")
            ),
            "action_target_speed_norm": (
                float(action[0]) if action.size > 0 else math.nan
            ),
            "action_target_yaw_rate_norm": (
                float(action[1]) if action.size > 1 else math.nan
            ),
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
            done
            or episode_step == 1
            or self.model.num_timesteps % self.live_every == 0
        ):
            print(
                f"{format_live_update(info, reward, done)} "
                f"ticket={ticket_id} displacement={displacement:.3f}m "
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
            self.outcomes[(ticket_id, reason)] += 1
            self._previous_xy = None
        return True

    def record_infrastructure_fault(self, info: dict) -> None:
        reason = str(info.get("reason", "unknown"))
        ticket_id = str(info.get("ticket_id", "unknown"))
        self.infrastructure_faults[reason] += 1
        self._previous_xy = None
        print(
            f"INFRA_RETRY ticket={ticket_id} reason={reason}",
            flush=True,
        )

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def _finite_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--total-valid-steps",
        type=int,
        default=DEFAULT_TOTAL_VALID_STEPS_V2,
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS_V2
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=V2_DEFAULT_PPO_EPOCHS
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--live-log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--carla-port", type=int, default=2040)
    parser.add_argument("--tm-port", type=int, default=8040)
    parser.add_argument("--sumo-port", type=int, default=8853)
    parser.add_argument(
        "--show-traffic-cones", action="store_true", default=False
    )
    parser.add_argument("--cones-only", action="store_true")
    parser.add_argument("--model", help="Optional compatible V2 PPO checkpoint")
    parser.add_argument("--run-dir", default="runs/s3_all_debug_v2")
    parser.add_argument("--save-model", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_valid_steps < 1 or args.rollout_steps < 1:
        raise ValueError("step counts must be positive")
    if args.total_valid_steps % args.rollout_steps != 0:
        raise ValueError(
            "--total-valid-steps must be divisible by --rollout-steps"
        )

    preflight = validate_scenario("s3")
    if not preflight.ok:
        for error in preflight.errors:
            print(f"ERROR {error}")
        return 2
    runnable = tuple(list_setting_ids("s3", runnable_only=True))
    if runnable != SETTINGS_V2:
        print(
            f"ERROR S3 runnable order mismatch: {runnable} != {SETTINGS_V2}",
            flush=True,
        )
        return 2

    configs = {setting: load_scenario(setting) for setting in SETTINGS_V2}
    origins = {
        setting: tuple(configs[setting].origin.spawn_points)
        for setting in SETTINGS_V2
    }
    origin_counts = {setting: len(points) for setting, points in origins.items()}
    bad_counts = {
        setting: count
        for setting, count in origin_counts.items()
        if count != ORIGINS_PER_SETTING_V2
    }
    if bad_counts:
        print(f"ERROR expected three origins per S3 setting: {bad_counts}")
        return 2

    selector = OrderedOriginSelector(
        SETTINGS_V2, origin_counts, repeats=1
    )
    binder = OriginBinder(selector, origins)
    geometries_by_ticket = {
        f"{setting}#origin{origin_index}": build_visualization_geometry_v2(
            load_scenario(setting), origin_index=origin_index
        )
        for setting in SETTINGS_V2
        for origin_index in range(origin_counts[setting])
    }
    spawn_xy_by_ticket = {
        f"{setting}#origin{origin_index}": (
            float(point.x),
            float(point.y),
        )
        for setting, points in origins.items()
        for origin_index, point in enumerate(points)
    }

    plan = plan_single(
        [configs[setting].episode.max_steps for setting in SETTINGS_V2],
        repeats=ORIGINS_PER_SETTING_V2,
        n_steps_override=args.rollout_steps,
    )
    coverage_worst_case_steps = sum(
        configs[setting].episode.max_steps for setting in SETTINGS_V2
    ) * ORIGINS_PER_SETTING_V2
    initial = configs[SETTINGS_V2[0]]
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
    callback = S3AllExplorationTraceCallbackV2(
        trace_path=trace_path,
        spawn_xy_by_ticket=spawn_xy_by_ticket,
        live_every=args.live_log_every,
        control_dt_s=float(initial.episode.sim_dt),
    )
    visualizer = None
    if args.show_traffic_cones:
        from tools.show_traffic_cones import WorkZonePropVisualizer

        visualizer = WorkZonePropVisualizer(
            SETTINGS_V2[0],
            host=initial.carla.host,
            carla_port=args.carla_port,
            cones_only=args.cones_only,
            draw_finish_line=False,
            load_town=False,
            move_spectator=False,
        )

    def episode_setup(setting_id: str) -> None:
        binder.bind(setting_id)
        ticket = selector.current_ticket
        print(
            f"DEBUG_TICKET setting={ticket.setting_id} "
            f"origin={ticket.origin_index} ticket={ticket.ticket_id}",
            flush=True,
        )
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
        return S3AllDebugWrapperV2(
            base_env,
            selector=selector,
            geometries_by_ticket=geometries_by_ticket,
        )

    print(
        "debug_scope scenario=s3 settings=9 origins=3 tickets=27 "
        "order=WZ-layout-origin spectator=manual realtime_throttle=off",
        flush=True,
    )
    print("ticket_order", TICKET_IDS_V2, flush=True)
    print("rollout_plan", describe_plan(plan), flush=True)
    print(
        f"training valid_steps={args.total_valid_steps} "
        f"short_rollouts={args.total_valid_steps // args.rollout_steps} "
        f"steps_per_rollout={args.rollout_steps}",
        flush=True,
    )
    if args.total_valid_steps < coverage_worst_case_steps:
        print(
            f"NOTE exact {args.total_valid_steps}-step budget preserves the "
            f"real episode limits; worst-case full 27-ticket coverage needs "
            f"{coverage_worst_case_steps} steps. Missing tickets will be "
            "reported at the end.",
            flush=True,
        )
    print(f"trace_csv={trace_path}", flush=True)

    vec_env = None
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
            "spectator=manual realtime_throttle=off",
            flush=True,
        )
        training_end_step, callback = model._setup_learn(
            args.total_valid_steps,
            callback,
            reset_num_timesteps=(args.model is None),
            tb_log_name="s3_all_debug_v2",
        )
        callback.on_training_start(locals(), globals())
        training_started = True
        rollout_count = args.total_valid_steps // args.rollout_steps
        for update_index in range(1, rollout_count + 1):
            callback.policy_update = update_index - 1
            print(
                f"ROLLOUT {update_index:02d}/{rollout_count:02d} START "
                f"valid_target={args.rollout_steps} "
                f"policy_version={update_index - 1}",
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

    completed = selector.completed_ticket_counts
    missing = [ticket_id for ticket_id, count in completed.items() if count < 1]
    print("coverage_complete", selector.complete, flush=True)
    print("completed_ticket_counts", completed, flush=True)
    print("missing_tickets", missing, flush=True)
    print("outcomes", dict(callback.outcomes), flush=True)
    print(
        "infrastructure_faults",
        dict(callback.infrastructure_faults),
        flush=True,
    )
    print(f"trace_csv={trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
