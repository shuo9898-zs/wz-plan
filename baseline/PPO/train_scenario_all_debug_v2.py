"""Shared visible PPO debug runner for the non-S3 scenarios.

Each thin scenario entry point pins one scenario and visits its runnable
settings in catalog order, with authored origins ordered 0, 1, 2.  The
native CARLA spectator is never moved.  Work-zone forbidden areas are drawn
in red using the same canonical geometry adapter as V2 termination.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from baseline.PPO.rollout_plan import describe_plan, plan_single
from baseline.PPO.runtime_v2 import (
    V2_DEFAULT_PPO_EPOCHS,
    build_or_load_ppo,
    configure_initial_config,
    save_ppo,
)
from baseline.PPO.train_s3_all_debug_v2 import (
    S3AllExplorationTraceCallbackV2,
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
    FINISH_YELLOW_V2,
    FORBIDDEN_RED_V2,
    ORIGIN_CYAN_V2,
    VisualizationGeometryV2,
    build_visualization_geometry_v2,
    hatch_segments_v2,
)


DEFAULT_TOTAL_VALID_STEPS_V2 = 10_000
DEFAULT_ROLLOUT_STEPS_V2 = 1_000
ORIGINS_PER_SETTING_V2 = 3
REWARD_OD_MAGENTA_V2 = (255, 40, 255)


@dataclass(frozen=True)
class ScenarioDebugSpecV2:
    scenario_id: str
    town: str
    carla_port: int
    tm_port: int
    sumo_port: int


SCENARIO_DEBUG_SPECS_V2 = {
    "s1": ScenarioDebugSpecV2("s1", "Town02", 2000, 8000, 8813),
    "s2": ScenarioDebugSpecV2("s2", "Town05", 2020, 8020, 8833),
    "s4": ScenarioDebugSpecV2("s4", "Town10HD", 2040, 8040, 8853),
    "s5": ScenarioDebugSpecV2("s5", "Town05", 2020, 8020, 8833),
    "s6": ScenarioDebugSpecV2("s6", "Town02", 2000, 8000, 8813),
}


def scenario_debug_spec_v2(scenario_id: str) -> ScenarioDebugSpecV2:
    normalized = str(scenario_id).strip().lower()
    try:
        return SCENARIO_DEBUG_SPECS_V2[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported non-S3 debug scenario: {scenario_id!r}"
        ) from exc


def scenario_settings_v2(scenario_id: str) -> tuple[str, ...]:
    spec = scenario_debug_spec_v2(scenario_id)
    settings = tuple(list_setting_ids(spec.scenario_id, runnable_only=True))
    if not settings:
        raise RuntimeError(f"{spec.scenario_id} has no runnable settings")
    return settings


def scenario_ticket_ids_v2(scenario_id: str) -> tuple[str, ...]:
    return tuple(
        f"{setting}#origin{origin_index}"
        for setting in scenario_settings_v2(scenario_id)
        for origin_index in range(ORIGINS_PER_SETTING_V2)
    )


class ScenarioAllDebugWrapperV2(gym.Wrapper):
    """Draw the active forbidden area and origin without moving the camera."""

    def __init__(
        self,
        env: gym.Env,
        *,
        selector: OrderedOriginSelector,
        spawn_xy_by_ticket: Mapping[str, tuple[float, float]],
        draw_reward_od_axis: bool = False,
    ) -> None:
        super().__init__(env)
        self._selector = selector
        self._spawn_xy_by_ticket = dict(spawn_xy_by_ticket)
        self._draw_reward_od_axis = bool(draw_reward_od_axis)
        control_dt_s = float(self.env.engine.cfg.episode.sim_dt)
        self._overlay_lifetime_s = OVERLAY_LIFETIME_S_V2
        self._overlay_refresh_interval_steps = max(
            1,
            int(round(OVERLAY_REFRESH_PERIOD_S_V2 / control_dt_s)),
        )
        self._overlay_refresh_countdown = 0
        self._overlay_key: tuple[Any, ...] | None = None
        self._active_ticket: EpisodeTicket | None = None
        self._geometry_cache: dict[str, VisualizationGeometryV2] = {}

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

    def _geometry_for_ticket(
        self,
        ticket: EpisodeTicket,
        carla_map: object,
    ) -> VisualizationGeometryV2:
        cached = self._geometry_cache.get(ticket.ticket_id)
        if cached is not None:
            return cached
        geometry = build_visualization_geometry_v2(
            load_scenario(ticket.setting_id),
            carla_map=carla_map,
        )
        geometry = replace(
            geometry,
            origin=self._spawn_xy_by_ticket[ticket.ticket_id],
        )
        self._geometry_cache[ticket.ticket_id] = geometry
        return geometry

    def _draw_overlay_if_needed(self) -> None:
        try:
            import carla

            ticket = self._active_ticket
            world = self.env.engine.world
            if ticket is None or world is None:
                return
            carla_map = world.get_map()
            geometry = self._geometry_for_ticket(ticket, carla_map)
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

            for layer in geometry.layers:
                # Every scenario handled by this runner uses forbidden-area
                # semantics.  Force one conspicuous colour here instead of
                # relying on a layer-specific colour so the live demo cannot
                # be mistaken for a drivable-area overlay.
                color = carla.Color(*FORBIDDEN_RED_V2)
                for start, end in zip(
                    layer.vertices, layer.vertices[1:] + layer.vertices[:1]
                ):
                    world.debug.draw_line(
                        location(start),
                        location(end),
                        thickness=0.20,
                        color=color,
                        life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
                    # S6 forbidden strips are only about 1.2--1.7 m wide and
                    # their ground hatch is easily hidden by the cone props.
                    # Repeat the exact same boundary above the road as a
                    # visible laser fence; this is visualization only.
                    if geometry.scenario_id == "s6":
                        world.debug.draw_line(
                            location(start, 0.85),
                            location(end, 0.85),
                            thickness=0.10,
                            color=color,
                            life_time=self._overlay_lifetime_s,
                            persistent_lines=False,
                        )
                        world.debug.draw_line(
                            location(start, 0.20),
                            location(start, 0.85),
                            thickness=0.08,
                            color=color,
                            life_time=self._overlay_lifetime_s,
                            persistent_lines=False,
                        )
                for start, end in hatch_segments_v2(
                    layer.vertices,
                    0.75,
                    True,
                    holes=layer.holes,
                ):
                    world.debug.draw_line(
                        location(start, 0.19),
                        location(end, 0.19),
                        thickness=0.055,
                        color=color,
                        life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
                label_x = sum(point[0] for point in layer.vertices) / len(
                    layer.vertices
                )
                label_y = sum(point[1] for point in layer.vertices) / len(
                    layer.vertices
                )
                world.debug.draw_string(
                    location((label_x, label_y), 0.52),
                    "RED = FORBIDDEN / VIOLATION",
                    draw_shadow=True,
                    color=color,
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
            cyan = carla.Color(*ORIGIN_CYAN_V2)
            label = f"{ticket.setting_id.upper()} origin{ticket.origin_index}"
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
            if self._draw_reward_od_axis:
                od_start = geometry.origin
                od_end = (
                    0.5 * (geometry.finish_line.first[0] + geometry.finish_line.second[0]),
                    0.5 * (geometry.finish_line.first[1] + geometry.finish_line.second[1]),
                )
                reward_color = carla.Color(*REWARD_OD_MAGENTA_V2)
                world.debug.draw_arrow(
                    location(od_start, 0.36),
                    location(od_end, 0.36),
                    thickness=0.14,
                    arrow_size=0.28,
                    color=reward_color,
                    life_time=self._overlay_lifetime_s,
                    persistent_lines=False,
                )
                axis_label_point = (
                    od_start[0] + 0.45 * (od_end[0] - od_start[0]),
                    od_start[1] + 0.45 * (od_end[1] - od_start[1]),
                )
                world.debug.draw_string(
                    location(axis_label_point, 0.62),
                    "FIXED OD REWARD AXIS",
                    draw_shadow=True,
                    color=reward_color,
                    life_time=self._overlay_lifetime_s,
                    persistent_lines=False,
                )
            self._overlay_key = key
            self._overlay_refresh_countdown = (
                self._overlay_refresh_interval_steps - 1
            )
        except Exception as error:
            print(
                f"WARN debug forbidden-area overlay could not be drawn: {error}",
                flush=True,
            )


class ScenarioAllExplorationTraceCallbackV2(
    S3AllExplorationTraceCallbackV2
):
    """Scenario-neutral spelling of the ticket-aware exploration trace."""


def build_parser_for_scenario_v2(
    scenario_id: str,
) -> argparse.ArgumentParser:
    spec = scenario_debug_spec_v2(scenario_id)
    parser = argparse.ArgumentParser(
        description=(
            f"Visible PPO debug across every {spec.scenario_id.upper()} "
            "work-zone, layout, and authored origin."
        )
    )
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
    parser.add_argument("--carla-port", type=int, default=spec.carla_port)
    parser.add_argument("--tm-port", type=int, default=spec.tm_port)
    parser.add_argument("--sumo-port", type=int, default=spec.sumo_port)
    parser.add_argument(
        "--show-traffic-cones", action="store_true", default=False
    )
    parser.add_argument("--cones-only", action="store_true")
    parser.add_argument("--model", help="Optional compatible V2 PPO checkpoint")
    parser.add_argument(
        "--run-dir", default=f"runs/{spec.scenario_id}_all_debug_v2"
    )
    parser.add_argument("--save-model", action="store_true")
    return parser


def main_for_scenario_v2(
    scenario_id: str,
    argv: list[str] | None = None,
) -> int:
    spec = scenario_debug_spec_v2(scenario_id)
    args = build_parser_for_scenario_v2(spec.scenario_id).parse_args(argv)
    if args.total_valid_steps < 1 or args.rollout_steps < 1:
        raise ValueError("step counts must be positive")
    if args.total_valid_steps % args.rollout_steps != 0:
        raise ValueError(
            "--total-valid-steps must be divisible by --rollout-steps"
        )

    preflight = validate_scenario(spec.scenario_id)
    if not preflight.ok:
        for error in preflight.errors:
            print(f"ERROR {error}")
        return 2
    settings = scenario_settings_v2(spec.scenario_id)
    configs = {setting: load_scenario(setting) for setting in settings}
    wrong_towns = {
        setting: config.carla.town
        for setting, config in configs.items()
        if config.carla.town != spec.town
    }
    if wrong_towns:
        print(f"ERROR town mismatch: {wrong_towns}", flush=True)
        return 2
    origins = {
        setting: tuple(configs[setting].origin.spawn_points)
        for setting in settings
    }
    origin_counts = {setting: len(points) for setting, points in origins.items()}
    bad_counts = {
        setting: count
        for setting, count in origin_counts.items()
        if count != ORIGINS_PER_SETTING_V2
    }
    if bad_counts:
        print(
            f"ERROR expected three origins per {spec.scenario_id.upper()} "
            f"setting: {bad_counts}",
            flush=True,
        )
        return 2

    selector = OrderedOriginSelector(settings, origin_counts, repeats=1)
    binder = OriginBinder(selector, origins)
    spawn_xy_by_ticket = {
        f"{setting}#origin{origin_index}": (
            float(point.x),
            float(point.y),
        )
        for setting, points in origins.items()
        for origin_index, point in enumerate(points)
    }
    ticket_ids = scenario_ticket_ids_v2(spec.scenario_id)

    plan = plan_single(
        [configs[setting].episode.max_steps for setting in settings],
        repeats=ORIGINS_PER_SETTING_V2,
        n_steps_override=args.rollout_steps,
    )
    coverage_worst_case_steps = sum(
        configs[setting].episode.max_steps for setting in settings
    ) * ORIGINS_PER_SETTING_V2
    initial = configs[settings[0]]
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
    callback = ScenarioAllExplorationTraceCallbackV2(
        trace_path=trace_path,
        spawn_xy_by_ticket=spawn_xy_by_ticket,
        live_every=args.live_log_every,
        control_dt_s=float(initial.episode.sim_dt),
    )
    visualizer = None
    if args.show_traffic_cones:
        from tools.show_traffic_cones import WorkZonePropVisualizer

        visualizer = WorkZonePropVisualizer(
            settings[0],
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
            scenario=settings[0],
            config=initial,
            worker_id=0,
            no_rendering_mode=False,
            scenario_selector=selector,
            episode_setup_callback=episode_setup,
        )
        binder.attach(base_env)
        return ScenarioAllDebugWrapperV2(
            base_env,
            selector=selector,
            spawn_xy_by_ticket=spawn_xy_by_ticket,
            draw_reward_od_axis=True,
        )

    print(
        f"debug_scope scenario={spec.scenario_id} settings={len(settings)} "
        f"origins=3 tickets={len(ticket_ids)} order=WZ-layout-origin "
        "zone=forbidden-red "
        f"reward_axis={'magenta-forward' if spec.scenario_id == 's2' else 'off'} "
        "spectator=manual realtime_throttle=off",
        flush=True,
    )
    print("ticket_order", ticket_ids, flush=True)
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
            f"real episode limits; worst-case full {len(ticket_ids)}-ticket "
            f"coverage needs {coverage_worst_case_steps} steps. Missing "
            "tickets will be reported at the end.",
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
            tb_log_name=f"{spec.scenario_id}_all_debug_v2",
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


__all__ = [
    "DEFAULT_ROLLOUT_STEPS_V2",
    "DEFAULT_TOTAL_VALID_STEPS_V2",
    "ORIGINS_PER_SETTING_V2",
    "REWARD_OD_MAGENTA_V2",
    "SCENARIO_DEBUG_SPECS_V2",
    "ScenarioAllDebugWrapperV2",
    "ScenarioAllExplorationTraceCallbackV2",
    "ScenarioDebugSpecV2",
    "build_parser_for_scenario_v2",
    "main_for_scenario_v2",
    "scenario_debug_spec_v2",
    "scenario_settings_v2",
    "scenario_ticket_ids_v2",
]
