"""Single-server MTR-PPO debug on experiment-facing Town02/S5 only.

The experiment-facing S5 label is backed by the stable runtime ``s1`` files.
This runner deliberately keeps PPO V2, reward, termination and observation
unchanged; only the feature extractor is replaced by InputMatchedMTREncoder.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from baseline.MTR_PPO.runtime import build_or_load_ppo
from baseline.PPO.rollout_plan import describe_plan, plan_single
from baseline.PPO.runtime_v2 import save_ppo
from baseline.PPO.train_s3_all_debug_v2 import S3AllExplorationTraceCallbackV2
from baseline.PPO.train_s3_wz1_origin2_debug_v2 import _close_debug_resources_v2
from baseline.PPO.valid_rollout import collect_valid_rollouts
from baseline.PPO.validate_allSwithoneExe import OrderedOriginSelector, OriginBinder
from config.scenario_catalog import list_setting_ids
from config.scenario_config import load_scenario
from tools.preflight import validate_scenario


RUNTIME_SCENARIO_ID = "s1"
DISPLAY_SCENARIO_ID = "s5"
TOWN = "Town02"
DEFAULT_UPDATES = 10
DEFAULT_STEPS_PER_UPDATE = 16_200
DEFAULT_RUN_DIR = Path("runs_mtr_ppo_debug_town02_s5")


class _TicketTelemetryWrapper(gym.Wrapper):
    """Attach the active setting/origin ticket to every debug trace row."""

    def __init__(self, env: gym.Env, selector: OrderedOriginSelector) -> None:
        super().__init__(env)
        self._selector = selector

    def _tag(self, info: dict) -> dict:
        tagged = dict(info)
        if "origin_index" in tagged and "ticket_id" in tagged:
            return tagged
        ticket = self._selector.current_ticket
        tagged.setdefault("origin_index", ticket.origin_index)
        tagged.setdefault("ticket_id", ticket.ticket_id)
        return tagged

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, self._tag(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, self._tag(info)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument(
        "--steps-per-update", type=int, default=DEFAULT_STEPS_PER_UPDATE
    )
    parser.add_argument("--ppo-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--carla-port", type=int, default=2060)
    parser.add_argument("--tm-port", type=int, default=8060)
    parser.add_argument("--sumo-port", type=int, default=8873)
    parser.add_argument("--live-log-every", type=int, default=1_000)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.updates < 1 or args.steps_per_update < 1:
        raise ValueError("updates and steps-per-update must be positive")
    if args.ppo_epochs != 20:
        raise ValueError("This controlled debug keeps the formal PPO value: 20 epochs")
    if args.batch_size != 512:
        raise ValueError("This controlled debug keeps the formal PPO batch size: 512")

    preflight = validate_scenario(RUNTIME_SCENARIO_ID)
    if not preflight.ok:
        for error in preflight.errors:
            print(f"ERROR {error}", flush=True)
        return 2

    settings = tuple(list_setting_ids(RUNTIME_SCENARIO_ID, runnable_only=True))
    configs = {setting: load_scenario(setting) for setting in settings}
    if not settings or any(config.carla.town != TOWN for config in configs.values()):
        raise RuntimeError("Town02/S5 runtime binding is invalid")
    origins = {
        setting: tuple(config.origin.spawn_points)
        for setting, config in configs.items()
    }
    if any(len(points) != 3 for points in origins.values()):
        raise RuntimeError("Every Town02/S5 setting must expose three origins")

    plan = plan_single(
        [config.episode.max_steps for config in configs.values()],
        repeats=3,
        n_steps_override=args.steps_per_update,
    )
    print(
        "MTR_PPO_DEBUG_PLAN",
        f"display={DISPLAY_SCENARIO_ID}",
        f"runtime={RUNTIME_SCENARIO_ID}",
        f"town={TOWN}",
        f"updates={args.updates}",
        f"steps_per_update={args.steps_per_update}",
        f"total_steps={args.updates * args.steps_per_update}",
        f"epochs={args.ppo_epochs}",
        f"batch={args.batch_size}",
        "target_kl=None",
        "no_rendering=True",
        flush=True,
    )
    print("rollout_plan", describe_plan(plan), flush=True)
    if args.check_only:
        return 0

    initial = configs[settings[0]]
    from baseline.PPO.runtime_v2 import configure_initial_config

    configure_initial_config(
        initial,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        sumo_port=args.sumo_port,
        no_rendering=True,
    )

    selector = OrderedOriginSelector(
        settings,
        {setting: len(points) for setting, points in origins.items()},
        repeats=1,
    )
    binder = OriginBinder(selector, origins)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / (
        "exploration_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    )
    spawn_xy_by_ticket = {
        f"{setting}#origin{origin_index}": (float(point.x), float(point.y))
        for setting, points in origins.items()
        for origin_index, point in enumerate(points)
    }
    callback = S3AllExplorationTraceCallbackV2(
        trace_path=trace_path,
        spawn_xy_by_ticket=spawn_xy_by_ticket,
        live_every=args.live_log_every,
        control_dt_s=float(initial.episode.sim_dt),
    )

    def episode_setup(setting_id: str) -> None:
        binder.bind(setting_id)

    def make_env():
        from env.gym_wrapper_v2 import CarlaSumoGymEnv

        env = CarlaSumoGymEnv(
            scenario=settings[0],
            config=initial,
            worker_id=60,
            no_rendering_mode=True,
            scenario_selector=selector,
            episode_setup_callback=episode_setup,
        )
        binder.attach(env)
        return _TicketTelemetryWrapper(env, selector)

    vec_env = None
    training_started = False
    try:
        vec_env = DummyVecEnv([make_env])
        model = build_or_load_ppo(
            vec_env,
            plan,
            model_path=None,
            ppo_epochs=args.ppo_epochs,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            target_kl=None,
        )
        training_end_step, callback = model._setup_learn(
            args.updates * args.steps_per_update,
            callback,
            reset_num_timesteps=True,
            tb_log_name="mtr_ppo_town02_s5_debug",
        )
        callback.on_training_start(locals(), globals())
        training_started = True

        for update in range(1, args.updates + 1):
            callback.policy_update = update - 1
            print(
                f"MTR_ROLLOUT_START update={update}/{args.updates} "
                f"target={args.steps_per_update}",
                flush=True,
            )
            complete = collect_valid_rollouts(
                model,
                vec_env,
                callback,
                model.rollout_buffer,
                n_rollout_steps=args.steps_per_update,
            )
            if not complete or not model.rollout_buffer.full:
                raise RuntimeError(f"MTR rollout {update} did not fill")
            model._update_current_progress_remaining(
                model.num_timesteps, training_end_step
            )
            epochs_before = int(model._n_updates)
            model.train()
            callback.policy_update = update
            model.logger.dump(step=model.num_timesteps)
            model_path = run_dir / "models" / f"policy_update_{update:03d}"
            save_ppo(model, model_path)
            print(
                f"MTR_PPO_UPDATE_END update={update}/{args.updates} "
                f"valid_steps={model.num_timesteps} "
                f"optimizer_epochs={int(model._n_updates) - epochs_before} "
                f"model={model_path}.zip",
                flush=True,
            )
    finally:
        _close_debug_resources_v2(
            callback=callback,
            training_started=training_started,
            vec_env=vec_env,
            visualizer=None,
        )

    print("MTR_PPO_TOWN02_S5_DEBUG_COMPLETE", f"trace={trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
