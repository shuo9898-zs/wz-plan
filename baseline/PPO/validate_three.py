"""Three-worker/three-server PPO mode grouped by CARLA town."""
from __future__ import annotations

import argparse

from stable_baselines3.common.vec_env import SubprocVecEnv

from baseline.PPO.callbacks import GroupCoverageCallback
from baseline.PPO.rollout_plan import describe_plan, plan_parallel
from baseline.PPO.runtime_v2 import (
    V2_DEFAULT_PPO_EPOCHS,
    build_or_load_ppo,
    collect_settings,
    configure_initial_config,
    save_ppo,
)
from config.scenario_config import load_scenario
from config.scenario_selector import CoverageSelector
from tools.preflight import validate_scenario


INFRASTRUCTURE_REASONS = CoverageSelector.INFRASTRUCTURE_REASONS


def _scenario_settings(scenarios: list[str]) -> list[str]:
    for scenario in scenarios:
        result = validate_scenario(scenario)
        if not result.ok:
            detail = "; ".join(result.errors)
            raise ValueError(f"{scenario} failed preflight: {detail}")
    return collect_settings(scenarios)


def _make_worker(
    settings: list[str],
    repeats: int,
    worker_id: int,
    carla_port: int,
    tm_port: int,
    sumo_port: int,
    no_rendering: bool,
):
    def init():
        from env.gym_wrapper_v2 import CarlaSumoGymEnv

        selector = CoverageSelector(settings, repeats)
        initial = load_scenario(settings[0])
        configure_initial_config(
            initial,
            carla_port=carla_port,
            tm_port=tm_port,
            sumo_port=sumo_port,
            no_rendering=no_rendering,
        )
        return CarlaSumoGymEnv(
            scenario=settings[0],
            config=initial,
            worker_id=worker_id,
            no_rendering_mode=no_rendering,
            scenario_selector=selector,
        )

    return init


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town02", nargs="+", required=True, help="Town02 scenarios, e.g. s1 s6")
    parser.add_argument("--town05", nargs="+", required=True, help="Town05 scenarios, e.g. s2 s5")
    parser.add_argument("--town10hd", nargs="+", required=True, help="Town10HD scenarios, e.g. s3 s4")
    parser.add_argument("--episodes-per-setting", type=int, default=1)
    parser.add_argument("--max-timesteps", type=int, default=None,
                        help="Training safety cap; default is one full auto-sized rollout")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Steps per worker per rollout; default fits the busiest town's complete N-ticket sweep")
    parser.add_argument("--ppo-epochs", type=int, default=V2_DEFAULT_PPO_EPOCHS,
                        help="Optimization passes over each completed rollout buffer")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--carla-ports", type=int, nargs=3, default=[2000, 2020, 2040])
    parser.add_argument("--tm-ports", type=int, nargs=3, default=[8000, 8020, 8040])
    parser.add_argument("--sumo-ports", type=int, nargs=3, default=[8813, 8833, 8853])
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--model", help="Optional PPO .zip to continue")
    parser.add_argument("--output", default="runs/three_server_validation")
    args = parser.parse_args(argv)

    if args.episodes_per_setting < 1:
        parser.error("--episodes-per-setting must be positive")
    groups = [args.town02, args.town05, args.town10hd]
    expected_towns = ["Town02", "Town05", "Town10HD"]
    try:
        group_settings = [_scenario_settings(group) for group in groups]
        for expected, settings in zip(expected_towns, group_settings):
            for setting in settings:
                if load_scenario(setting).carla.town != expected:
                    raise ValueError(f"{setting} is not on {expected}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR {exc}")
        return 2

    all_settings = [setting for group in group_settings for setting in group]
    group_step_limits = [
        [load_scenario(setting).episode.max_steps for setting in settings]
        for settings in group_settings
    ]
    plan = plan_parallel(group_step_limits, args.episodes_per_setting, args.n_steps)
    total_timesteps = args.max_timesteps or plan.buffer_size_steps
    print("rollout_plan", describe_plan(plan))
    if args.n_steps is not None and plan.n_steps_per_env < max(plan.group_step_budgets):
        print("WARN manual --n-steps is smaller than the busiest town's complete setting sweep")
    env_fns = [
        _make_worker(
            settings,
            args.episodes_per_setting,
            worker_id,
            args.carla_ports[worker_id],
            args.tm_ports[worker_id],
            args.sumo_ports[worker_id],
            args.no_rendering,
        )
        for worker_id, settings in enumerate(group_settings)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    callback = GroupCoverageCallback(all_settings, args.episodes_per_setting)
    model = build_or_load_ppo(
        vec_env,
        plan,
        model_path=args.model,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        save_ppo(model, args.output)
    finally:
        vec_env.close()

    print("coverage", {setting: callback.completed[setting] for setting in all_settings})
    print("outcomes", dict(callback.outcomes))
    complete = all(callback.completed[s] >= args.episodes_per_setting for s in all_settings)
    if not complete:
        print("ERROR max timesteps reached before coverage completed")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
