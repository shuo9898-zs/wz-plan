"""Single-CARLA-server PPO validation over Scenario -> WZ -> a/b/c."""
from __future__ import annotations

import argparse

from stable_baselines3.common.vec_env import DummyVecEnv

from baseline.PPO.callbacks import CoverageStopCallback
from baseline.PPO.rollout_plan import describe_plan, plan_single
from baseline.PPO.runtime_v2 import (
    V2_DEFAULT_PPO_EPOCHS,
    build_or_load_ppo,
    configure_initial_config,
    save_ppo,
)
from config.scenario_catalog import CORE_SCENARIOS, list_setting_ids
from config.scenario_config import load_scenario
from config.scenario_selector import CoverageSelector, RandomWorkZoneSelector
from tools.preflight import validate_scenario


class _FullStepBudgetCallback(CoverageStopCallback):
    """Track the same diagnostics without stopping when coverage completes."""

    def _on_rollout_end(self) -> None:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=list(CORE_SCENARIOS))
    parser.add_argument("--episodes-per-setting", type=int, default=1)
    parser.add_argument(
        "--selection", choices=("ordered", "random"), default="ordered",
        help="Episode order; random keeps this scenario fixed and shuffles WZ then layout",
    )
    parser.add_argument("--max-timesteps", type=int, default=None,
                        help="Training safety cap; default is one full auto-sized rollout")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Steps per environment per rollout; default fits N max-length episodes for every setting")
    parser.add_argument(
        "--continue-after-coverage",
        action="store_true",
        help="Keep collecting and updating until --max-timesteps after coverage completes",
    )
    parser.add_argument("--ppo-epochs", type=int, default=V2_DEFAULT_PPO_EPOCHS,
                        help="Optimization passes over each completed rollout buffer")
    parser.add_argument("--live-log-every", type=int, default=0,
                        help="Print setting/ego/SUMO status every N steps; 0 disables")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--sumo-port", type=int, default=8813)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--show-traffic-cones", action="store_true",
                        help="Demo/debug: switch visible cone/sign props with every setting")
    parser.add_argument("--cones-only", action="store_true",
                        help="With --show-traffic-cones, omit warning signs")
    parser.add_argument("--model", help="Optional PPO .zip to continue/evaluate while learning")
    parser.add_argument("--output", default="runs/single_server_validation")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the trained PPO model")
    args = parser.parse_args(argv)

    preflight = validate_scenario(args.scenario)
    if not preflight.ok:
        for error in preflight.errors:
            print(f"ERROR {error}")
        return 2
    settings = list_setting_ids(args.scenario, runnable_only=True)
    configs = [load_scenario(setting) for setting in settings]
    step_limits = [config.episode.max_steps for config in configs]
    plan = plan_single(step_limits, args.episodes_per_setting, args.n_steps)
    total_timesteps = args.max_timesteps or plan.buffer_size_steps
    print("rollout_plan", describe_plan(plan))
    if args.n_steps is not None and plan.n_steps_per_env < plan.group_step_budgets[0]:
        print("WARN manual --n-steps is smaller than one worst-case complete setting sweep")
    if args.selection == "random":
        selector = RandomWorkZoneSelector(
            settings, args.episodes_per_setting, seed=args.seed
        )
    else:
        selector = CoverageSelector(settings, args.episodes_per_setting)
    print("selection", args.selection, "scenario", args.scenario, "settings", settings)
    initial = configs[0]
    configure_initial_config(
        initial,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        sumo_port=args.sumo_port,
        no_rendering=args.no_rendering,
    )

    visualizer = None
    if args.show_traffic_cones:
        from tools.show_traffic_cones import WorkZonePropVisualizer

        visualizer = WorkZonePropVisualizer(
            settings[0],
            host=initial.carla.host,
            carla_port=args.carla_port,
            cones_only=args.cones_only,
            load_town=False,
            # Keep the spectator under manual control throughout PPO.  The
            # visualizer may replace WZ/layout props on reset, but it must not
            # overwrite the camera transform the user selected in CARLA.
            move_spectator=False,
        )

    def make_env():
        from env.gym_wrapper_v2 import CarlaSumoGymEnv

        return CarlaSumoGymEnv(
            scenario=settings[0],
            config=initial,
            worker_id=0,
            no_rendering_mode=args.no_rendering,
            scenario_selector=selector,
            episode_setup_callback=(visualizer.switch if visualizer is not None else None),
        )

    vec_env = DummyVecEnv([make_env])
    callback_type = (
        _FullStepBudgetCallback
        if args.continue_after_coverage
        else CoverageStopCallback
    )
    callback = callback_type(selector, args.live_log_every)
    model = build_or_load_ppo(
        vec_env,
        plan,
        model_path=args.model,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
        if not args.no_save:
            save_ppo(model, args.output)
    finally:
        vec_env.close()
        if visualizer is not None:
            visualizer.close()

    print("coverage", selector.completed_counts)
    print("outcomes", dict(callback.outcomes))
    if not selector.complete:
        print("ERROR max timesteps reached before coverage completed")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
