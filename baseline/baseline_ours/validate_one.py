"""Standalone PPO check for one fixed Scenario/WZ.

This file intentionally does not change or replace ``validate_single.py``.
It is the visual/demo entry point for selecting one canonical setting such as
``s1/wz1`` while either fixing or episode-randomizing its A/B/C layout.
"""
from __future__ import annotations

import argparse

from stable_baselines3.common.vec_env import DummyVecEnv

from baseline.baseline_ours.callbacks import CoverageStopCallback
from baseline.baseline_ours.rollout_plan import describe_plan, plan_single
from baseline.baseline_ours.runtime_v2 import (
    V2_DEFAULT_PPO_EPOCHS,
    build_or_load_ppo,
    configure_initial_config,
    save_ppo,
)
from config.scenario_catalog import list_setting_ids, scenario_ids
from config.scenario_config import load_scenario
from config.scenario_selector import CoverageSelector, RandomLayoutSelector
from tools.preflight import validate_scenario


class LiveCoverageCallback(CoverageStopCallback):
    """Print one-setting CARLA/SUMO/PPO progress to the demo terminal."""

    def __init__(self, selector, every_steps: int) -> None:
        # CoverageStopCallback already owns the LIVE_PPO formatter.  Passing
        # the interval to it avoids printing the same transition once in the
        # parent callback and once again in this subclass.
        super().__init__(selector, every_steps=max(1, int(every_steps)))


def _canonical_wz(value: str) -> str:
    normalized = value.strip().lower()
    return f"wz{normalized}" if normalized.isdigit() else normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=scenario_ids())
    parser.add_argument("--wz", required=True, help="Work-zone id or number, e.g. wz1 or 1")
    parser.add_argument(
        "--layout", "--abc", dest="layout", default="random",
        choices=("a", "b", "c", "random"),
        help="Fixed layout, or random A/B/C per episode (default: random)",
    )
    parser.add_argument("--episodes", type=int, default=1,
                        help="Required valid episodes per selected layout")
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=V2_DEFAULT_PPO_EPOCHS)
    parser.add_argument("--live-log-every", type=int, default=25,
                        help="Print ego/SUMO/PPO status every N environment steps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--sumo-port", type=int, default=8813)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--show-traffic-cones", action="store_true",
                        help="Demo/debug only: spawn this setting's manual cone/sign props")
    parser.add_argument("--cones-only", action="store_true",
                        help="With --show-traffic-cones, omit warning signs")
    parser.add_argument("--model", help="Optional compatible V2 1311D/2D PPO model")
    parser.add_argument("--output", default="runs/one_setting_validation")
    args = parser.parse_args(argv)

    preflight = validate_scenario(args.scenario)
    if not preflight.ok:
        for error in preflight.errors:
            print(f"ERROR {error}")
        return 2

    available = list_setting_ids(args.scenario, runnable_only=True)
    scenario_wz = f"{args.scenario}/{_canonical_wz(args.wz)}"
    if args.layout == "random":
        settings = [f"{scenario_wz}/{layout}" for layout in ("a", "b", "c")]
        missing = [setting for setting in settings if setting not in available]
        if missing:
            print(f"ERROR random layout mode requires runnable A/B/C for {scenario_wz}")
            print("missing", missing)
            print("available", available)
            return 2
        selector = RandomLayoutSelector(settings, args.episodes, seed=args.seed)
    else:
        settings = [f"{scenario_wz}/{args.layout}"]
        if settings[0] not in available:
            print(f"ERROR {settings[0]} is not runnable")
            print("available", available)
            return 2
        selector = CoverageSelector(settings, args.episodes)

    configs = [load_scenario(setting) for setting in settings]
    cfg = configs[0]
    plan = plan_single(
        [item.episode.max_steps for item in configs], args.episodes, args.n_steps
    )
    total_timesteps = args.max_timesteps or plan.buffer_size_steps
    print("fixed_scenario_wz", scenario_wz)
    print("layout_mode", args.layout, "eligible", settings)
    print("rollout_plan", describe_plan(plan))
    print(
        "manual_od",
        f"origins={len(cfg.origin.spawn_points)}",
        f"finish_line={cfg.destination.finish_line}",
    )

    configure_initial_config(
        cfg,
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
            host=cfg.carla.host,
            carla_port=args.carla_port,
            cones_only=args.cones_only,
            load_town=False,
            # Episode resets may change the A/B/C props, but the spectator is
            # deliberately left where the user moves it in the CARLA window.
            move_spectator=False,
        )

    def make_env():
        from baseline.baseline_ours.road_arc_env_v2 import CarlaSumoGymEnv

        return CarlaSumoGymEnv(
            scenario=settings[0],
            config=cfg,
            worker_id=0,
            no_rendering_mode=args.no_rendering,
            scenario_selector=selector,
            episode_setup_callback=(visualizer.switch if visualizer is not None else None),
        )

    vec_env = DummyVecEnv([make_env])
    callback = LiveCoverageCallback(selector, args.live_log_every)
    try:
        model = build_or_load_ppo(
            vec_env,
            plan,
            model_path=args.model,
            ppo_epochs=args.ppo_epochs,
            seed=args.seed,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        save_ppo(model, args.output)
    finally:
        vec_env.close()
        if visualizer is not None:
            visualizer.close()

    print("coverage", selector.completed_counts)
    print("outcomes", dict(callback.outcomes))
    if not selector.complete:
        print("ERROR max timesteps reached before the requested episode coverage")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
