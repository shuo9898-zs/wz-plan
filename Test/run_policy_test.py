"""Run one frozen PPO checkpoint on one final held-out test scenario."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from baseline.PPO.runtime_v2 import configure_initial_config
from baseline.PPO.validate_allSwithoneExe import OrderedOriginSelector, OriginBinder
from config.scenario_selector import CoverageSelector
from logic.episode_termination_center_v2 import (
    CENTER_TERMINATION_CONTRACT_VERSION_V2,
)
from Test.test_case import PROJECT_ROOT, TEST_SPECS, TestCase, load_test_case
from validation_debug.runner_v2 import (
    _format_rate_v2,
    _outcome_flags_v2,
    _validate_model_spaces_v2,
    summarize_validation_episodes_v2,
)


DEFAULT_EPISODES_PER_ORIGIN = 50
DEFAULT_TEST_SEED = 1007
TEST_ENV_MODULE = "env.gym_wrapper_center_v2"
TEST_REWARD_MODULE = "logic.reward_v2"
TEST_EGO_JUDGEMENT = "ego_center_point_segment"


def _suppress_windows_crash_dialogs() -> None:
    """Let a failed simulator child exit instead of blocking on a GUI popup."""
    if os.name != "nt":
        return
    import ctypes

    ctypes.windll.kernel32.SetErrorMode(0x8003)


class _OriginTrackingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, selector: OrderedOriginSelector) -> None:
        super().__init__(env)
        self._selector = selector
        self.active_origin_index = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.active_origin_index = int(self._selector.current_ticket.origin_index)
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    if not result:
        raise ValueError("--policy-name must contain a filename-safe character")
    return result


def _background_traffic_profile(cfg: Any) -> dict[str, Any] | None:
    if cfg.sumo is None:
        return None
    return {
        "initial_background_vehicles": int(cfg.sumo.initial_background_vehicles),
        "max_background_vehicles": int(cfg.sumo.max_background_vehicles),
        "bg_spawn_rate_veh_s": float(cfg.sumo.bg_spawn_rate_veh_s),
        "bg_route_traffic_overrides": {
            route_id: dict(profile)
            for route_id, profile in sorted(
                cfg.sumo.bg_route_traffic_overrides.items()
            )
        },
    }


def _disable_background_traffic(cfg: Any) -> None:
    """Keep SUMO/Ego synchronization but remove every background agent."""
    if not cfg.uses_sumo or cfg.sumo is None:
        raise ValueError("background-traffic ablation requires a SUMO scenario")
    cfg.sumo.initial_background_vehicles = 0
    cfg.sumo.max_background_vehicles = 0
    cfg.sumo.bg_spawn_rate_veh_s = 0.0
    cfg.sumo.bg_route_traffic_overrides = {
        route_id: {
            "initial_background_vehicles": 0,
            "max_background_vehicles": 0,
            "bg_spawn_rate_veh_s": 0.0,
        }
        for route_id in cfg.sumo.bg_route_traffic_overrides
    }


def _require_carla(case: TestCase, host: str, port: int, timeout_s: float) -> None:
    try:
        import carla

        client = carla.Client(host, int(port))
        client.set_timeout(float(timeout_s))
        current = client.get_world().get_map().name.split("/")[-1]
    except Exception as exc:
        raise RuntimeError(f"CARLA is not ready at {host}:{port}: {exc}") from exc
    current_name = current.lower()
    expected_name = case.spec.town.lower()
    if current_name.endswith("_opt"):
        current_name = current_name[:-4]
    if expected_name.endswith("_opt"):
        expected_name = expected_name[:-4]
    if current_name != expected_name:
        raise RuntimeError(
            f"CARLA {host}:{port} has {current}, test {case.spec.scenario_id} "
            f"requires {case.spec.town}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=tuple(TEST_SPECS))
    parser.add_argument("--model", required=True, help="Frozen PPO .zip checkpoint")
    parser.add_argument(
        "--policy-name",
        help="Stable result label; default is the checkpoint filename stem",
    )
    parser.add_argument(
        "--episodes-per-origin",
        type=int,
        default=DEFAULT_EPISODES_PER_ORIGIN,
        help="Completed episodes for each of the three origins (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--rendering", dest="no_rendering", action="store_false")
    parser.set_defaults(no_rendering=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int)
    parser.add_argument("--tm-port", type=int)
    parser.add_argument("--sumo-port", type=int)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--live-log-every", type=int, default=100)
    parser.add_argument(
        "--background-traffic-mode",
        choices=("configured", "none"),
        default="configured",
        help=(
            "Keep configured traffic, or retain SUMO/Ego synchronization "
            "while setting every background-vehicle pool to zero"
        ),
    )
    parser.add_argument(
        "--run-root",
        default=str(PROJECT_ROOT / "Test" / "results"),
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Only for live geometry verification before manifest status=ready",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _suppress_windows_crash_dialogs()
    args = _parser().parse_args(argv)
    if args.episodes_per_origin < 1:
        raise ValueError("--episodes-per-origin must be positive")
    if args.live_log_every < 0:
        raise ValueError("--live-log-every cannot be negative")

    case = load_test_case(args.scenario, allow_blocked=args.allow_blocked)
    cfg = case.config
    configured_background_traffic = _background_traffic_profile(cfg)
    if args.background_traffic_mode == "none":
        if case.spec.scenario_id != "s2":
            raise ValueError(
                "--background-traffic-mode=none is currently bounded to the "
                "S2 diagnostic"
            )
        _disable_background_traffic(cfg)
    effective_background_traffic = _background_traffic_profile(cfg)
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint does not exist: {model_path}")
    policy_name = _safe_name(args.policy_name or model_path.stem)

    carla_port = int(args.carla_port or case.spec.carla_port)
    tm_port = int(args.tm_port or case.spec.tm_port)
    sumo_port = int(args.sumo_port or case.spec.sumo_port)
    configure_initial_config(
        cfg,
        carla_port=carla_port,
        tm_port=tm_port,
        sumo_port=sumo_port,
        no_rendering=args.no_rendering,
    )
    cfg.carla.host = str(args.host)
    _require_carla(case, args.host, carla_port, args.connect_timeout)

    env_module = importlib.import_module(TEST_ENV_MODULE)
    env_type = env_module.CarlaSumoGymEnv
    reward_module = importlib.import_module(TEST_REWARD_MODULE)
    reward_contract = str(
        getattr(reward_module, "REWARD_CONTRACT_VERSION_V2", "unknown")
    )
    reward_source = Path(reward_module.__file__).resolve()

    controller_path: Path | None = None
    original_jaywalker_controller: Any = None
    legacy_env_module: Any = None
    if case.spec.runtime_scenario_id == "s4":
        controller_module = importlib.import_module(
            "Test.Scenarios.test.S4_Town10HD_Jaywalker.jaywalker_controller"
        )
        legacy_env_module = importlib.import_module("env.carla_sumo_env")
        original_jaywalker_controller = legacy_env_module.JaywalkerController
        legacy_env_module.JaywalkerController = controller_module.JaywalkerController
        controller_path = Path(controller_module.__file__).resolve()

    origins = tuple(cfg.origin.spawn_points)
    selector = OrderedOriginSelector(
        [cfg.setting_id],
        {cfg.setting_id: len(origins)},
        repeats=args.episodes_per_origin,
    )
    binder = OriginBinder(selector, {cfg.setting_id: origins})

    def episode_setup(setting_id: str) -> None:
        binder.bind(setting_id)

    base_env = env_type(
        scenario=cfg.setting_id,
        config=cfg,
        mode="eval",
        worker_id=0,
        no_rendering_mode=args.no_rendering,
        scenario_selector=selector,
        episode_setup_callback=episode_setup,
    )
    binder.attach(base_env)
    env = _OriginTrackingWrapper(base_env, selector)

    model = PPO.load(str(model_path), device=args.device)
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    _validate_model_spaces_v2(model, env)

    target_episodes = len(origins) * int(args.episodes_per_origin)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = Path(args.run_root) / policy_name / case.spec.scenario_id
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = run_dir / f"episodes_{timestamp}.csv"
    summary_path = run_dir / f"summary_{timestamp}.json"

    rows: list[dict[str, Any]] = []
    origin_counts: Counter[int] = Counter()
    infra_faults: Counter[str] = Counter()
    attempted_steps = 0
    valid_steps = 0
    episode_step = 0
    episode_return = 0.0
    completed = 0
    interrupted = False
    max_sumo_mirrors = 0
    steps_with_sumo_mirrors = 0

    print(
        f"[TEST {case.spec.scenario_id}] policy={policy_name} "
        f"model={model_path.name} episodes={target_episodes} "
        f"({args.episodes_per_origin}/origin) deterministic={not args.stochastic} "
        f"background_traffic={args.background_traffic_mode}",
        flush=True,
    )
    try:
        with episodes_path.open("w", newline="", encoding="utf-8") as stream:
            fields = (
                "episode", "origin_index", "origin_episode", "ticket_id",
                "valid_episode_step_end", "episode_length", "episode_return",
                "reason", "success", "collision", "timeout",
                "workzone_violation", "off_road", "safety_failure",
                "other_failure",
            )
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            observation, _ = env.reset(seed=args.seed)
            while completed < target_episodes:
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                action_array = np.asarray(action, dtype=np.float32).reshape(-1)
                observation, reward, terminated, truncated, info = env.step(
                    action_array
                )
                attempted_steps += 1
                episode_step += 1
                episode_return += float(reward)
                sumo_mirrors = int(info.get("sumo_mirrors_in_carla", 0))
                max_sumo_mirrors = max(max_sumo_mirrors, sumo_mirrors)
                if sumo_mirrors > 0:
                    steps_with_sumo_mirrors += 1
                done = bool(terminated or truncated)
                reason = str(info.get("reason", "running"))
                if args.live_log_every and (
                    done or attempted_steps == 1
                    or attempted_steps % args.live_log_every == 0
                ):
                    print(
                        f"TEST step={attempted_steps} ep={completed + 1}/"
                        f"{target_episodes} origin={env.active_origin_index} "
                        f"ep_step={episode_step} reward={float(reward):.3f} "
                        f"return={episode_return:.3f} reason={reason}",
                        flush=True,
                    )
                if not done:
                    continue
                if reason in CoverageSelector.INFRASTRUCTURE_REASONS:
                    infra_faults[reason] += 1
                    print(f"INFRA_RETRY reason={reason}", flush=True)
                else:
                    origin_index = int(env.active_origin_index)
                    origin_counts[origin_index] += 1
                    completed += 1
                    valid_steps += episode_step
                    row: dict[str, Any] = {
                        "episode": completed,
                        "origin_index": origin_index,
                        "origin_episode": origin_counts[origin_index],
                        "ticket_id": str(
                            info.get(
                                "ticket_id",
                                f"{cfg.setting_id}#origin{origin_index}",
                            )
                        ),
                        "valid_episode_step_end": valid_steps,
                        "episode_length": episode_step,
                        "episode_return": episode_return,
                        "reason": reason,
                    }
                    row.update(_outcome_flags_v2(reason))
                    rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                episode_step = 0
                episode_return = 0.0
                if completed < target_episodes:
                    observation, _ = env.reset()
    except KeyboardInterrupt:
        interrupted = True
        print("KeyboardInterrupt: closing test cleanly.", flush=True)
    finally:
        try:
            env.close()
        finally:
            if legacy_env_module is not None:
                legacy_env_module.JaywalkerController = original_jaywalker_controller

    protocol_complete = completed == target_episodes and selector.complete
    metrics = summarize_validation_episodes_v2(rows, origin_count=len(origins))
    summary = {
        "dataset_split": "test",
        "test_id": case.spec.scenario_id,
        "runtime_scenario_id": case.spec.runtime_scenario_id,
        "display_name": case.spec.display_name,
        "town": case.spec.town,
        "setting_id": cfg.setting_id,
        "manifest_status": case.status,
        "policy_name": policy_name,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "seed": int(args.seed),
        "deterministic": not bool(args.stochastic),
        "experimental_condition": {
            "background_traffic_mode": args.background_traffic_mode,
            "sumo_cosimulation_retained": bool(cfg.uses_sumo),
            "configured_background_traffic": configured_background_traffic,
            "effective_background_traffic": effective_background_traffic,
            "max_sumo_mirrors_in_carla": max_sumo_mirrors,
            "steps_with_sumo_mirrors": steps_with_sumo_mirrors,
        },
        "env_module": TEST_ENV_MODULE,
        "ego_judgement": TEST_EGO_JUDGEMENT,
        "termination_contract": CENTER_TERMINATION_CONTRACT_VERSION_V2,
        "reward_module": TEST_REWARD_MODULE,
        "reward_contract": reward_contract,
        "reward_source_sha256": _sha256(reward_source),
        "test_inputs": {
            "manifest": str(case.manifest_path),
            "manifest_sha256": _sha256(case.manifest_path),
            "config": str(case.config_path),
            "config_sha256": _sha256(case.config_path),
            "network_sha256": (
                _sha256(case.network_path) if case.network_path else None
            ),
            "route_sha256": _sha256(case.route_path) if case.route_path else None,
            "jaywalker_controller": (
                str(controller_path) if controller_path else None
            ),
            "jaywalker_controller_sha256": (
                _sha256(controller_path) if controller_path else None
            ),
        },
        "protocol": {
            "origin_count": len(origins),
            "episodes_per_origin": int(args.episodes_per_origin),
            "target_episodes": target_episodes,
            "infrastructure_faults_excluded": True,
        },
        "completed": protocol_complete,
        "interrupted": interrupted,
        "completed_episodes": completed,
        "attempted_steps": attempted_steps,
        "valid_episode_steps": valid_steps,
        "infrastructure_faults": dict(sorted(infra_faults.items())),
        **metrics,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    overall = metrics["overall"]
    print(
        f"[TEST {case.spec.scenario_id}] done ep={completed}/{target_episodes} "
        f"success={_format_rate_v2(overall['success_rate'])} "
        f"end={overall['reason_counts']} summary={summary_path}",
        flush=True,
    )
    if not interrupted and not protocol_complete:
        raise RuntimeError("test ended before every origin reached its episode quota")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
