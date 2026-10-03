"""Frozen V2 policy closed-loop runner for validation debug."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from stable_baselines3 import PPO

from baseline.PPO.runtime_v2 import configure_initial_config
from baseline.PPO.validate_allSwithoneExe import (
    OrderedOriginSelector,
    OriginBinder,
)
from config.scenario_selector import CoverageSelector
from logic.reward_v2 import (
    COLLISION_JAYWALKER_V2,
    COLLISION_SUMO_VEHICLE_V2,
    GOAL_REACHED_V2,
    OFF_ROAD_V2,
    TIMEOUT_V2,
    WORKZONE_VIOLATION_V2,
)
from logic.termination_checker_v2 import TERMINATION_CONTRACT_VERSION_V2
from validation_debug.loader_v2 import (
    PROJECT_ROOT,
    ValidationCaseV2,
    load_validation_case_v2,
    validation_debug_spec_v2,
)
from validation_debug.visuals_v2 import ValidationDebugVisualWrapperV2


DEFAULT_EPISODES_PER_ORIGIN_V2 = 10
DEFAULT_LIVE_LOG_EVERY_V2 = 100
DEFAULT_RUN_ROOT_V2 = PROJECT_ROOT / "runs" / "ppo_v2_lanegrid_fresh_20260816"
DEFAULT_RUN_STATE_V2 = DEFAULT_RUN_ROOT_V2 / "run_state.json"
DEFAULT_BEST_MODEL_FALLBACK_V2 = DEFAULT_RUN_ROOT_V2 / "models" / "best_model.zip"
SWEPT_EGO_JUDGEMENT_V2 = "swept_ego_obb"


def _canonical_town_v2(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized[:-4] if normalized.endswith("_opt") else normalized


def carla_launch_command_v2(scenario_id: str) -> str:
    spec = validation_debug_spec_v2(scenario_id)
    exe = os.environ.get("CARLA_EXE", "CarlaUE4.exe")
    return (
        f'"{exe}" /Game/Carla/Maps/{spec.town} '
        f"-carla-rpc-port={spec.carla_port}"
    )


def resolve_v2_best_model_path(
    override: str | Path | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PPO model does not exist: {path}")
        return path, None

    state: dict[str, Any] | None = None
    if DEFAULT_RUN_STATE_V2.is_file():
        state = json.loads(DEFAULT_RUN_STATE_V2.read_text(encoding="utf-8"))
        best = state.get("best_model") or {}
        authored = best.get("file")
        candidate = Path(str(authored)) if authored else None
        if candidate is not None and candidate.is_file():
            path = candidate.resolve()
        else:
            path = DEFAULT_BEST_MODEL_FALLBACK_V2.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "V2 run_state does not point to an available best_model.zip: "
                f"{DEFAULT_RUN_STATE_V2}"
            )
        expected_sha = str(best.get("sha256", "")).strip().lower()
        if expected_sha:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_sha:
                raise RuntimeError(
                    f"V2 best-model SHA256 mismatch: {path} "
                    f"expected={expected_sha} actual={digest}"
                )
        return path, state

    path = DEFAULT_BEST_MODEL_FALLBACK_V2.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Missing the default V2 best model and run_state: "
            f"{DEFAULT_BEST_MODEL_FALLBACK_V2}"
        )
    return path, None


def _require_live_carla_v2(
    case: ValidationCaseV2,
    *,
    host: str,
    port: int,
    timeout_s: float,
) -> None:
    try:
        import carla

        client = carla.Client(host, int(port))
        client.set_timeout(float(timeout_s))
        world = client.get_world()
        current = world.get_map().name.split("/")[-1]
    except Exception as error:
        command = carla_launch_command_v2(case.spec.scenario_id)
        raise RuntimeError(
            f"CARLA is not ready at {host}:{port}. Start it first with:\n"
            f"{command}\nOriginal error: {error}"
        ) from error
    if _canonical_town_v2(current) != _canonical_town_v2(case.spec.town):
        command = carla_launch_command_v2(case.spec.scenario_id)
        raise RuntimeError(
            f"CARLA {host}:{port} has {current}, but this validation case "
            f"requires {case.spec.town}. Restart/load it with:\n{command}"
        )


def build_validation_debug_parser_v2(
    scenario_id: str,
) -> argparse.ArgumentParser:
    spec = validation_debug_spec_v2(scenario_id)
    parser = argparse.ArgumentParser(
        description=(
            f"Frozen-policy debug for validation "
            f"{spec.scenario_id.upper()} ({spec.display_name}); runtime "
            f"scenario ID is {spec.runtime_scenario_id.upper()}."
        )
    )
    parser.add_argument(
        "--episodes-per-origin",
        type=int,
        default=DEFAULT_EPISODES_PER_ORIGIN_V2,
        help=(
            "Completed non-infrastructure episodes for each origin "
            "(default: 10; three origins = 30 episodes)."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", help="Override the default V2 best model")
    parser.add_argument(
        "--ego-judgement",
        choices=(SWEPT_EGO_JUDGEMENT_V2, "ego_center_point_segment"),
        default=SWEPT_EGO_JUDGEMENT_V2,
        help="Work-zone geometry reference; default keeps the swept-OBB contract.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions; default is deterministic evaluation.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Optionally throttle evaluation to the 10 Hz simulation clock.",
    )
    parser.add_argument(
        "--no-rendering",
        action="store_true",
        help="Disable CARLA rendering for automated validation.",
    )
    parser.add_argument(
        "--no-props",
        action="store_true",
        help="Do not spawn physical cones/signs; laser geometry remains visible.",
    )
    parser.add_argument(
        "--cones-only",
        action="store_true",
        help="Spawn cones but omit the large board/sign props.",
    )
    parser.add_argument(
        "--live-log-every", type=int, default=DEFAULT_LIVE_LOG_EVERY_V2
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=spec.carla_port)
    parser.add_argument("--tm-port", type=int, default=spec.tm_port)
    parser.add_argument("--sumo-port", type=int, default=spec.sumo_port)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--run-dir",
        default=str(PROJECT_ROOT / "runs" / "validation_debug" / spec.scenario_id),
    )
    return parser


def _validate_model_spaces_v2(model: PPO, env: Any) -> None:
    model_obs = tuple(getattr(model.observation_space, "shape", ()) or ())
    env_obs = tuple(getattr(env.observation_space, "shape", ()) or ())
    model_action = tuple(getattr(model.action_space, "shape", ()) or ())
    env_action = tuple(getattr(env.action_space, "shape", ()) or ())
    if model_obs != env_obs or model_action != env_action:
        raise RuntimeError(
            "V2 best model is incompatible with this environment: "
            f"obs {model_obs}!={env_obs} or action {model_action}!={env_action}"
        )


def _finite(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


SUCCESS_REASONS_V2 = frozenset({GOAL_REACHED_V2})
COLLISION_REASONS_V2 = frozenset(
    {COLLISION_JAYWALKER_V2, COLLISION_SUMO_VEHICLE_V2}
)


def _outcome_flags_v2(reason: str) -> dict[str, bool]:
    normalized = str(reason)
    success = normalized in SUCCESS_REASONS_V2
    collision = normalized in COLLISION_REASONS_V2 or normalized.startswith(
        "collision_"
    )
    workzone_violation = normalized == WORKZONE_VIOLATION_V2
    off_road = normalized == OFF_ROAD_V2
    timeout = normalized == TIMEOUT_V2
    safety_failure = collision or workzone_violation or off_road
    known = success or safety_failure or timeout
    return {
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "workzone_violation": workzone_violation,
        "off_road": off_road,
        "safety_failure": safety_failure,
        "other_failure": not known,
    }


def _summarize_group_v2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    reason_counts = Counter(str(row["reason"]) for row in rows)
    flags = {
        name: sum(bool(row[name]) for row in rows)
        for name in (
            "success",
            "collision",
            "timeout",
            "workzone_violation",
            "off_road",
            "safety_failure",
            "other_failure",
        )
    }
    result: dict[str, Any] = {
        "episodes": count,
        "failure_count": count - flags["success"],
        "failure_rate": ((count - flags["success"]) / count if count else None),
        "mean_episode_length": (
            sum(int(row["episode_length"]) for row in rows) / count
            if count
            else None
        ),
        "mean_episode_return": (
            sum(float(row["episode_return"]) for row in rows) / count
            if count
            else None
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    for name, value in flags.items():
        result[f"{name}_count"] = value
        result[f"{name}_rate"] = value / count if count else None
    return result


def summarize_validation_episodes_v2(
    rows: Sequence[Mapping[str, Any]],
    *,
    origin_count: int,
) -> dict[str, Any]:
    """Return overall and per-origin outcome metrics for completed episodes."""
    by_origin: dict[str, Any] = {}
    for origin_index in range(int(origin_count)):
        origin_rows = [
            row for row in rows if int(row["origin_index"]) == origin_index
        ]
        by_origin[f"origin{origin_index}"] = _summarize_group_v2(origin_rows)
    return {
        "overall": _summarize_group_v2(rows),
        "by_origin": by_origin,
    }


def _format_rate_v2(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _print_metrics_v2(scope: str, metrics: Mapping[str, Any]) -> None:
    print(
        f"VALIDATION_METRICS scope={scope} episodes={metrics['episodes']} "
        f"success={metrics['success_count']}({ _format_rate_v2(metrics['success_rate'])}) "
        f"collision={metrics['collision_count']}({ _format_rate_v2(metrics['collision_rate'])}) "
        f"timeout={metrics['timeout_count']}({ _format_rate_v2(metrics['timeout_rate'])}) "
        f"workzone_violation={metrics['workzone_violation_count']}"
        f"({ _format_rate_v2(metrics['workzone_violation_rate'])}) "
        f"off_road={metrics['off_road_count']}({ _format_rate_v2(metrics['off_road_rate'])}) "
        f"mean_return={metrics['mean_episode_return']} "
        f"mean_length={metrics['mean_episode_length']} "
        f"reasons={metrics['reason_counts']}",
        flush=True,
    )


def _print_live_v2(
    *,
    runtime_step: int,
    episode: int,
    origin_index: int,
    episode_step: int,
    reward: float,
    episode_return: float,
    action: np.ndarray,
    info: dict[str, Any],
) -> None:
    reason = str(info.get("reason", "running"))
    speed = _finite(
        info.get("control_ego_speed_mps", info.get("ego_speed_mps"))
    )
    progress = _finite(info.get("reward_normalized_progress"))
    print(
        f"VAL runtime_step={runtime_step} episode={episode} origin={origin_index} "
        f"ep_step={episode_step} "
        f"reward={reward:.3f} return={episode_return:.3f} "
        f"speed={speed:.3f}m/s progress={progress:.3f} "
        f"action=({float(action[0]):.3f},{float(action[1]):.3f}) "
        f"reason={reason}",
        flush=True,
    )


def main_for_validation_scenario_v2(
    scenario_id: str,
    argv: list[str] | None = None,
) -> int:
    spec = validation_debug_spec_v2(scenario_id)
    args = build_validation_debug_parser_v2(spec.scenario_id).parse_args(argv)
    if args.episodes_per_origin < 1:
        raise ValueError("--episodes-per-origin must be positive")
    if args.live_log_every < 0:
        raise ValueError("--live-log-every cannot be negative")

    case = load_validation_case_v2(spec.scenario_id)
    cfg = case.config
    configure_initial_config(
        cfg,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        sumo_port=args.sumo_port,
        no_rendering=args.no_rendering,
    )
    cfg.carla.host = str(args.host)
    try:
        _require_live_carla_v2(
            case,
            host=args.host,
            port=args.carla_port,
            timeout_s=args.connect_timeout,
        )
    except RuntimeError as error:
        print(f"ERROR {error}", flush=True)
        return 2

    model_path, run_state = resolve_v2_best_model_path(args.model)
    model = PPO.load(str(model_path), device=args.device)
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    best_meta = (run_state or {}).get("best_model") or {}
    print(
        f"[VAL {spec.scenario_id}] start town={spec.town} "
        f"episodes={len(cfg.origin.spawn_points) * args.episodes_per_origin} "
        f"model={model_path.name}",
        flush=True,
    )
    if case.status == "blocked":
        print(
            "WARNING manifest status=blocked: this is an owner-authored "
            "live-debug case, not yet an approved validation benchmark.",
            flush=True,
        )

    reward_module_name = os.environ.get("AUG24_REWARD_MODULE", "logic.reward_v2")
    reward_module = importlib.import_module(reward_module_name)
    reward_contract = str(reward_module.REWARD_CONTRACT_VERSION_V2)
    reward_source = Path(reward_module.__file__).resolve()
    wrapper_module_name = os.environ.get("AUG24_GYM_WRAPPER_MODULE")

    if wrapper_module_name:
        wrapper_module = importlib.import_module(wrapper_module_name)
        if args.ego_judgement == "ego_center_point_segment":
            SelectedGymEnvV2 = wrapper_module.CarlaSumoGymEnvCenterV2
            from logic.episode_termination_center_v2 import (
                CENTER_TERMINATION_CONTRACT_VERSION_V2 as termination_contract,
            )
        else:
            SelectedGymEnvV2 = wrapper_module.CarlaSumoGymEnvV2
            termination_contract = TERMINATION_CONTRACT_VERSION_V2
    elif args.ego_judgement == "ego_center_point_segment":
        from env.gym_wrapper_center_v2 import (
            CarlaSumoGymEnvCenterV2 as SelectedGymEnvV2,
        )
        from logic.episode_termination_center_v2 import (
            CENTER_TERMINATION_CONTRACT_VERSION_V2 as termination_contract,
        )
    else:
        from env.gym_wrapper_v2 import CarlaSumoGymEnvV2 as SelectedGymEnvV2

        termination_contract = TERMINATION_CONTRACT_VERSION_V2

    authored_origins = tuple(cfg.origin.spawn_points)
    target_episodes = len(authored_origins) * int(args.episodes_per_origin)
    selector = OrderedOriginSelector(
        [cfg.setting_id],
        {cfg.setting_id: len(authored_origins)},
        repeats=args.episodes_per_origin,
    )
    binder = OriginBinder(
        selector,
        {cfg.setting_id: authored_origins},
    )

    def episode_setup(setting_id: str) -> None:
        binder.bind(setting_id)

    base_env = SelectedGymEnvV2(
        scenario=cfg.setting_id,
        config=cfg,
        mode="eval",
        worker_id=0,
        no_rendering_mode=args.no_rendering,
        scenario_selector=selector,
        episode_setup_callback=episode_setup,
    )
    binder.attach(base_env)
    env = ValidationDebugVisualWrapperV2(
        base_env,
        case=case,
        selector=selector,
        show_props=not args.no_props,
        cones_only=args.cones_only,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    episodes_path = run_dir / f"episodes_{timestamp}.csv"
    summary_path = run_dir / f"summary_{timestamp}.json"
    infra_faults: Counter[str] = Counter()
    episode_rows: list[dict[str, Any]] = []
    origin_episode_counts: Counter[int] = Counter()
    attempted_steps = 0
    valid_episode_steps = 0
    completed_episodes = 0
    episode_step = 0
    episode_return = 0.0
    interrupted = False

    try:
        _validate_model_spaces_v2(model, env)
        with episodes_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "episode",
                    "origin_index",
                    "origin_episode",
                    "ticket_id",
                    "valid_episode_step_end",
                    "episode_length",
                    "episode_return",
                    "reason",
                    "success",
                    "collision",
                    "timeout",
                    "workzone_violation",
                    "off_road",
                    "safety_failure",
                    "other_failure",
                ),
            )
            writer.writeheader()
            observation, _ = env.reset(seed=args.seed)
            while completed_episodes < target_episodes:
                cycle_started = time.perf_counter()
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                action_array = np.asarray(action, dtype=np.float32).reshape(-1)
                observation, reward, terminated, truncated, info = env.step(
                    action_array
                )
                done = bool(terminated or truncated)
                attempted_steps += 1
                episode_step += 1
                episode_return += float(reward)

                if args.realtime:
                    remaining = float(cfg.episode.sim_dt) - (
                        time.perf_counter() - cycle_started
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)

                reason = str(info.get("reason", "running"))
                infra = done and reason in CoverageSelector.INFRASTRUCTURE_REASONS

                if args.live_log_every and (
                    done
                    or attempted_steps == 1
                    or attempted_steps % args.live_log_every == 0
                ):
                    _print_live_v2(
                        runtime_step=attempted_steps,
                        episode=completed_episodes + 1,
                        origin_index=env.active_origin_index,
                        episode_step=episode_step,
                        reward=float(reward),
                        episode_return=episode_return,
                        action=action_array,
                        info=info,
                    )

                if done:
                    if infra:
                        infra_faults[reason] += 1
                        print(f"INFRA_RETRY reason={reason}", flush=True)
                    else:
                        origin_index = int(env.active_origin_index)
                        origin_episode_counts[origin_index] += 1
                        completed_episodes += 1
                        valid_episode_steps += episode_step
                        row: dict[str, Any] = {
                            "episode": completed_episodes,
                            "origin_index": origin_index,
                            "origin_episode": origin_episode_counts[origin_index],
                            "ticket_id": str(
                                info.get(
                                    "ticket_id",
                                    f"{cfg.setting_id}#origin{origin_index}",
                                )
                            ),
                            "valid_episode_step_end": valid_episode_steps,
                            "episode_length": episode_step,
                            "episode_return": episode_return,
                            "reason": reason,
                        }
                        row.update(_outcome_flags_v2(reason))
                        episode_rows.append(row)
                        writer.writerow(row)
                        file.flush()
                    episode_step = 0
                    episode_return = 0.0
                    if completed_episodes < target_episodes:
                        observation, _ = env.reset()
    except KeyboardInterrupt:
        interrupted = True
        print("KeyboardInterrupt: closing validation debug cleanly.", flush=True)
    finally:
        env.close()

    protocol_complete = (
        completed_episodes == target_episodes and selector.complete
    )
    metrics = summarize_validation_episodes_v2(
        episode_rows,
        origin_count=len(authored_origins),
    )
    summary = {
        "validation_id": spec.scenario_id,
        "runtime_scenario_id": spec.runtime_scenario_id,
        "display_name": spec.display_name,
        "setting_id": cfg.setting_id,
        "seed": int(args.seed),
        "deterministic": not bool(args.stochastic),
        "ego_judgement": str(args.ego_judgement),
        "reward_contract": reward_contract,
        "reward_source_sha256": hashlib.sha256(reward_source.read_bytes()).hexdigest(),
        "termination_contract": termination_contract,
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "protocol": {
            "origin_count": len(authored_origins),
            "episodes_per_origin": int(args.episodes_per_origin),
            "target_episodes": target_episodes,
            "infrastructure_faults_excluded": True,
        },
        "completed": protocol_complete,
        "interrupted": interrupted,
        "completed_episodes": completed_episodes,
        "attempted_steps": attempted_steps,
        "valid_episode_steps": valid_episode_steps,
        "infrastructure_faults": dict(sorted(infra_faults.items())),
        **metrics,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    overall = metrics["overall"]
    print(
        f"[VAL {spec.scenario_id}] done ep={completed_episodes}/{target_episodes} "
        f"Rmean={float(overall['mean_episode_return']):.2f} "
        f"success={_format_rate_v2(overall['success_rate'])} "
        f"end={overall['reason_counts']} infra={dict(infra_faults)}",
        flush=True,
    )
    if not interrupted and not protocol_complete:
        raise RuntimeError(
            "Validation episode protocol ended without exactly the requested "
            "number of completed episodes for every origin"
        )
    return 130 if interrupted else 0


__all__ = [
    "DEFAULT_BEST_MODEL_FALLBACK_V2",
    "DEFAULT_RUN_STATE_V2",
    "DEFAULT_EPISODES_PER_ORIGIN_V2",
    "build_validation_debug_parser_v2",
    "carla_launch_command_v2",
    "main_for_validation_scenario_v2",
    "resolve_v2_best_model_path",
    "summarize_validation_episodes_v2",
]
