"""Synchronous PPO V2 training with one fixed Town per simulator server.

This entry point is deliberately separate from ``train_all_v2``.  It loads
Town02, Town05 and Town10HD on three independent CARLA servers once, launches
one single-environment rollout process per server, and uses a strict barrier:
all three frozen-policy fragments must finish before the parent merges them
and performs exactly one PPO update.

The three rollout processes never update policy parameters and never change
Town.  Within a Town they only rotate through that Town's scenarios/settings.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv

from baseline.baseline_ours.global_rollout_checkpoint import (
    atomic_write_json,
    canonical_digest,
    file_digest,
    load_rollout_buffer,
    policy_digest,
    save_model_atomic,
    save_rollout_buffer,
)
from baseline.baseline_ours.runtime_v2 import (
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    V2_DEFAULT_PPO_EPOCHS,
    build_or_load_ppo,
    configure_initial_config,
)
from baseline.baseline_ours.training_config_v2 import DEFAULT_THREE_SERVER_TRAINING_V2
from baseline.baseline_ours.training_logging import (
    RolloutTelemetryCallback,
    TrainingTelemetry,
    print_context_table,
    train_ppo_with_telemetry,
)
from baseline.baseline_ours.valid_rollout import collect_valid_rollouts
from baseline.baseline_ours.validate_allSwithoneExe import (
    DEFAULT_MAP_LOADER,
    OriginBinder,
    OrderedOriginSelector,
    TownPhase,
    _checkpoint_plan_payload,
    _global_plan,
    _make_plan,
    _outcome_records,
    _town_step_quotas,
    build_phase_plans,
    merge_rollout_buffers,
)
from env.observation_encoder_v2 import DEFAULT_OBSERVATION_DIM_V2
from tools.preflight import validate_scenario


SCHEMA_VERSION = 1
THREE_SERVER_CONTRACT = "baseline_ours_road_arc_no_speed_bonus_v1"
_TRAINING_DEFAULTS = DEFAULT_THREE_SERVER_TRAINING_V2
TOWNS = tuple(worker.town for worker in _TRAINING_DEFAULTS.town_workers)
DEFAULT_TOWN_STEPS = _TRAINING_DEFAULTS.town_step_quotas


@dataclass(frozen=True)
class ServerAssignment:
    worker_id: int
    town: str
    carla_port: int
    tm_port: int
    sumo_port: int


class _SpaceOnlyEnv(gym.Env):
    """Space provider used by the parent; it never touches CARLA or SUMO."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(DEFAULT_OBSERVATION_DIM_V2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):  # pragma: no cover - parent never samples this env
        raise RuntimeError("The parent space-only environment cannot be stepped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=_TRAINING_DEFAULTS.run_root)
    parser.add_argument("--total-updates", type=int, default=_TRAINING_DEFAULTS.total_updates)
    parser.add_argument(
        "--episodes-per-origin",
        type=int,
        default=_TRAINING_DEFAULTS.episodes_per_origin,
    )
    parser.add_argument("--town02-steps", type=int, default=DEFAULT_TOWN_STEPS["Town02"])
    parser.add_argument("--town05-steps", type=int, default=DEFAULT_TOWN_STEPS["Town05"])
    parser.add_argument("--town10hd-steps", type=int, default=DEFAULT_TOWN_STEPS["Town10HD"])
    parser.add_argument(
        "--batch-size", type=int, default=_TRAINING_DEFAULTS.ppo.batch_size
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=V2_DEFAULT_PPO_EPOCHS,
        help=(
            "maximum PPO passes over each synchronized buffer "
            f"(default: {V2_DEFAULT_PPO_EPOCHS})"
        ),
    )
    parser.set_defaults(target_kl=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=_TRAINING_DEFAULTS.device,
    )
    parser.add_argument("--seed", type=int, default=_TRAINING_DEFAULTS.seed)
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--live-log-every",
        type=int,
        default=_TRAINING_DEFAULTS.live_log_every_steps,
    )
    parser.add_argument("--carla-host", default=_TRAINING_DEFAULTS.carla_host)
    parser.add_argument(
        "--carla-ports",
        type=int,
        nargs=3,
        default=[worker.carla_port for worker in _TRAINING_DEFAULTS.town_workers],
        metavar=("TOWN02", "TOWN05", "TOWN10HD"),
    )
    parser.add_argument(
        "--tm-ports",
        type=int,
        nargs=3,
        default=[
            worker.traffic_manager_port for worker in _TRAINING_DEFAULTS.town_workers
        ],
        metavar=("TOWN02", "TOWN05", "TOWN10HD"),
    )
    parser.add_argument(
        "--sumo-ports",
        type=int,
        nargs=3,
        default=[worker.sumo_port for worker in _TRAINING_DEFAULTS.town_workers],
        metavar=("TOWN02", "TOWN05", "TOWN10HD"),
    )
    parser.add_argument("--map-loader", type=Path, default=DEFAULT_MAP_LOADER)
    parser.add_argument(
        "--map-rpc-timeout",
        type=float,
        default=_TRAINING_DEFAULTS.map_rpc_timeout_s,
    )
    parser.add_argument(
        "--map-ready-sleep",
        type=float,
        default=_TRAINING_DEFAULTS.map_ready_sleep_s,
    )
    parser.add_argument(
        "--skip-map-load",
        action="store_true",
        help="Servers are already on their assigned Towns; verify manually and do not load maps",
    )
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Enable rendering (default is no-rendering on all three workers)",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name, value in {
        "--total-updates": args.total_updates,
        "--episodes-per-origin": args.episodes_per_origin,
        "--town02-steps": args.town02_steps,
        "--town05-steps": args.town05_steps,
        "--town10hd-steps": args.town10hd_steps,
        "--batch-size": args.batch_size,
        "--ppo-epochs": args.ppo_epochs,
        "--live-log-every": args.live_log_every,
    }.items():
        if value < 1:
            parser.error(f"{name} must be positive")
    if args.map_rpc_timeout <= 0.0 or not math.isfinite(args.map_rpc_timeout):
        parser.error("--map-rpc-timeout must be finite and positive")
    if args.map_ready_sleep < 0.0 or not math.isfinite(args.map_ready_sleep):
        parser.error("--map-ready-sleep must be finite and non-negative")
    if args.initial_model is not None:
        model_path = args.initial_model.with_suffix(".zip")
        if not model_path.is_file():
            parser.error(f"--initial-model not found: {model_path}")
    try:
        _assignments(args)
    except ValueError as error:
        parser.error(str(error))


def _assignments(args: argparse.Namespace) -> tuple[ServerAssignment, ...]:
    groups = (args.carla_ports, args.tm_ports, args.sumo_ports)
    if any(len(group) != 3 for group in groups):
        raise ValueError("exactly three CARLA, TM, and SUMO ports are required")
    all_ports: List[int] = []
    for group in groups:
        for port in group:
            if not 1 <= int(port) <= 65535:
                raise ValueError(f"invalid server port: {port}")
            all_ports.append(int(port))
    # CARLA also owns rpc_port + 1 for streaming.
    reserved = [int(port) + 1 for port in args.carla_ports]
    if any(port > 65535 for port in reserved):
        raise ValueError("a CARLA RPC port leaves no valid streaming port")
    combined = all_ports + reserved
    if len(set(combined)) != len(combined):
        raise ValueError(
            "CARLA RPC/streaming, Traffic Manager, and SUMO ports must all be unique"
        )
    return tuple(
        ServerAssignment(
            worker_id=index,
            town=town,
            carla_port=int(args.carla_ports[index]),
            tm_port=int(args.tm_ports[index]),
            sumo_port=int(args.sumo_ports[index]),
        )
        for index, town in enumerate(TOWNS)
    )


def _town_quotas(args: argparse.Namespace, phases: Sequence[TownPhase]) -> Dict[str, int]:
    return _town_step_quotas(
        phases,
        None,
        {
            "Town02": args.town02_steps,
            "Town05": args.town05_steps,
            "Town10HD": args.town10hd_steps,
        },
        None,
        require_worst_case=False,
    )


def _training_plan(
    args: argparse.Namespace,
    phases: Sequence[TownPhase],
    town_quotas: Mapping[str, int],
    assignments: Sequence[ServerAssignment],
) -> Dict[str, Any]:
    base = _checkpoint_plan_payload(
        phases,
        town_quotas,
        episodes_per_origin=args.episodes_per_origin,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        target_kl=args.target_kl,
    )
    initial_model = None
    if args.initial_model is not None:
        path = args.initial_model.with_suffix(".zip").resolve()
        initial_model = {"file": str(path), "sha256": file_digest(path)}
    workspace_root = Path(__file__).resolve().parents[2]
    local_root = Path(__file__).resolve().parent
    frozen_sources = (
        Path(__file__).resolve(),
        local_root / "encoder_v2.py",
        local_root / "runtime_v2.py",
        local_root / "training_config_v2.py",
        local_root / "valid_rollout.py",
        local_root / "road_arc_env_v2.py",
        local_root / "road_arc_progress_reward.py",
        local_root / "road_arc_reference.py",
        local_root / "road_arc_references_v1.json",
        workspace_root / "config" / "scenario_config.py",
        workspace_root / "env" / "observation_encoder_v2.py",
        workspace_root / "env" / "gym_wrapper_v2.py",
        workspace_root / "env" / "carla_sumo_env_v2.py",
        workspace_root / "env" / "sumo_runtime_v2.py",
        workspace_root / "sync" / "background_traffic_v2.py",
        workspace_root / "logic" / "termination_checker_v2.py",
        workspace_root / "logic" / "episode_termination_v2.py",
        workspace_root / "logic" / "scenario_geometry_adapter_v2.py",
        workspace_root / "baseline" / "controllers_v2.py",
    )
    return {
        "contract": THREE_SERVER_CONTRACT,
        "total_updates": int(args.total_updates),
        "steps_per_update": int(sum(town_quotas.values())),
        "initial_model": initial_model,
        "servers": [assignment.__dict__ for assignment in assignments],
        "rollout_and_ppo": base,
        "frozen_source_sha256": {
            str(path.relative_to(workspace_root)): file_digest(path)
            for path in frozen_sources
        },
    }


def _new_state(plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "plan": dict(plan),
        "plan_fingerprint": _plan_fingerprint(plan),
        "next_update": 1,
        "last_model": None,
        "last_model_alias": None,
        "best_model": None,
        "active_update": None,
        "completed_updates": [],
    }


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    immutable_plan = dict(plan)
    immutable_plan.pop("total_updates", None)
    return canonical_digest(immutable_plan)


def _load_state(path: Path, plan: Mapping[str, Any]) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"resume state not found: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported three-server checkpoint schema")
    if state.get("plan_fingerprint") != _plan_fingerprint(plan):
        raise ValueError(
            "training parameters, ports, scenario inputs, or V2 source files changed; "
            "start a new --run-root"
        )
    return state


def _map_commands(
    args: argparse.Namespace,
    assignments: Sequence[ServerAssignment],
) -> List[List[str]]:
    return [
        [
            sys.executable,
            str(args.map_loader),
            assignment.town,
            "--host",
            args.carla_host,
            "--port",
            str(assignment.carla_port),
            "--rpc-timeout",
            str(args.map_rpc_timeout),
        ]
        for assignment in assignments
    ]


def _load_fixed_maps(
    args: argparse.Namespace,
    assignments: Sequence[ServerAssignment],
) -> None:
    if args.skip_map_load:
        print("FIXED_MAP_LOAD skipped=true responsibility=user", flush=True)
        return
    if not args.map_loader.is_file():
        raise FileNotFoundError(f"CARLA map loader not found: {args.map_loader}")
    processes = []
    for assignment, command in zip(assignments, _map_commands(args, assignments)):
        print(
            f"FIXED_MAP_LOAD_START town={assignment.town} "
            f"carla_port={assignment.carla_port}",
            flush=True,
        )
        processes.append((assignment, subprocess.Popen(command)))
    failures = []
    for assignment, process in processes:
        returncode = process.wait()
        print(
            f"FIXED_MAP_LOAD_END town={assignment.town} returncode={returncode}",
            flush=True,
        )
        if returncode != 0:
            failures.append(f"{assignment.town}:{returncode}")
    if failures:
        raise RuntimeError("fixed Town initialization failed: " + ", ".join(failures))
    if args.map_ready_sleep:
        time.sleep(args.map_ready_sleep)


def _worker_seed(master_seed: int, worker_id: int, update_index: int) -> int:
    return int(master_seed + worker_id * 1_000_003 + update_index * 10_007)


def _worker_spec(
    args: argparse.Namespace,
    *,
    assignment: ServerAssignment,
    phase: TownPhase,
    town_quota: int,
    total_steps: int,
    update_index: int,
    plan_fingerprint: str,
    frozen_model: Path,
    frozen_policy_sha256: str,
    update_dir: Path,
    log_root: Path,
) -> Dict[str, Any]:
    town_offset = sum(
        int(getattr(args, f"{town.lower()}_steps"))
        for town in TOWNS[: assignment.worker_id]
    )
    return {
        "contract": THREE_SERVER_CONTRACT,
        "plan_fingerprint": plan_fingerprint,
        "update_index": int(update_index),
        "assignment": assignment.__dict__,
        "scenarios": list(phase.scenarios),
        "settings": list(phase.settings),
        "episodes_per_origin": int(args.episodes_per_origin),
        "town_quota": int(town_quota),
        "total_steps": int(total_steps),
        "batch_size": int(args.batch_size),
        "ppo_epochs": int(args.ppo_epochs),
        "target_kl": None,
        "worker_seed": _worker_seed(args.seed, assignment.worker_id, update_index),
        "carla_host": args.carla_host,
        "no_rendering": not args.rendering,
        "live_log_every": int(args.live_log_every),
        "global_step_offset": int((update_index - 1) * total_steps + town_offset),
        "frozen_model": str(frozen_model.resolve()),
        "frozen_model_sha256": file_digest(frozen_model),
        "frozen_policy_sha256": frozen_policy_sha256,
        "buffer_path": str((update_dir / "fragments" / f"{phase.town}.npz").resolve()),
        "descriptor_path": str((update_dir / "fragments" / f"{phase.town}.json").resolve()),
        "worker_log_dir": str((log_root / "workers" / phase.town).resolve()),
    }


def _worker_command(spec_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "baseline.baseline_ours.train_three_servers_v2",
        "--worker-spec",
        str(spec_path),
    ]


def _valid_fragment(
    descriptor_path: Path,
    *,
    plan_fingerprint: str,
    update_index: int,
    assignment: ServerAssignment,
    frozen_policy_sha256: str,
) -> Dict[str, Any] | None:
    if not descriptor_path.is_file():
        return None
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        buffer_path = Path(str(descriptor["buffer_file"]))
        valid = (
            descriptor.get("contract") == THREE_SERVER_CONTRACT
            and descriptor.get("plan_fingerprint") == plan_fingerprint
            and int(descriptor.get("update_index", -1)) == update_index
            and descriptor.get("town") == assignment.town
            and int(descriptor.get("worker_id", -1)) == assignment.worker_id
            and descriptor.get("policy_sha256") == frozen_policy_sha256
            and buffer_path.is_file()
            and file_digest(buffer_path) == descriptor.get("sha256")
        )
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return descriptor if valid else None


def _wait_for_workers(
    processes: Sequence[tuple[ServerAssignment, Any]],
) -> Dict[str, int]:
    """Wait for every process even when an earlier worker has failed."""
    returncodes: Dict[str, int] = {}
    for assignment, process in processes:
        returncodes[assignment.town] = int(process.wait())
        print(
            f"ROLLOUT_PROCESS_END town={assignment.town} "
            f"returncode={returncodes[assignment.town]}",
            flush=True,
        )
    return returncodes


def _space_env() -> DummyVecEnv:
    return DummyVecEnv([_SpaceOnlyEnv])


def _load_parent_model(
    model_path: str | None,
    global_plan,
    args: argparse.Namespace,
):
    env = _space_env()
    model = build_or_load_ppo(
        env,
        global_plan,
        model_path=model_path,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        target_kl=args.target_kl,
    )
    return model, env


def _freeze_policy(
    source_model: str | None,
    target: Path,
    global_plan,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model, env = _load_parent_model(source_model, global_plan, args)
    try:
        saved = save_model_atomic(model, target)
        saved["file"] = str(target.with_suffix(".zip").resolve())
        saved["policy_sha256"] = policy_digest(model)
        saved["model_num_timesteps"] = int(model.num_timesteps)
        return saved
    finally:
        env.close()


def _import_worker_episodes(
    telemetry: TrainingTelemetry,
    descriptors: Iterable[Mapping[str, Any]],
) -> None:
    for descriptor in descriptors:
        for record in descriptor.get("episodes", []):
            telemetry.episode(record)


def _worker_main(spec_path: Path) -> int:
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("contract") != THREE_SERVER_CONTRACT:
            raise ValueError("worker specification contract mismatch")
        assignment = ServerAssignment(**spec["assignment"])
        phases = build_phase_plans(int(spec["episodes_per_origin"]))
        phase = next(item for item in phases if item.town == assignment.town)
        if list(phase.scenarios) != list(spec["scenarios"]):
            raise ValueError("worker scenario assignment changed")
        frozen_model = Path(spec["frozen_model"])
        if file_digest(frozen_model) != spec["frozen_model_sha256"]:
            raise ValueError("frozen model file changed before worker collection")
        for scenario in phase.scenarios:
            result = validate_scenario(scenario)
            if not result.ok:
                raise ValueError(f"{scenario} failed preflight: {'; '.join(result.errors)}")

        origins = {
            config.setting_id: tuple(config.origin.spawn_points)
            for config in phase.configs
        }
        selector = OrderedOriginSelector(
            phase.settings,
            {setting: len(origins[setting]) for setting in phase.settings},
            repeats=int(spec["episodes_per_origin"]),
        )
        binder = OriginBinder(selector, origins)
        initial = phase.configs[0]
        configure_initial_config(
            initial,
            carla_port=assignment.carla_port,
            tm_port=assignment.tm_port,
            sumo_port=assignment.sumo_port,
            no_rendering=bool(spec["no_rendering"]),
        )
        initial.carla.host = str(spec["carla_host"])

        def episode_setup(setting_id: str) -> None:
            binder.bind(setting_id)

        def make_env():
            from baseline.baseline_ours.road_arc_env_v2 import CarlaSumoGymEnv

            gym_env = CarlaSumoGymEnv(
                scenario=phase.settings[0],
                config=initial,
                worker_id=assignment.worker_id,
                no_rendering_mode=bool(spec["no_rendering"]),
                scenario_selector=selector,
                episode_setup_callback=episode_setup,
            )
            binder.attach(gym_env)
            return gym_env

        vec_env = DummyVecEnv([make_env])
        telemetry = TrainingTelemetry(
            Path(spec["worker_log_dir"]),
            resume=int(spec["update_index"]) > 1,
        )
        town_plan = _make_plan(
            phase,
            int(spec["episodes_per_origin"]),
            int(spec["town_quota"]),
        )
        # A worker owns one single-env Town fragment and always runs inference
        # on CPU.  Network weights are identical to the parent's frozen model;
        # only the unused SB3 rollout-buffer capacity is Town-sized here.
        model = build_or_load_ppo(
            vec_env,
            town_plan,
            model_path=str(frozen_model),
            ppo_epochs=int(spec["ppo_epochs"]),
            seed=int(spec["worker_seed"]),
            device="cpu",
            batch_size=int(spec["batch_size"]),
            target_kl=None,
        )
        model.set_random_seed(int(spec["worker_seed"]))
        frozen_digest = policy_digest(model)
        if frozen_digest != spec["frozen_policy_sha256"]:
            raise ValueError("worker loaded a different frozen policy")

        callback = RolloutTelemetryCallback(
            selector,
            telemetry,
            town=phase.town,
            target_steps=int(spec["town_quota"]),
            every_steps=int(spec["live_log_every"]),
            step_offset=int(spec["global_step_offset"]),
        )
        _, callback = model._setup_learn(
            int(spec["total_steps"]),
            callback,
            reset_num_timesteps=False,
            tb_log_name=f"three_servers_{phase.town}",
        )
        town_buffer = model.rollout_buffer
        print(
            f"WORKER_ROLLOUT_START town={phase.town} worker={assignment.worker_id} "
            f"carla={assignment.carla_port} tm={assignment.tm_port} "
            f"sumo={assignment.sumo_port} steps={town_buffer.buffer_size}",
            flush=True,
        )
        callback.on_training_start(locals(), globals())
        try:
            complete = collect_valid_rollouts(
                model,
                vec_env,
                callback,
                town_buffer,
                n_rollout_steps=town_buffer.buffer_size,
            )
        finally:
            callback.on_training_end()
        if not complete or not town_buffer.full:
            raise RuntimeError(f"incomplete frozen rollout for {phase.town}")
        if policy_digest(model) != frozen_digest:
            raise RuntimeError("policy changed inside a rollout worker")

        buffer_path = Path(spec["buffer_path"])
        fragment = save_rollout_buffer(buffer_path, town_buffer)
        context_rows = callback.context_summary()
        descriptor = {
            **fragment,
            "buffer_file": str(buffer_path.resolve()),
            "contract": THREE_SERVER_CONTRACT,
            "plan_fingerprint": spec["plan_fingerprint"],
            "update_index": int(spec["update_index"]),
            "town": phase.town,
            "worker_id": assignment.worker_id,
            "policy_sha256": frozen_digest,
            "coverage_complete": bool(selector.complete),
            "coverage": selector.completed_ticket_counts,
            "outcomes": _outcome_records(callback.outcomes),
            "episodes": callback.episode_records,
            "context_summary": context_rows,
            "valid_steps": town_buffer.buffer_size,
            "physical_env_steps": int(town_buffer.physical_env_steps),
            "infrastructure_faults": int(town_buffer.infrastructure_faults),
        }
        atomic_write_json(Path(spec["descriptor_path"]), descriptor)
        print_context_table(context_rows)
        print(
            f"WORKER_ROLLOUT_COMPLETE town={phase.town} "
            f"valid={town_buffer.buffer_size} physical={town_buffer.physical_env_steps} "
            f"infra={town_buffer.infrastructure_faults} coverage={selector.complete}",
            flush=True,
        )
        return 0
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"WORKER_ERROR spec={spec_path} error={error}", flush=True)
        return 3
    finally:
        local_vec_env = locals().get("vec_env")
        if local_vec_env is not None:
            local_vec_env.close()


def _reward_summary(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    returns = [float(record["episode_return"]) for record in episodes]
    return {
        "count": len(returns),
        "mean": float(np.mean(returns)) if returns else None,
        "std": float(np.std(returns)) if returns else None,
        "min": min(returns) if returns else None,
        "max": max(returns) if returns else None,
    }


_SAFETY_FAILURE_REASONS = {
    "collision_jaywalker",
    "collision_sumo_vehicle",
    "workzone_violation",
    "off_road",
}


def _policy_selection_metrics(
    episodes: Sequence[Mapping[str, Any]],
    expected_units: Sequence[tuple[str, int]],
) -> Dict[str, float | int]:
    """Score one frozen policy without letting long scenarios dominate."""
    expected = tuple((str(setting), int(origin)) for setting, origin in expected_units)
    if not expected:
        raise ValueError("policy selection requires expected setting/origin units")
    grouped: Dict[tuple[str, int], list[Mapping[str, Any]]] = {
        unit: [] for unit in expected
    }
    for record in episodes:
        try:
            unit = (str(record["setting_id"]), int(record["origin_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        if unit in grouped:
            grouped[unit].append(record)

    covered = sum(bool(records) for records in grouped.values())
    success_rates = []
    safety_failure_rates = []
    for records in grouped.values():
        if not records:
            success_rates.append(0.0)
            safety_failure_rates.append(1.0)
            continue
        count = len(records)
        success_rates.append(
            sum(int(bool(record.get("success", False))) for record in records) / count
        )
        safety_failure_rates.append(
            sum(
                str(record.get("reason", "")) in _SAFETY_FAILURE_REASONS
                for record in records
            )
            / count
        )
    returns = [float(record["episode_return"]) for record in episodes]
    return {
        "expected_units": len(expected),
        "covered_units": covered,
        "coverage_fraction": covered / len(expected),
        "macro_success_rate": sum(success_rates) / len(success_rates),
        "macro_safety_failure_rate": (
            sum(safety_failure_rates) / len(safety_failure_rates)
        ),
        "mean_episode_return": (
            sum(returns) / len(returns) if returns else -1.0e30
        ),
    }


def _policy_selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["coverage_fraction"]),
        float(metrics["macro_success_rate"]),
        -float(metrics["macro_safety_failure_rate"]),
        float(metrics["mean_episode_return"]),
    )


def _atomic_policy_alias(source: Path, target: Path) -> Dict[str, str]:
    """Copy one complete SB3 zip to a stable best/last alias atomically."""
    source = Path(source)
    target = Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    if file_digest(source) != file_digest(target):
        raise RuntimeError(f"policy alias verification failed: {target}")
    return {"file": str(target.resolve()), "sha256": file_digest(target)}


def _run_parent(args: argparse.Namespace) -> int:
    assignments = _assignments(args)
    for _, scenarios in ((phase.town, phase.scenarios) for phase in build_phase_plans(1)):
        for scenario in scenarios:
            result = validate_scenario(scenario)
            if not result.ok:
                for error in result.errors:
                    print(f"ERROR {scenario}: {error}", flush=True)
                return 2
    phases = build_phase_plans(args.episodes_per_origin)
    town_quotas = _town_quotas(args, phases)
    global_plan = _global_plan(phases, args.episodes_per_origin, town_quotas)
    plan = _training_plan(args, phases, town_quotas, assignments)
    plan_fingerprint = _plan_fingerprint(plan)

    run_root = args.run_root.resolve()
    state_path = run_root / "run_state.json"
    if args.resume:
        try:
            state = _load_state(state_path, plan)
        except (OSError, TypeError, ValueError) as error:
            print(f"ERROR cannot resume: {error}", flush=True)
            return 2
    else:
        if state_path.exists():
            print(
                f"ERROR run already exists: {state_path}; use --resume or a new --run-root",
                flush=True,
            )
            return 2
        state = _new_state(plan)
        atomic_write_json(state_path, state)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("ERROR --device cuda requested, but CUDA is unavailable", flush=True)
            return 2

    telemetry = TrainingTelemetry(run_root / "logs", resume=args.resume)
    TrainingTelemetry.banner(
        "PPO V2 — THREE FIXED TOWN SERVERS",
        f"steps/update={global_plan.buffer_size_steps:,} | barrier=all-3 | "
        f"rollout=CPU | PPO={args.device}",
    )
    for assignment in assignments:
        phase = next(item for item in phases if item.town == assignment.town)
        print(
            f"SERVER worker={assignment.worker_id} town={assignment.town} "
            f"scenarios={'->'.join(phase.scenarios)} carla={assignment.carla_port} "
            f"tm={assignment.tm_port} sumo={assignment.sumo_port}",
            flush=True,
        )
    try:
        _load_fixed_maps(args, assignments)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR {error}", flush=True)
        return 2

    phase_by_town = {phase.town: phase for phase in phases}
    expected_units = tuple(
        (setting, origin_index)
        for phase in phases
        for setting in phase.settings
        for origin_index in range(3)
    )
    total_steps = global_plan.buffer_size_steps
    workspace_root = Path(__file__).resolve().parents[2]
    for update_index in range(int(state["next_update"]), args.total_updates + 1):
        update_dir = run_root / "checkpoints" / f"update_{update_index:03d}"
        update_dir.mkdir(parents=True, exist_ok=True)
        frozen_path = update_dir / "frozen_policy.zip"
        active = state.get("active_update")
        if not active or int(active.get("index", -1)) != update_index:
            previous_model = state.get("last_model")
            if previous_model is None and plan.get("initial_model") is not None:
                previous_model = plan["initial_model"]["file"]
            frozen_info = _freeze_policy(
                previous_model,
                frozen_path,
                global_plan,
                args,
            )
            active = {
                "index": update_index,
                "state": "collecting",
                "frozen_model": frozen_info,
                "episodes_imported": False,
            }
            state["active_update"] = active
            state["status"] = "collecting"
            atomic_write_json(state_path, state)
        else:
            frozen_info = dict(active["frozen_model"])
            if (
                not frozen_path.is_file()
                or file_digest(frozen_path) != frozen_info.get("sha256")
            ):
                print("ERROR frozen policy checkpoint is missing or corrupt", flush=True)
                return 2

        frozen_policy_sha256 = str(frozen_info["policy_sha256"])
        processes = []
        descriptors: Dict[str, Dict[str, Any]] = {}
        for assignment in assignments:
            phase = phase_by_town[assignment.town]
            descriptor_path = update_dir / "fragments" / f"{assignment.town}.json"
            descriptor = _valid_fragment(
                descriptor_path,
                plan_fingerprint=plan_fingerprint,
                update_index=update_index,
                assignment=assignment,
                frozen_policy_sha256=frozen_policy_sha256,
            )
            if descriptor is not None:
                descriptors[assignment.town] = descriptor
                print(f"ROLLOUT_RESUME_REUSE town={assignment.town}", flush=True)
                continue
            spec = _worker_spec(
                args,
                assignment=assignment,
                phase=phase,
                town_quota=town_quotas[assignment.town],
                total_steps=total_steps,
                update_index=update_index,
                plan_fingerprint=plan_fingerprint,
                frozen_model=frozen_path,
                frozen_policy_sha256=frozen_policy_sha256,
                update_dir=update_dir,
                log_root=run_root / "logs",
            )
            spec_path = update_dir / "worker_specs" / f"{assignment.town}.json"
            atomic_write_json(spec_path, spec)
            command = _worker_command(spec_path)
            print(
                f"ROLLOUT_PROCESS_START town={assignment.town} "
                f"steps={town_quotas[assignment.town]:,}",
                flush=True,
            )
            processes.append(
                (assignment, subprocess.Popen(command, cwd=str(workspace_root)))
            )

        print(
            f"ROLLOUT_BARRIER_WAIT update={update_index} "
            f"pending={len(processes)} required=3",
            flush=True,
        )
        returncodes = _wait_for_workers(processes)
        failed = [town for town, code in returncodes.items() if code != 0]
        for assignment in assignments:
            descriptor_path = update_dir / "fragments" / f"{assignment.town}.json"
            descriptor = _valid_fragment(
                descriptor_path,
                plan_fingerprint=plan_fingerprint,
                update_index=update_index,
                assignment=assignment,
                frozen_policy_sha256=frozen_policy_sha256,
            )
            if descriptor is None:
                failed.append(assignment.town)
            else:
                descriptors[assignment.town] = descriptor
        if failed:
            state["status"] = "paused"
            state["next_update"] = update_index
            atomic_write_json(state_path, state)
            print(
                f"TRAINING_PAUSED update={update_index} failed={sorted(set(failed))} "
                f"resume_with=--resume",
                flush=True,
            )
            return 3

        ordered_descriptors = [descriptors[town] for town in TOWNS]
        if not active.get("episodes_imported", False):
            _import_worker_episodes(telemetry, ordered_descriptors)
            active["episodes_imported"] = True
            active["state"] = "ready_to_update"
            state["active_update"] = active
            atomic_write_json(state_path, state)

        model, parent_env = _load_parent_model(str(frozen_path), global_plan, args)
        try:
            if policy_digest(model) != frozen_policy_sha256:
                raise RuntimeError("parent loaded a different frozen policy")
            buffers = [
                load_rollout_buffer(
                    Path(str(descriptor["buffer_file"])),
                    descriptor,
                    model,
                )
                for descriptor in ordered_descriptors
            ]
            merged = merge_rollout_buffers(buffers)
            if merged.buffer_size != total_steps:
                raise RuntimeError(
                    f"merged buffer has {merged.buffer_size} steps, expected {total_steps}"
                )
            model.rollout_buffer = merged
            model.n_steps = merged.buffer_size
            model.batch_size = args.batch_size
            model.num_timesteps = int(frozen_info["model_num_timesteps"]) + total_steps
            model._total_timesteps = model.num_timesteps
            model._update_current_progress_remaining(
                model.num_timesteps,
                model._total_timesteps,
            )
            model.set_logger(
                configure_logger(
                    model.verbose,
                    model.tensorboard_log,
                    f"three_servers_update_{update_index:03d}",
                    False,
                )
            )
            print(
                f"ROLLOUT_BARRIER_RELEASE update={update_index} "
                f"fragments=3 steps={merged.buffer_size:,}",
                flush=True,
            )
            TrainingTelemetry.banner(
                f"PPO UPDATE {update_index}/{args.total_updates}",
                f"samples={merged.buffer_size:,} | epochs={model.n_epochs} | "
                f"device={model.device}",
            )
            updates_before = model._n_updates
            train_summary = train_ppo_with_telemetry(model, telemetry)
            model.logger.dump(step=model.num_timesteps)
            output = run_root / "models" / f"policy_update_{update_index:03d}"
            model_info = save_model_atomic(model, output)
            model_path = output.with_suffix(".zip").resolve()
            model_info.update(
                update=update_index,
                file=str(model_path),
                policy_sha256=policy_digest(model),
                model_num_timesteps=int(model.num_timesteps),
                ppo_epoch_updates=int(model._n_updates - updates_before),
            )
            last_alias = _atomic_policy_alias(
                model_path,
                run_root / "models" / "last_model.zip",
            )
        finally:
            parent_env.close()

        all_episodes = [
            record
            for descriptor in ordered_descriptors
            for record in descriptor.get("episodes", [])
        ]
        all_outcomes: Counter[tuple[str, str]] = Counter()
        for descriptor in ordered_descriptors:
            for record in descriptor.get("outcomes", []):
                all_outcomes[(str(record["setting_id"]), str(record["reason"]))] += int(
                    record["count"]
                )
        selection_metrics = _policy_selection_metrics(all_episodes, expected_units)
        previous_best = state.get("best_model")
        best_updated = (
            previous_best is None
            or _policy_selection_key(selection_metrics)
            > _policy_selection_key(previous_best["metrics"])
        )
        if best_updated:
            best_alias = _atomic_policy_alias(
                frozen_path,
                run_root / "models" / "best_model.zip",
            )
            best_model = {
                **best_alias,
                "evaluated_during_update": update_index,
                "source_policy_sha256": frozen_policy_sha256,
                "source_model_num_timesteps": int(
                    frozen_info["model_num_timesteps"]
                ),
                "metrics": selection_metrics,
            }
            telemetry.event("best_policy_updated", **best_model)
        else:
            best_model = dict(previous_best)
        summary = {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "update": update_index,
            "output": str(model_path),
            "total_steps": total_steps,
            "aggregate_environment_steps": int(model_info["model_num_timesteps"]),
            "ppo_device": str(args.device),
            "ppo_gamma": PPO_GAMMA,
            "ppo_gae_lambda": PPO_GAE_LAMBDA,
            "ppo_epochs": args.ppo_epochs,
            "batch_size": args.batch_size,
            "barrier": "all_three_towns",
            "town_valid_steps": {
                descriptor["town"]: int(descriptor["valid_steps"])
                for descriptor in ordered_descriptors
            },
            "town_physical_steps": {
                descriptor["town"]: int(descriptor["physical_env_steps"])
                for descriptor in ordered_descriptors
            },
            "town_infrastructure_faults": {
                descriptor["town"]: int(descriptor["infrastructure_faults"])
                for descriptor in ordered_descriptors
            },
            "reward_distribution": _reward_summary(all_episodes),
            "evaluated_policy_metrics": selection_metrics,
            "best_model": best_model,
            "training": train_summary,
            "outcomes": _outcome_records(all_outcomes),
        }
        telemetry.write_summary(summary)
        telemetry.event("three_server_update_complete", **summary)
        completed = [
            item for item in state.get("completed_updates", [])
            if int(item["update"]) != update_index
        ]
        completed.append(model_info)
        completed.sort(key=lambda item: int(item["update"]))
        state.update(
            status="running",
            next_update=update_index + 1,
            last_model=str(model_path),
            last_model_alias=last_alias,
            best_model=best_model,
            active_update=None,
            completed_updates=completed,
        )
        atomic_write_json(state_path, state)
        print(
            f"THREE_SERVER_UPDATE_COMPLETE update={update_index}/{args.total_updates} "
            f"model={model_path} best_updated={best_updated}",
            flush=True,
        )

    state["status"] = "complete"
    atomic_write_json(state_path, state)
    print(
        f"THREE_SERVER_TRAINING_COMPLETE updates={args.total_updates} "
        f"last={state['last_model']} best={state['best_model']['file']}",
        flush=True,
    )
    return 0


def main(argv: List[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--worker-spec"]:
        if len(arguments) != 2:
            print("ERROR --worker-spec requires exactly one path", flush=True)
            return 2
        return _worker_main(Path(arguments[1]))
    parser = build_parser()
    args = parser.parse_args(arguments)
    _validate_args(parser, args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
