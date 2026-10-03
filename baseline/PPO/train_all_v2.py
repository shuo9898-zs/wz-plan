"""Run the frozen PPO V2 design for multiple resumable global updates.

Each update delegates one unchanged three-Town rollout/update transaction to
``validate_allSwithoneExe``.  Town fragments keep their existing strict
checkpoint semantics, while this module adds an outer update checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from baseline.PPO.global_rollout_checkpoint import (
    atomic_write_json,
    canonical_digest,
    file_digest,
)
from baseline.PPO.encoder_v2 import (
    ENCODER_ATTENTION_HEADS_V2,
    ENCODER_BRANCH_HIDDEN_DIM_V2,
    ENCODER_CONTRACT_VERSION_V2,
    ENCODER_FEATURES_DIM_V2,
    ENCODER_TOKEN_DIM_V2,
)
from baseline.PPO.validate_allSwithoneExe import DEFAULT_MAP_LOADER
from baseline.PPO.runtime_v2 import (
    V2_DEFAULT_PPO_EPOCHS,
    V2_POLICY_ACTIVATION,
    V2_POLICY_HIDDEN_SIZES,
    V2_POLICY_TRAINABLE_PARAMETERS,
)
from baseline.PPO.training_config_v2 import DEFAULT_THREE_SERVER_TRAINING_V2
from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_DIM_V2,
    OBSERVATION_CONTRACT_VERSION_V2,
)
from logic.reward_v2 import REWARD_CONTRACT_VERSION_V2
from logic.termination_checker_v2 import TERMINATION_CONTRACT_VERSION_V2
from baseline.controllers_v2 import CONTROLLER_CONTRACT_VERSION_V2


SCHEMA_VERSION = 1
_TRAINING_DEFAULTS = DEFAULT_THREE_SERVER_TRAINING_V2
DEFAULT_TOWN_STEPS = _TRAINING_DEFAULTS.town_step_quotas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/ppo_v2_final"))
    parser.add_argument("--total-updates", type=int, default=_TRAINING_DEFAULTS.total_updates)
    parser.add_argument(
        "--episodes-per-origin", type=int, default=_TRAINING_DEFAULTS.episodes_per_origin
    )
    parser.add_argument("--town02-steps", type=int, default=DEFAULT_TOWN_STEPS["Town02"])
    parser.add_argument("--town05-steps", type=int, default=DEFAULT_TOWN_STEPS["Town05"])
    parser.add_argument("--town10hd-steps", type=int, default=DEFAULT_TOWN_STEPS["Town10HD"])
    parser.add_argument("--batch-size", type=int, default=_TRAINING_DEFAULTS.ppo.batch_size)
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=V2_DEFAULT_PPO_EPOCHS,
        help="maximum PPO passes over each fresh rollout buffer (default: 50)",
    )
    parser.add_argument(
        "--target-kl", type=float, default=_TRAINING_DEFAULTS.ppo.target_kl
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=_TRAINING_DEFAULTS.device
    )
    parser.add_argument("--seed", type=int, default=_TRAINING_DEFAULTS.seed)
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--live-log-every", type=int, default=_TRAINING_DEFAULTS.live_log_every_steps
    )
    parser.add_argument("--carla-host", default=_TRAINING_DEFAULTS.carla_host)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--sumo-port", type=int, default=8813)
    parser.add_argument("--map-loader", type=Path, default=DEFAULT_MAP_LOADER)
    parser.add_argument("--map-rpc-timeout", type=float, default=60.0)
    parser.add_argument("--teardown-sleep", type=float, default=3.0)
    parser.add_argument("--map-ready-sleep", type=float, default=30.0)
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Enable CARLA rendering (the frozen training default is no-rendering)",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive_ints = {
        "--total-updates": args.total_updates,
        "--episodes-per-origin": args.episodes_per_origin,
        "--town02-steps": args.town02_steps,
        "--town05-steps": args.town05_steps,
        "--town10hd-steps": args.town10hd_steps,
        "--batch-size": args.batch_size,
        "--ppo-epochs": args.ppo_epochs,
        "--live-log-every": args.live_log_every,
    }
    for name, value in positive_ints.items():
        if value < 1:
            parser.error(f"{name} must be positive")
    if args.target_kl is not None and (
        not math.isfinite(args.target_kl) or args.target_kl <= 0.0
    ):
        parser.error("--target-kl must be finite and positive")
    if args.map_rpc_timeout <= 0.0:
        parser.error("--map-rpc-timeout must be positive")
    if args.teardown_sleep < 0.0 or args.map_ready_sleep < 0.0:
        parser.error("sleep durations must be non-negative")
    if args.initial_model is not None and not args.initial_model.with_suffix(".zip").is_file():
        parser.error(f"--initial-model not found: {args.initial_model.with_suffix('.zip')}")


def _plan(args: argparse.Namespace) -> Dict[str, Any]:
    initial_model = None
    if args.initial_model is not None:
        model_path = args.initial_model.with_suffix(".zip").resolve()
        initial_model = {
            "file": str(model_path),
            "sha256": file_digest(model_path),
        }
    workspace_root = Path(__file__).resolve().parents[2]
    frozen_sources = (
        workspace_root / "env" / "observation_encoder_v2.py",
        workspace_root / "env" / "carla_sumo_env_v2.py",
        workspace_root / "env" / "gym_wrapper_v2.py",
        workspace_root / "logic" / "reward_v2.py",
        workspace_root / "logic" / "termination_checker_v2.py",
        workspace_root / "baseline" / "controllers_v2.py",
        workspace_root / "baseline" / "PPO" / "runtime_v2.py",
        workspace_root / "baseline" / "PPO" / "training_config_v2.py",
        workspace_root / "baseline" / "PPO" / "encoder_v2.py",
        workspace_root / "baseline" / "PPO" / "valid_rollout.py",
        workspace_root / "baseline" / "PPO" / "validate_allSwithoneExe.py",
    )
    return {
        "contract": "ppo_v2_repeated_training_v1",
        "total_updates": int(args.total_updates),
        "episodes_per_origin": int(args.episodes_per_origin),
        "town_steps": {
            "Town02": int(args.town02_steps),
            "Town05": int(args.town05_steps),
            "Town10HD": int(args.town10hd_steps),
        },
        "steps_per_update": int(
            args.town02_steps + args.town05_steps + args.town10hd_steps
        ),
        "batch_size": int(args.batch_size),
        "ppo_epochs": int(args.ppo_epochs),
        "target_kl": None if args.target_kl is None else float(args.target_kl),
        "gamma": _TRAINING_DEFAULTS.ppo.gamma,
        "gae_lambda": _TRAINING_DEFAULTS.ppo.gae_lambda,
        "learning_rate": _TRAINING_DEFAULTS.ppo.learning_rate,
        "clip_range": _TRAINING_DEFAULTS.ppo.clip_range,
        "clip_range_vf": _TRAINING_DEFAULTS.ppo.clip_range_vf,
        "normalize_advantage": _TRAINING_DEFAULTS.ppo.normalize_advantage,
        "entropy_coefficient": _TRAINING_DEFAULTS.ppo.entropy_coefficient,
        "value_function_coefficient": (
            _TRAINING_DEFAULTS.ppo.value_function_coefficient
        ),
        "max_gradient_norm": _TRAINING_DEFAULTS.ppo.max_gradient_norm,
        "seed": int(args.seed),
        "observation_contract": OBSERVATION_CONTRACT_VERSION_V2,
        "observation_dim": DEFAULT_OBSERVATION_DIM_V2,
        "policy_architecture": {
            "encoder_contract": ENCODER_CONTRACT_VERSION_V2,
            "encoder_token_dim": ENCODER_TOKEN_DIM_V2,
            "encoder_branch_hidden_dim": ENCODER_BRANCH_HIDDEN_DIM_V2,
            "encoder_features_dim": ENCODER_FEATURES_DIM_V2,
            "encoder_attention_heads": ENCODER_ATTENTION_HEADS_V2,
            "actor_critic_share_encoder": False,
            "actor_hidden": list(V2_POLICY_HIDDEN_SIZES),
            "critic_hidden": list(V2_POLICY_HIDDEN_SIZES),
            "activation": V2_POLICY_ACTIVATION,
            "continuous_actions": 2,
            "trainable_parameters": V2_POLICY_TRAINABLE_PARAMETERS,
        },
        "action_contract": CONTROLLER_CONTRACT_VERSION_V2,
        "reward_contract": REWARD_CONTRACT_VERSION_V2,
        "termination_contract": TERMINATION_CONTRACT_VERSION_V2,
        "initial_model": initial_model,
        "frozen_source_sha256": {
            str(path.relative_to(workspace_root)): file_digest(path)
            for path in frozen_sources
        },
    }


def _new_state(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "plan": plan,
        "plan_fingerprint": _plan_fingerprint(plan),
        "next_update": 1,
        "last_model": None,
        "completed_updates": [],
    }


def _plan_fingerprint(plan: Dict[str, Any]) -> str:
    immutable_plan = dict(plan)
    immutable_plan.pop("total_updates", None)
    return canonical_digest(immutable_plan)


def _load_state(path: Path, plan: Dict[str, Any]) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"resume state not found: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported repeated-training checkpoint schema")
    if state.get("plan_fingerprint") != _plan_fingerprint(plan):
        raise ValueError(
            "training parameters differ from the saved run; start a new --run-root "
            "from the last completed model instead of reusing rollout buffers"
        )
    return state


def _completed_child_state(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    return state if state.get("state") == "iteration_complete" else None


def _child_command(
    args: argparse.Namespace,
    *,
    update_index: int,
    output: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    model_path: str | None,
    resume_iteration: bool,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "baseline.PPO.validate_allSwithoneExe",
        "--episodes-per-origin", str(args.episodes_per_origin),
        "--town02-steps", str(args.town02_steps),
        "--town05-steps", str(args.town05_steps),
        "--town10hd-steps", str(args.town10hd_steps),
        "--estimated-coverage",
        "--batch-size", str(args.batch_size),
        "--ppo-epochs", str(args.ppo_epochs),
        "--device", args.device,
        "--seed", str(args.seed),
        "--live-log-every", str(args.live_log_every),
        "--carla-host", args.carla_host,
        "--carla-port", str(args.carla_port),
        "--tm-port", str(args.tm_port),
        "--sumo-port", str(args.sumo_port),
        "--map-loader", str(args.map_loader),
        "--map-rpc-timeout", str(args.map_rpc_timeout),
        "--teardown-sleep", str(args.teardown_sleep),
        "--map-ready-sleep", str(args.map_ready_sleep),
        "--output", str(output),
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-dir", str(log_dir),
        "--global-step-offset", str(
            (update_index - 1)
            * (args.town02_steps + args.town05_steps + args.town10hd_steps)
        ),
    ]
    if args.target_kl is not None:
        command.extend(("--target-kl", str(args.target_kl)))
    if not args.rendering:
        command.append("--no-rendering")
    if update_index > 1:
        command.append("--append-logs")
    if resume_iteration:
        command.append("--resume")
    elif model_path is not None:
        command.extend(("--model", model_path))
    return command


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    run_root = args.run_root.resolve()
    state_path = run_root / "run_state.json"
    plan = _plan(args)
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

    log_dir = run_root / "logs"
    total_steps = int(plan["steps_per_update"])
    for update_index in range(int(state["next_update"]), args.total_updates + 1):
        update_name = f"update_{update_index:03d}"
        checkpoint_dir = run_root / "checkpoints" / update_name
        checkpoint_state_path = checkpoint_dir / "state.json"
        output = run_root / "models" / f"policy_{update_name}"

        completed_child = _completed_child_state(checkpoint_state_path)
        if completed_child is not None:
            model_path = str(completed_child["final_model"]["file"])
            child_returncode = 0
        else:
            resume_iteration = checkpoint_state_path.is_file()
            previous_model = state.get("last_model")
            if previous_model is None and plan.get("initial_model") is not None:
                previous_model = plan["initial_model"]["file"]
            command = _child_command(
                args,
                update_index=update_index,
                output=output,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                model_path=None if resume_iteration else previous_model,
                resume_iteration=resume_iteration,
            )
            print(
                f"V2_UPDATE_START {update_index}/{args.total_updates} "
                f"steps={total_steps:,} resume={resume_iteration}",
                flush=True,
            )
            child_returncode = subprocess.run(command, check=False).returncode
            completed_child = _completed_child_state(checkpoint_state_path)
            model_path = (
                str(completed_child["final_model"]["file"])
                if completed_child is not None
                else None
            )

        if child_returncode != 0 or completed_child is None or model_path is None:
            state["status"] = "paused"
            state["next_update"] = update_index
            atomic_write_json(state_path, state)
            print(
                f"V2_TRAINING_PAUSED update={update_index} returncode={child_returncode} "
                f"resume_with=--resume",
                flush=True,
            )
            return child_returncode or 3

        model_file = Path(model_path)
        model_info = {
            "update": update_index,
            "file": str(model_file.resolve()),
            "sha256": file_digest(model_file),
        }
        completed_updates = [
            item for item in state.get("completed_updates", [])
            if int(item["update"]) != update_index
        ]
        completed_updates.append(model_info)
        completed_updates.sort(key=lambda item: int(item["update"]))
        state.update(
            status="running",
            next_update=update_index + 1,
            last_model=model_info["file"],
            completed_updates=completed_updates,
        )
        atomic_write_json(state_path, state)
        print(
            f"V2_UPDATE_COMPLETE {update_index}/{args.total_updates} model={model_path}",
            flush=True,
        )

    state["status"] = "complete"
    atomic_write_json(state_path, state)
    print(
        f"V2_TRAINING_COMPLETE updates={args.total_updates} "
        f"total_steps={args.total_updates * total_steps:,} "
        f"model={state['last_model']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
