"""Compact Aug-24 orchestration: rollout -> PPO -> validation -> repeat."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .bootstrap import portable_subprocess_environment
from .experiment_config import (
    BATCH_SIZE,
    DEFAULT_RUN_ROOT,
    MAX_EPOCHS_PER_UPDATE,
    MAP_LOADER,
    TOTAL_UPDATES,
    TOWNS,
)
from .validate import validate_policy_after_update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--total-updates", type=int, default=TOTAL_UPDATES)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--rendering", action="store_true")
    parser.add_argument(
        "--skip-first-map-load",
        action="store_true",
        help="Use only when all three CARLA servers are already on their fixed Towns.",
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser


def _read_state(run_root: Path) -> dict:
    path = run_root / "run_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _policy_path(run_root: Path, update: int) -> Path:
    return run_root / "models" / f"policy_update_{update:03d}.zip"


def _training_command(
    args: argparse.Namespace,
    *,
    target_update: int,
    resume: bool,
    skip_map_load: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "baseline.PPO.train_three_servers_v2",
        "--run-root",
        str(args.run_root),
        "--total-updates",
        str(target_update),
        "--ppo-epochs",
        str(MAX_EPOCHS_PER_UPDATE),
        "--batch-size",
        str(BATCH_SIZE),
        "--live-log-every",
        "5000",
        "--map-loader",
        str(MAP_LOADER),
        "--carla-ports",
        *(str(town.carla_port) for town in TOWNS),
        "--tm-ports",
        *(str(town.tm_port) for town in TOWNS),
        "--sumo-ports",
        *(str(town.sumo_port) for town in TOWNS),
        "--town02-steps",
        str(TOWNS[0].rollout_steps),
        "--town05-steps",
        str(TOWNS[1].rollout_steps),
        "--town10hd-steps",
        str(TOWNS[2].rollout_steps),
    ]
    if resume:
        command.append("--resume")
    if args.initial_model is not None:
        command.extend(("--initial-model", str(args.initial_model)))
    if skip_map_load:
        command.append("--skip-map-load")
    if args.rendering:
        command.append("--rendering")
    return command


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_root = args.run_root.resolve()
    if args.total_updates < 1:
        raise ValueError("--total-updates must be positive")
    state = _read_state(args.run_root)
    next_update = int(state.get("next_update", 1))
    if next_update > args.total_updates:
        print("AUG24_TRAINING_ALREADY_COMPLETE", flush=True)
        return 0

    # If training finished an update but validation was interrupted, complete
    # that frozen evaluation before collecting any newer policy data.
    if next_update > 1 and not args.skip_validation:
        prior = next_update - 1
        validation_summary = (
            args.run_root / "validation" / f"update_{prior:03d}" / "summary.json"
        )
        if not validation_summary.is_file():
            validate_policy_after_update(
                _policy_path(args.run_root, prior),
                update=prior,
                run_root=args.run_root,
            )

    for update in range(next_update, args.total_updates + 1):
        resume = (args.run_root / "run_state.json").is_file()
        command = _training_command(
            args,
            target_update=update,
            resume=resume,
            skip_map_load=(args.skip_first_map_load or update > next_update),
        )
        print(
            f"AUG24_UPDATE_START update={update}/{args.total_updates} "
            f"epochs={MAX_EPOCHS_PER_UPDATE}",
            flush=True,
        )
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(MAP_LOADER.parent),
            env=portable_subprocess_environment(),
        )
        if completed.returncode != 0:
            return int(completed.returncode)
        policy = _policy_path(args.run_root, update)
        if not args.skip_validation:
            validate_policy_after_update(
                policy,
                update=update,
                run_root=args.run_root,
            )
        print(f"AUG24_UPDATE_END update={update}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
