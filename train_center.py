"""Centre-ego control run: rollout -> PPO -> matched validation -> repeat."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from baseline.PPO.global_rollout_checkpoint import file_digest
from baseline.PPO.train_three_servers_center_v2 import (
    CENTER_THREE_SERVER_CONTRACT_V2,
)
from . import train as swept_train
from .bootstrap import BUNDLE_ROOT, portable_subprocess_environment
from .experiment_config import MAX_EPOCHS_PER_UPDATE, VALIDATION_TOTAL_EPISODES
from .validate import validate_policy_after_update
from logic.episode_termination_center_v2 import (
    CENTER_EGO_JUDGEMENT_V2,
    CENTER_TERMINATION_CONTRACT_VERSION_V2,
)
from logic.reward_v2 import REWARD_CONTRACT_VERSION_V2


CENTER_DEFAULT_RUN_ROOT = BUNDLE_ROOT / "runs_center_ego"
CENTER_TOTAL_UPDATES = 35
CENTER_TRAINING_MODULE = "baseline.PPO.train_three_servers_center_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = swept_train.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        run_root=CENTER_DEFAULT_RUN_ROOT,
        total_updates=CENTER_TOTAL_UPDATES,
    )
    return parser


def _training_command(
    args: argparse.Namespace,
    *,
    target_update: int,
    resume: bool,
    skip_map_load: bool,
) -> list[str]:
    command = swept_train._training_command(
        args,
        target_update=target_update,
        resume=resume,
        skip_map_load=skip_map_load,
    )
    command[2] = CENTER_TRAINING_MODULE
    return command


def _validate(policy: Path, *, update: int, run_root: Path) -> None:
    _assert_center_run_state(swept_train._read_state(run_root))
    validate_policy_after_update(
        policy,
        update=update,
        run_root=run_root,
        ego_judgement=CENTER_EGO_JUDGEMENT_V2,
    )


def _assert_center_run_state(state: dict) -> None:
    """Reject an accidental resume of the swept-OBB experiment."""
    if not state:
        return
    plan = state.get("plan") or {}
    rollout = plan.get("rollout_and_ppo") or {}
    if plan.get("contract") != CENTER_THREE_SERVER_CONTRACT_V2:
        raise RuntimeError(
            "The selected --run-root is not a center-ego experiment; "
            "use a new empty directory."
        )
    if plan.get("ego_judgement") != CENTER_EGO_JUDGEMENT_V2:
        raise RuntimeError("center-ego run_state judgement contract mismatch")
    if rollout.get("reward_contract") != REWARD_CONTRACT_VERSION_V2:
        raise RuntimeError("center-ego run_state reward contract mismatch")
    if (
        rollout.get("termination_contract")
        != CENTER_TERMINATION_CONTRACT_VERSION_V2
    ):
        raise RuntimeError("center-ego run_state termination contract mismatch")

    frozen = {
        str(name).replace("\\", "/"): str(digest)
        for name, digest in (plan.get("frozen_source_sha256") or {}).items()
    }
    for relative_name, expected_hash in frozen.items():
        source = BUNDLE_ROOT / Path(relative_name)
        if not source.is_file() or file_digest(source) != expected_hash:
            raise RuntimeError(
                f"{relative_name} changed after this center-ego run started"
            )


def _center_validation_is_complete(
    summary_path: Path,
    *,
    policy: Path,
    update: int,
) -> bool:
    """Accept only a complete validation of this exact center policy."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        reward_hash = file_digest(BUNDLE_ROOT / "logic" / "reward_v2.py")
        return (
            int(summary["update"]) == int(update)
            and Path(str(summary["policy"])).resolve() == policy.resolve()
            and summary["policy_sha256"] == file_digest(policy)
            and summary["ego_judgement"] == CENTER_EGO_JUDGEMENT_V2
            and summary["reward_contract"] == REWARD_CONTRACT_VERSION_V2
            and summary["reward_source_sha256"] == reward_hash
            and summary["termination_contract"]
            == CENTER_TERMINATION_CONTRACT_VERSION_V2
            and metrics["ego_judgement"] == CENTER_EGO_JUDGEMENT_V2
            and metrics["reward_contract"] == REWARD_CONTRACT_VERSION_V2
            and metrics["reward_source_sha256"] == reward_hash
            and metrics["termination_contract"]
            == CENTER_TERMINATION_CONTRACT_VERSION_V2
            and metrics["model_sha256"] == file_digest(policy)
            and int(metrics["episodes"]) == VALIDATION_TOTAL_EPISODES
            and int(metrics["scenario_count"]) == 6
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_root = args.run_root.resolve()
    if args.total_updates < 1:
        raise ValueError("--total-updates must be positive")
    state = swept_train._read_state(args.run_root)
    _assert_center_run_state(state)
    next_update = int(state.get("next_update", 1))

    if next_update > 1 and not args.skip_validation:
        prior = next_update - 1
        prior_policy = swept_train._policy_path(args.run_root, prior)
        validation_summary = (
            args.run_root / "validation" / f"update_{prior:03d}" / "summary.json"
        )
        if not _center_validation_is_complete(
            validation_summary,
            policy=prior_policy,
            update=prior,
        ):
            _validate(
                prior_policy,
                update=prior,
                run_root=args.run_root,
            )

    if next_update > args.total_updates:
        print("CENTER_EGO_TRAINING_ALREADY_COMPLETE", flush=True)
        return 0

    for update in range(next_update, args.total_updates + 1):
        resume = (args.run_root / "run_state.json").is_file()
        command = _training_command(
            args,
            target_update=update,
            resume=resume,
            skip_map_load=(args.skip_first_map_load or update > next_update),
        )
        print(
            f"CENTER_EGO_UPDATE_START update={update}/{args.total_updates} "
            f"epochs={MAX_EPOCHS_PER_UPDATE}",
            flush=True,
        )
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(swept_train.MAP_LOADER.parent),
            env=portable_subprocess_environment(),
        )
        if completed.returncode != 0:
            return int(completed.returncode)
        policy = swept_train._policy_path(args.run_root, update)
        if not args.skip_validation:
            _validate(policy, update=update, run_root=args.run_root)
        print(f"CENTER_EGO_UPDATE_END update={update}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CENTER_DEFAULT_RUN_ROOT",
    "CENTER_TOTAL_UPDATES",
    "build_parser",
    "main",
]
