"""Three-server PPO V2 training for the centre-ego controlled ablation.

The production three-server implementation is reused rather than copied.
This module installs one explicit environment variant in both parent/worker
processes and gives the run a distinct immutable contract and source hashes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import baseline.PPO.train_three_servers_v2 as base
import env.gym_wrapper_v2 as default_gym_wrapper
from env.gym_wrapper_center_v2 import CarlaSumoGymEnvCenterV2
from logic.episode_termination_center_v2 import (
    CENTER_EGO_JUDGEMENT_V2,
    CENTER_TERMINATION_CONTRACT_VERSION_V2,
)


CENTER_THREE_SERVER_CONTRACT_V2 = (
    "ppo_v2_three_fixed_town_servers_center_ego_no_kl_v1"
)

_ORIGINAL_TRAINING_PLAN = base._training_plan
_ORIGINAL_WORKER_SPEC = base._worker_spec
_PATCHED = False


def _center_training_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
    plan = _ORIGINAL_TRAINING_PLAN(*args, **kwargs)
    plan["contract"] = CENTER_THREE_SERVER_CONTRACT_V2
    plan["ego_judgement"] = CENTER_EGO_JUDGEMENT_V2
    rollout = dict(plan["rollout_and_ppo"])
    rollout["termination_contract"] = CENTER_TERMINATION_CONTRACT_VERSION_V2
    rollout["workzone_geometry_reference"] = CENTER_EGO_JUDGEMENT_V2
    plan["rollout_and_ppo"] = rollout

    workspace_root = Path(__file__).resolve().parents[2]
    validation_code = (
        workspace_root / "train_center.py",
        workspace_root / "validate.py",
        workspace_root / "experiment_config.py",
        workspace_root / "baseline" / "PPO" / "validate_allSwithoneExe.py",
        workspace_root / "config" / "scenario_catalog.py",
        workspace_root / "config" / "scenario_selector.py",
        workspace_root / "validation_debug" / "loader_v2.py",
        workspace_root / "validation_debug" / "runner_v2.py",
        workspace_root / "validation_debug" / "visuals_v2.py",
        *(
            workspace_root / "validation_debug" / f"debug_s{index}.py"
            for index in range(1, 7)
        ),
    )
    validation_inputs = tuple(
        path
        for path in sorted((workspace_root / "validation").rglob("*"))
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".xml"}
    )
    extra_sources = (
        Path(__file__).resolve(),
        workspace_root / "logic" / "episode_termination_center_v2.py",
        workspace_root / "env" / "carla_sumo_env_center_v2.py",
        workspace_root / "env" / "gym_wrapper_center_v2.py",
        *validation_code,
        *validation_inputs,
    )
    frozen = dict(plan["frozen_source_sha256"])
    frozen.update({
        str(path.relative_to(workspace_root)): base.file_digest(path)
        for path in extra_sources
    })
    plan["frozen_source_sha256"] = frozen
    return plan


def _center_worker_spec(*args: Any, **kwargs: Any) -> dict[str, Any]:
    spec = _ORIGINAL_WORKER_SPEC(*args, **kwargs)
    spec["contract"] = CENTER_THREE_SERVER_CONTRACT_V2
    spec["ego_judgement"] = CENTER_EGO_JUDGEMENT_V2
    spec["termination_contract"] = CENTER_TERMINATION_CONTRACT_VERSION_V2
    return spec


def _center_worker_command(spec_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "baseline.PPO.train_three_servers_center_v2",
        "--worker-spec",
        str(spec_path),
    ]


def _install_center_contract() -> None:
    global _PATCHED
    if _PATCHED:
        return
    base.THREE_SERVER_CONTRACT = CENTER_THREE_SERVER_CONTRACT_V2
    base._training_plan = _center_training_plan
    base._worker_spec = _center_worker_spec
    base._worker_command = _center_worker_command
    # _worker_main imports this alias lazily.  Every worker is a dedicated
    # process, so the substitution cannot leak into swept-OBB training.
    default_gym_wrapper.CarlaSumoGymEnv = CarlaSumoGymEnvCenterV2
    _PATCHED = True


def _validate_worker_spec_argument(arguments: list[str]) -> None:
    if arguments[:1] != ["--worker-spec"]:
        return
    if len(arguments) != 2:
        return
    spec = json.loads(Path(arguments[1]).read_text(encoding="utf-8"))
    if spec.get("contract") != CENTER_THREE_SERVER_CONTRACT_V2:
        raise ValueError("center-ego worker specification contract mismatch")
    if spec.get("ego_judgement") != CENTER_EGO_JUDGEMENT_V2:
        raise ValueError("center-ego worker specification judgement mismatch")
    if (
        spec.get("termination_contract")
        != CENTER_TERMINATION_CONTRACT_VERSION_V2
    ):
        raise ValueError("center-ego worker termination contract mismatch")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    _install_center_contract()
    _validate_worker_spec_argument(arguments)
    return base.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CENTER_THREE_SERVER_CONTRACT_V2",
    "main",
]
