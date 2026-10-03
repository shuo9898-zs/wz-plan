"""Offline contract checks for the center-ego PPO experiment.

This script never creates a CARLA/SUMO environment and never starts training.
It exercises only pure reward/geometry code, command construction, mocked
validation orchestration, and checkpoint-contract metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.PPO import train_three_servers_center_v2 as center_trainer
from baseline.PPO import train_three_servers_v2 as base_trainer
from env.carla_sumo_env_center_v2 import CarlaSumoEnvCenterV2
from env.carla_sumo_env_v2 import CarlaSumoEnvV2
from env.gym_wrapper_v2 import CarlaSumoGymEnv, CarlaSumoGymEnvV2
from logic.episode_termination_center_v2 import (
    CENTER_EGO_JUDGEMENT_V2,
    CENTER_TERMINATION_CONTRACT_VERSION_V2,
)
from logic.reward_v2 import (
    COLLISION_SUMO_VEHICLE_V2,
    GOAL_REACHED_V2,
    OFF_ROAD_V2,
    REWARD_CONTRACT_VERSION_V2,
    RUNNING_V2,
    TIMEOUT_V2,
    WORKZONE_VIOLATION_V2,
    RewardCalculatorV2,
)
from logic.termination_checker_v2 import FinishLineV2, TerminationCheckerV2
from standalone_launcher import load_module
from validation_debug.runner_v2 import (
    SWEPT_EGO_JUDGEMENT_V2,
    build_validation_debug_parser_v2,
)


train_center = load_module("train_center")
validate = load_module("validate")


EXPECTED_REWARD_SHA256 = (
    "d0a552a8995f249ee84690aa23fc0b3627575565e7c7b728b03449ca5ac22d5c"
)


def _require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def _reward_hash() -> str:
    return hashlib.sha256((ROOT / "logic" / "reward_v2.py").read_bytes()).hexdigest()


def _validation_row(validation_id: str, model_sha256: str = "offline-model") -> dict:
    return {
        "validation_id": validation_id,
        "completed": True,
        "ego_judgement": CENTER_EGO_JUDGEMENT_V2,
        "reward_contract": REWARD_CONTRACT_VERSION_V2,
        "reward_source_sha256": _reward_hash(),
        "termination_contract": CENTER_TERMINATION_CONTRACT_VERSION_V2,
        "model": "offline-policy.zip",
        "model_sha256": model_sha256,
        "protocol": {
            "origin_count": 3,
            "episodes_per_origin": 10,
            "target_episodes": 30,
        },
        "overall": {
            "episodes": 30,
            "success_rate": 0.5,
            "collision_rate": 0.1,
            "workzone_violation_rate": 0.1,
            "off_road_rate": 0.1,
            "timeout_rate": 0.2,
            "safety_failure_rate": 0.3,
            "mean_episode_return": 10.0,
            "mean_episode_length": 100.0,
        },
        "infrastructure_faults": {},
    }


def _center_training_plan() -> dict:
    center_trainer._install_center_contract()
    args = base_trainer.build_parser().parse_args([])
    phases = base_trainer.build_phase_plans(args.episodes_per_origin)
    assignments = base_trainer._assignments(args)
    quotas = base_trainer._town_quotas(args, phases)
    return base_trainer._training_plan(args, phases, quotas, assignments)


def main() -> int:
    checks: list[str] = []

    fresh_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from env.carla_sumo_env_center_v2 import "
                "CarlaSumoEnvCenterV2; print(CarlaSumoEnvCenterV2.__name__)"
            ),
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        fresh_import.returncode == 0
        and "CarlaSumoEnvCenterV2" in fresh_import.stdout,
        "fresh center validation import pins TraCI before the shared engine",
        checks,
    )

    _require(
        CarlaSumoGymEnv is CarlaSumoGymEnvV2,
        "default gym entry remains swept-OBB before center worker installation",
        checks,
    )
    _require(
        CarlaSumoEnvCenterV2._initialize_reward_progress
        is CarlaSumoEnvV2._initialize_reward_progress,
        "center env inherits the exact reward initializer",
        checks,
    )
    _require(
        CarlaSumoEnvCenterV2._compute_reward is CarlaSumoEnvV2._compute_reward,
        "center env inherits the exact reward step implementation",
        checks,
    )
    _require(
        _reward_hash() == EXPECTED_REWARD_SHA256,
        "reward source SHA256 matches the prior/current frozen contract",
        checks,
    )

    reward = RewardCalculatorV2(
        origin=(0.0, 0.0),
        finish_line=((100.0, -1.0), (100.0, 1.0)),
    )
    first = reward.step((50.0, 0.0), speed_mps=1.0)
    retreat = reward.step((40.0, 0.0), speed_mps=1.0)
    second = reward.step((60.0, 0.0), speed_mps=1.0)
    goal = reward.step(
        (100.0, 0.0),
        speed_mps=1.0,
        terminal_reason=GOAL_REACHED_V2,
    )
    _require(first.reward == 25.0, "50% new progress pays +25", checks)
    _require(retreat.reward == 0.0, "retreat pays no repeated progress", checks)
    _require(
        math.isclose(second.reward, 5.0, rel_tol=0.0, abs_tol=1e-12),
        "50% to 60% new progress pays +5",
        checks,
    )
    _require(goal.reward == 100.0, "goal terminal step pays +100", checks)

    terminal_expectations = {
        TIMEOUT_V2: -50.0,
        COLLISION_SUMO_VEHICLE_V2: -100.0,
        WORKZONE_VIOLATION_V2: -100.0,
        OFF_ROAD_V2: -100.0,
    }
    for reason, expected in terminal_expectations.items():
        result = RewardCalculatorV2(
            origin=(0.0, 0.0),
            finish_line=((100.0, -1.0), (100.0, 1.0)),
        ).step((0.0, 0.0), speed_mps=1.0, terminal_reason=reason)
        _require(result.reward == expected, f"{reason} pays {expected:+.0f}", checks)
    fast_goal = RewardCalculatorV2(
        origin=(0.0, 0.0),
        finish_line=((100.0, -1.0), (100.0, 1.0)),
    ).step(
        (100.0, 0.0),
        speed_mps=6.0,
        terminal_reason=GOAL_REACHED_V2,
    )
    _require(fast_goal.reward == 120.0, "existing speed-bonus contract remains +20", checks)

    geometry = TerminationCheckerV2(
        scenario_id="s1",
        origin=(0.0, 0.0),
        finish_line=FinishLineV2((10.0, -1.0), (10.0, 1.0)),
        max_episode_steps=100,
        forbidden_polygon=((4.0, -1.0), (6.0, -1.0), (6.0, 1.0), (4.0, 1.0)),
        boundary_tolerance_m=0.0,
    )
    center_clear = geometry.judge(
        previous_position=(3.0, 0.0),
        current_position=(3.0, 0.0),
        episode_step=1,
    )
    footprint_hit = geometry.judge(
        previous_position=(3.0, 0.0),
        current_position=(3.0, 0.0),
        episode_step=1,
        ego_footprint=((3.5, -0.5), (4.5, -0.5), (4.5, 0.5), (3.5, 0.5)),
    )
    center_hit = geometry.judge(
        previous_position=(3.0, 0.0),
        current_position=(5.0, 0.0),
        episode_step=2,
    )
    _require(
        center_clear.reason == RUNNING_V2,
        "center mode ignores a footprint-only work-zone contact",
        checks,
    )
    _require(
        footprint_hit.reason == WORKZONE_VIOLATION_V2,
        "swept footprint detects the same footprint-only contact",
        checks,
    )
    _require(
        center_hit.reason == WORKZONE_VIOLATION_V2,
        "center trajectory still detects a true center-line intrusion",
        checks,
    )

    plan = _center_training_plan()
    rollout = plan["rollout_and_ppo"]
    frozen = {
        str(name).replace("\\", "/"): digest
        for name, digest in plan["frozen_source_sha256"].items()
    }
    _require(
        plan["contract"] == center_trainer.CENTER_THREE_SERVER_CONTRACT_V2,
        "training plan uses the isolated center contract",
        checks,
    )
    _require(plan["steps_per_update"] == 102_400, "update size remains 102,400", checks)
    _require(rollout["ppo_epochs"] == 20, "PPO keeps all 20 epochs", checks)
    _require(rollout["ppo_target_kl"] is None, "KL early stopping remains disabled", checks)
    _require(
        rollout["reward_contract"] == REWARD_CONTRACT_VERSION_V2,
        "training plan keeps the frozen reward contract",
        checks,
    )
    _require(
        rollout["termination_contract"] == CENTER_TERMINATION_CONTRACT_VERSION_V2,
        "training plan records center termination",
        checks,
    )
    _require(
        frozen["logic/reward_v2.py"] == EXPECTED_REWARD_SHA256,
        "training plan freezes the verified reward source",
        checks,
    )
    _require(
        all(
            name in frozen
            for name in (
                "train_center.py",
                "validate.py",
                "baseline/PPO/validate_allSwithoneExe.py",
                "config/scenario_catalog.py",
                "config/scenario_selector.py",
                "validation_debug/loader_v2.py",
                "validation_debug/runner_v2.py",
            )
        ),
        "training plan freezes the center validation implementation",
        checks,
    )
    _require(
        any(name.startswith("validation/") for name in frozen),
        "training plan freezes the held-out validation inputs",
        checks,
    )
    _require(
        base_trainer._worker_command(Path("worker.json"))[2]
        == "baseline.PPO.train_three_servers_center_v2",
        "all rollout workers re-enter the center trainer",
        checks,
    )

    center_args = train_center.build_parser().parse_args([])
    command = train_center._training_command(
        center_args,
        target_update=1,
        resume=False,
        skip_map_load=False,
    )
    _require(
        center_args.run_root == ROOT / "runs_center_ego",
        "default run root is isolated as runs_center_ego",
        checks,
    )
    _require(center_args.total_updates == 35, "new experiment defaults to 35 updates", checks)
    _require(
        command[2] == "baseline.PPO.train_three_servers_center_v2",
        "top-level training command selects the center trainer",
        checks,
    )

    parser = build_validation_debug_parser_v2("s1")
    _require(
        parser.parse_args([]).ego_judgement == SWEPT_EGO_JUDGEMENT_V2,
        "existing standalone validation still defaults to swept OBB",
        checks,
    )
    _require(
        parser.parse_args(
            ["--ego-judgement", CENTER_EGO_JUDGEMENT_V2]
        ).ego_judgement
        == CENTER_EGO_JUDGEMENT_V2,
        "validation parser explicitly accepts center judgement",
        checks,
    )

    summaries = [_validation_row(f"s{index}") for index in range(1, 7)]
    aggregate = validate.aggregate_validation_summaries(summaries)
    _require(aggregate["episodes"] == 180, "validation aggregates exactly 180 episodes", checks)
    _require(
        set(aggregate["scenario_metrics"]) == {f"s{index}" for index in range(1, 7)},
        "validation contains each of S1-S6 exactly once",
        checks,
    )
    mixed = [dict(row) for row in summaries]
    mixed[0] = {**mixed[0], "ego_judgement": SWEPT_EGO_JUDGEMENT_V2}
    try:
        validate.aggregate_validation_summaries(mixed)
    except ValueError:
        checks.append("mixed center/swept validation summaries are rejected")
    else:
        raise AssertionError("mixed validation contracts were accepted")

    duplicate = [dict(row) for row in summaries]
    duplicate[0] = {**duplicate[0], "validation_id": "s2"}
    try:
        validate.aggregate_validation_summaries(duplicate)
    except ValueError:
        checks.append("duplicate/missing scenario IDs are rejected")
    else:
        raise AssertionError("duplicate validation scenarios were accepted")

    with tempfile.TemporaryDirectory(
        prefix="center_validation_verify_",
        dir=str(ROOT / "tools"),
    ) as temp:
        run_root = Path(temp)
        policy = run_root / "models" / "policy_update_001.zip"
        policy.parent.mkdir(parents=True)
        policy.write_bytes(b"offline policy placeholder")
        (run_root / "run_state.json").write_text(
            json.dumps({"plan": plan, "next_update": 2}),
            encoding="utf-8",
        )
        command_validation_root = run_root / "command_validation"
        command_run_dir = command_validation_root / "update_001" / "s1"
        command_run_dir.mkdir(parents=True)
        (command_run_dir / "summary_offline.json").write_text(
            json.dumps({
                "model": str(policy.resolve()),
                "model_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "overall": {"success_rate": 0.0},
            }),
            encoding="utf-8",
        )
        with patch.object(
            validate.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as mocked_subprocess:
            validate._run_scenario(
                "s1",
                policy=policy,
                update=1,
                validation_root=command_validation_root,
                ego_judgement=CENTER_EGO_JUDGEMENT_V2,
            )
        validation_command = mocked_subprocess.call_args.args[0]
        ego_flag = validation_command.index("--ego-judgement")
        _require(
            validation_command[ego_flag + 1] == CENTER_EGO_JUDGEMENT_V2,
            "validation child command explicitly selects center judgement",
            checks,
        )

        policy_hash = hashlib.sha256(policy.read_bytes()).hexdigest()

        def fake_town(scenarios, **_kwargs):
            return [
                _validation_row(scenario, model_sha256=policy_hash)
                for scenario in scenarios
            ]

        with patch.object(validate, "_run_town", side_effect=fake_town):
            result = validate.validate_policy_after_update(
                policy,
                update=1,
                run_root=run_root,
                ego_judgement=CENTER_EGO_JUDGEMENT_V2,
            )
        _require(
            result["metrics"]["episodes"] == 180,
            "mocked end-to-end validation preserves the 180-episode contract",
            checks,
        )
        _require(
            (run_root / "models" / "validation_best_model.zip").is_file(),
            "center validation best model stays inside its own run root",
            checks,
        )

        validation_state_path = run_root / "validation" / "state.json"
        validation_state = json.loads(validation_state_path.read_text(encoding="utf-8"))
        validation_state["best_model"]["metrics"]["ego_judgement"] = (
            SWEPT_EGO_JUDGEMENT_V2
        )
        validation_state_path.write_text(json.dumps(validation_state), encoding="utf-8")
        with patch.object(validate, "_run_town", side_effect=fake_town):
            try:
                validate.validate_policy_after_update(
                    policy,
                    update=2,
                    run_root=run_root,
                    ego_judgement=CENTER_EGO_JUDGEMENT_V2,
                )
            except RuntimeError as error:
                _require(
                    "different contract" in str(error),
                    "foreign validation best-model state is rejected",
                    checks,
                )
            else:
                raise AssertionError("foreign validation state was accepted")

    with tempfile.TemporaryDirectory(
        prefix="center_resume_verify_",
        dir=str(ROOT / "tools"),
    ) as temp:
        run_root = Path(temp)
        policy = run_root / "models" / "policy_update_035.zip"
        policy.parent.mkdir(parents=True)
        policy.write_bytes(b"offline policy placeholder")
        (run_root / "run_state.json").write_text(
            json.dumps({"plan": plan, "next_update": 36}),
            encoding="utf-8",
        )
        with patch.object(train_center, "_validate") as mocked_validate:
            code = train_center.main(
                ["--run-root", str(run_root), "--total-updates", "35"]
            )
        _require(code == 0, "completed training exits cleanly after repair", checks)
        _require(
            mocked_validate.call_count == 1,
            "interrupted final validation is repaired before the complete exit",
            checks,
        )

    print(f"CENTER_EXPERIMENT_OFFLINE_OK checks={len(checks)}")
    for check in checks:
        print(f"OK {check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
