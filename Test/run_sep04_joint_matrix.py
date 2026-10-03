"""Run the Sep-04 Joint S1-S6 checkpoints on the Draft-4 S1345 test set.

The queue is intentionally isolated from the older unseen-test matrices.  It
locks the two imported checkpoint hashes, records code/runtime/test-input
provenance, and resumes only from completed, internally verified cells.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from Test.test_case import PROJECT_ROOT, TEST_SPECS, load_test_case


BATCH_ID = "sep04_joint_seed017_p27_seed047_p29_s1345"
RESULT_ROOT = PROJECT_ROOT / "Test" / "results"
STATE_ROOT = PROJECT_ROOT / "Test" / "runtime"
DEFAULT_TEST_SEED = 1007


@dataclass(frozen=True)
class Policy:
    name: str
    model: Path
    sha256: str


@dataclass(frozen=True)
class Task:
    policy: Policy
    scenario: str

    @property
    def task_id(self) -> str:
        return f"{self.policy.name}/{self.scenario}"


POLICIES = (
    Policy(
        "joint_seed017_p27",
        PROJECT_ROOT
        / "Test/Sep04_Seed17P27_Seed47P29/Seed17_P27/policy_update_027.zip",
        "0100c7211aac80394ead7df7a3e336a813a95ec7903eb32bac926de409b62b2c",
    ),
    Policy(
        "joint_seed047_p29",
        PROJECT_ROOT
        / "Test/Sep04_Seed17P27_Seed47P29/Seed47_P29/policy_update_029.zip",
        "375c583100d675156addf9ebeb45b0c61f6f350861af115415b09009fd62254e",
    ),
)

TOWN_SCENARIOS = {
    "town02": ("s5",),
    "town05": ("s1",),
    "town10hd": ("s3", "s4"),
}

PROVENANCE_FILES = (
    PROJECT_ROOT / "Test/run_sep04_joint_matrix.py",
    PROJECT_ROOT / "Test/run_policy_test.py",
    PROJECT_ROOT / "Test/test_case.py",
    PROJECT_ROOT / "Test/Scenarios/test/check_templates.py",
    PROJECT_ROOT / "baseline/PPO/validate_allSwithoneExe.py",
    PROJECT_ROOT / "env/gym_wrapper_center_v2.py",
    PROJECT_ROOT / "env/carla_sumo_env_center_v2.py",
    PROJECT_ROOT / "env/carla_sumo_env_v2.py",
    PROJECT_ROOT / "env/observation_encoder_v2.py",
    PROJECT_ROOT / "logic/reward_v2.py",
    PROJECT_ROOT / "logic/episode_termination_center_v2.py",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_hash(path: Path | None) -> str | None:
    return _sha256(path) if path is not None else None


def _test_input_hashes(scenario: str) -> dict[str, str | None]:
    case = load_test_case(scenario)
    controller_path = None
    if case.spec.runtime_scenario_id == "s4":
        module = importlib.import_module(
            "Test.Scenarios.test.S4_Town10HD_Jaywalker.jaywalker_controller"
        )
        controller_path = Path(module.__file__).resolve()
    return {
        "manifest_sha256": _sha256(case.manifest_path),
        "config_sha256": _sha256(case.config_path),
        "network_sha256": _optional_hash(case.network_path),
        "route_sha256": _optional_hash(case.route_path),
        "jaywalker_controller_sha256": _optional_hash(controller_path),
    }


def _runtime_provenance() -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("gymnasium", "numpy", "stable-baselines3", "torch", "eclipse-sumo"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at": _now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "requested_device": "cuda",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in PROVENANCE_FILES
        },
        "policies": {
            policy.name: {
                "path": str(policy.model),
                "expected_sha256": policy.sha256,
                "actual_sha256": _sha256(policy.model),
            }
            for policy in POLICIES
        },
        "test_inputs": {
            scenario: _test_input_hashes(scenario)
            for scenario in ("s1", "s3", "s4", "s5")
        },
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _matching_episode_csv(summary_path: Path) -> Path:
    prefix = "summary_"
    timestamp = (
        summary_path.stem[len(prefix) :]
        if summary_path.stem.startswith(prefix)
        else summary_path.stem
    )
    return summary_path.with_name(f"episodes_{timestamp}.csv")


def _verify_episode_csv(
    csv_path: Path,
    *,
    episodes_per_origin: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    target = 3 * episodes_per_origin
    if len(rows) != target:
        raise ValueError(f"{csv_path} has {len(rows)} rows, expected {target}")
    origins = Counter(int(row["origin_index"]) for row in rows)
    expected_origins = {index: episodes_per_origin for index in range(3)}
    if dict(origins) != expected_origins:
        raise ValueError(f"origin coverage mismatch in {csv_path}: {dict(origins)}")
    episodes = [int(row["episode"]) for row in rows]
    if episodes != list(range(1, target + 1)):
        raise ValueError(f"non-contiguous episode IDs in {csv_path}")
    reasons = Counter(row["reason"] for row in rows)
    summary_reasons = Counter((summary.get("overall") or {}).get("reason_counts") or {})
    if reasons != summary_reasons:
        raise ValueError(
            f"terminal-reason mismatch in {csv_path}: {dict(reasons)} != "
            f"{dict(summary_reasons)}"
        )
    return {
        "path": str(csv_path),
        "sha256": _sha256(csv_path),
        "rows": len(rows),
        "origin_counts": dict(sorted(origins.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def _latest_valid_result(
    task: Task,
    *,
    episodes_per_origin: int,
    expected_inputs: dict[str, str | None],
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    result_dir = RESULT_ROOT / task.policy.name / task.scenario
    for path in sorted(result_dir.glob("summary_*.json"), reverse=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            protocol = value.get("protocol") or {}
            condition = value.get("experimental_condition") or {}
            recorded_inputs = value.get("test_inputs") or {}
            valid = (
                value.get("completed") is True
                and value.get("interrupted") is False
                and value.get("dataset_split") == "test"
                and value.get("deterministic") is True
                and value.get("manifest_status") == "ready"
                and value.get("policy_name") == task.policy.name
                and value.get("test_id") == task.scenario
                and int(value.get("seed", -1)) == DEFAULT_TEST_SEED
                and value.get("model_sha256") == task.policy.sha256
                and int(protocol.get("episodes_per_origin", -1))
                == episodes_per_origin
                and int(protocol.get("origin_count", -1)) == 3
                and int(protocol.get("target_episodes", -1))
                == 3 * episodes_per_origin
                and protocol.get("infrastructure_faults_excluded") is True
                and int(value.get("completed_episodes", -1))
                == 3 * episodes_per_origin
                and condition.get("background_traffic_mode") == "configured"
                and all(
                    recorded_inputs.get(key) == expected
                    for key, expected in expected_inputs.items()
                )
            )
            if not valid:
                continue
            csv_audit = _verify_episode_csv(
                _matching_episode_csv(path),
                episodes_per_origin=episodes_per_origin,
                summary=value,
            )
            return path, value, csv_audit
        except Exception:
            continue
    return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", required=True, choices=tuple(TOWN_SCENARIOS))
    parser.add_argument("--episodes-per-origin", type=int, default=50)
    args = parser.parse_args()
    if args.episodes_per_origin < 1:
        raise ValueError("--episodes-per-origin must be positive")

    provenance = _runtime_provenance()
    if not provenance["cuda_available"]:
        raise RuntimeError("CUDA is required for the Sep-04 formal test matrix")
    for name, metadata in provenance["policies"].items():
        if metadata["actual_sha256"] != metadata["expected_sha256"]:
            raise ValueError(f"model hash mismatch for {name}: {metadata}")

    state_path = STATE_ROOT / f"{BATCH_ID}_{args.town}.json"
    state: dict[str, Any] = {
        "batch_id": BATCH_ID,
        "town": args.town,
        "episodes_per_origin": args.episodes_per_origin,
        "status": "running",
        "started_at": _now(),
        "provenance": provenance,
        "tasks": [],
    }
    _write_state(state_path, state)

    tasks = [
        Task(policy, scenario)
        for policy in POLICIES
        for scenario in TOWN_SCENARIOS[args.town]
    ]
    for task in tasks:
        expected_inputs = provenance["test_inputs"][task.scenario]
        summary_path, summary, csv_audit = _latest_valid_result(
            task,
            episodes_per_origin=args.episodes_per_origin,
            expected_inputs=expected_inputs,
        )
        if summary is not None:
            print(f"[SEP04 {args.town}] skip verified {task.task_id}", flush=True)
            state["tasks"].append(
                {
                    "task_id": task.task_id,
                    "status": "skipped_verified",
                    "summary": str(summary_path),
                    "summary_sha256": _sha256(summary_path),
                    "csv_audit": csv_audit,
                    "success_rate": (summary.get("overall") or {}).get(
                        "success_rate"
                    ),
                    "infrastructure_faults": summary.get("infrastructure_faults")
                    or {},
                }
            )
            _write_state(state_path, state)
            continue

        spec = TEST_SPECS[task.scenario]
        command = [
            sys.executable,
            "-B",
            "-m",
            "Test.run_policy_test",
            "--scenario",
            task.scenario,
            "--model",
            str(task.policy.model),
            "--policy-name",
            task.policy.name,
            "--episodes-per-origin",
            str(args.episodes_per_origin),
            "--seed",
            str(DEFAULT_TEST_SEED),
            "--device",
            "cuda",
            "--carla-port",
            str(spec.carla_port),
            "--tm-port",
            str(spec.tm_port),
            "--sumo-port",
            str(spec.sumo_port),
            "--live-log-every",
            "500",
        ]
        record = {
            "task_id": task.task_id,
            "status": "running",
            "model": str(task.policy.model),
            "model_sha256": task.policy.sha256,
            "command": command,
            "started_at": _now(),
        }
        state["tasks"].append(record)
        _write_state(state_path, state)
        print(f"[SEP04 {args.town}] start {task.task_id}", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            record["status"] = "failed"
            record["returncode"] = int(result.returncode)
            record["ended_at"] = _now()
            state["status"] = "failed"
            state["ended_at"] = record["ended_at"]
            _write_state(state_path, state)
            return int(result.returncode)

        summary_path, summary, csv_audit = _latest_valid_result(
            task,
            episodes_per_origin=args.episodes_per_origin,
            expected_inputs=expected_inputs,
        )
        if summary is None or summary_path is None or csv_audit is None:
            raise RuntimeError(f"no verified completed result after {task.task_id}")
        infra = summary.get("infrastructure_faults") or {}
        record.update(
            {
                "status": (
                    "complete_with_infrastructure_retries"
                    if sum(infra.values())
                    else "complete"
                ),
                "ended_at": _now(),
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "csv_audit": csv_audit,
                "success_rate": (summary.get("overall") or {}).get(
                    "success_rate"
                ),
                "infrastructure_faults": infra,
            }
        )
        _write_state(state_path, state)

    state["status"] = "complete"
    state["ended_at"] = _now()
    _write_state(state_path, state)
    print(f"[SEP04 {args.town}] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
