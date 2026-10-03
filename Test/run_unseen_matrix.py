"""Run the preregistered unseen-test matrix, one CARLA town per process."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from Test.test_case import PROJECT_ROOT, TEST_SPECS


RESULT_ROOT = PROJECT_ROOT / "Test" / "results"
STATE_ROOT = PROJECT_ROOT / "Test" / "runtime"


@dataclass(frozen=True)
class Policy:
    name: str
    model: Path
    sha256: str
    scenarios: tuple[str, ...]


POLICIES = (
    Policy(
        "joint_seed007_p28",
        PROJECT_ROOT / "runs_center_ego/models/policy_update_028.zip",
        "7010f6999fce18d0597e3d92d2e907589394075a1bf2ccbd9cd7bc934e3cb76e",
        ("s1", "s2", "s3", "s4", "s5", "s6"),
    ),
    Policy(
        "joint_seed027_p32",
        PROJECT_ROOT.parent
        / "Aug24_ppo_OldOD_MultiSeed/runs_center_ego_old_od_seed27_r1/models/policy_update_032.zip",
        "4a333ff7b8160daabab981f1bbf9595af720a63588d0e0f7e5c32cfcf0083591",
        ("s1", "s2", "s3", "s4", "s5", "s6"),
    ),
    Policy(
        "joint_seed037_p26",
        PROJECT_ROOT.parent
        / "Aug24_ppo_OldOD_MultiSeed/runs_center_ego_old_od_seed37_r1/models/policy_update_026.zip",
        "cfbbbacb846b493d630b147cb26e5624c5fa7e320f2c922ce5851296fea8d18c",
        ("s1", "s2", "s3", "s4", "s5", "s6"),
    ),
    Policy(
        "s1to5_seed007_p13",
        PROJECT_ROOT.parent
        / "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/s1to5_seed007_r1/models/policy_update_013.zip",
        "36ae09f082757541d30c8b3556e00ec6f1b4e051f358749bba0853f2915269bb",
        ("s1", "s2", "s3", "s4", "s5"),
    ),
    Policy(
        "s1to5_seed027_p17",
        PROJECT_ROOT.parent
        / "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/s1to5_seed027_r1/models/policy_update_017.zip",
        "db199576440d122fbc115f08466c13624cdf9987c6ea63672424f1878d81b156",
        ("s1", "s2", "s3", "s4", "s5"),
    ),
    Policy(
        "s1to5_seed047_p19",
        PROJECT_ROOT.parent
        / "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/s1to5_seed047_r1/models/policy_update_019.zip",
        "004b7ba3bfd09383977fd42a5218e2ab3db7bdea3b275458794eaf6496f8feb4",
        ("s1", "s2", "s3", "s4", "s5"),
    ),
    Policy(
        "s6_specialist_seed007_p32",
        PROJECT_ROOT.parent
        / "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/s6_specialist_seed007_r1/models/policy_update_032.zip",
        "31ded4e990704b0b143e935377309e56005d48728c0e339687004ee66fb38847",
        ("s6",),
    ),
    Policy(
        "s6_specialist_seed027_p30",
        PROJECT_ROOT.parent
        / "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/s6_specialist_seed027_r1/models/policy_update_030.zip",
        "462f70006f02e8aaf59beef0371fa963b9679af5deacb0471bf334accebfdbe7",
        ("s6",),
    ),
    Policy(
        "s2_specialist_seed007_p31",
        PROJECT_ROOT.parent
        / "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/s2_specialist_seed007_r1/models/policy_update_031.zip",
        "056f3219d9964904422b7e91563318d1157fff51f8f013616303ee23c9f089cc",
        ("s2",),
    ),
)

TOWN_SCENARIOS = {
    "town02": ("s5", "s6"),
    "town05": ("s1", "s2"),
    "town10hd": ("s3", "s4"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed(policy: Policy, scenario: str, episodes_per_origin: int) -> bool:
    directory = RESULT_ROOT / policy.name / scenario
    if not directory.is_dir():
        return False
    for path in sorted(directory.glob("summary_*.json"), reverse=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        protocol = value.get("protocol") or {}
        if (
            value.get("completed") is True
            and value.get("manifest_status") == "ready"
            and value.get("model_sha256") == policy.sha256
            and int(protocol.get("episodes_per_origin", -1)) == episodes_per_origin
            and int(value.get("completed_episodes", -1)) == 3 * episodes_per_origin
        ):
            return True
    return False


def _write_state(town: str, payload: dict) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path = STATE_ROOT / f"unseen_queue_{town}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", required=True, choices=tuple(TOWN_SCENARIOS))
    parser.add_argument("--episodes-per-origin", type=int, default=50)
    args = parser.parse_args()
    if args.episodes_per_origin < 1:
        raise ValueError("--episodes-per-origin must be positive")

    tasks = [
        (policy, scenario)
        for policy in POLICIES
        for scenario in policy.scenarios
        if scenario in TOWN_SCENARIOS[args.town]
    ]
    state = {
        "town": args.town,
        "episodes_per_origin": args.episodes_per_origin,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tasks": [],
    }
    _write_state(args.town, state)
    for policy, scenario in tasks:
        if not policy.model.is_file():
            raise FileNotFoundError(policy.model)
        actual_sha = _sha256(policy.model)
        if actual_sha != policy.sha256:
            raise ValueError(
                f"model hash mismatch for {policy.name}: {actual_sha} != {policy.sha256}"
            )
        if _completed(policy, scenario, args.episodes_per_origin):
            print(f"[QUEUE {args.town}] skip complete {policy.name}/{scenario}", flush=True)
            state["tasks"].append(
                {"policy": policy.name, "scenario": scenario, "status": "skipped_complete"}
            )
            _write_state(args.town, state)
            continue
        spec = TEST_SPECS[scenario]
        task = {"policy": policy.name, "scenario": scenario, "status": "running"}
        state["tasks"].append(task)
        _write_state(args.town, state)
        command = [
            sys.executable,
            "-m",
            "Test.run_policy_test",
            "--scenario",
            scenario,
            "--model",
            str(policy.model),
            "--policy-name",
            policy.name,
            "--episodes-per-origin",
            str(args.episodes_per_origin),
            "--carla-port",
            str(spec.carla_port),
            "--tm-port",
            str(spec.tm_port),
            "--sumo-port",
            str(spec.sumo_port),
            "--live-log-every",
            "500",
        ]
        print(f"[QUEUE {args.town}] start {policy.name}/{scenario}", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            task["status"] = "failed"
            task["returncode"] = result.returncode
            state["status"] = "failed"
            state["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            _write_state(args.town, state)
            return int(result.returncode)
        task["status"] = "complete"
        _write_state(args.town, state)
    state["status"] = "complete"
    state["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_state(args.town, state)
    print(f"[QUEUE {args.town}] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
