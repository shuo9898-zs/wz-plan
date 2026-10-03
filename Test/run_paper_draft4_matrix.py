"""Run the validation-selected Draft-4 paper test backlog.

This queue is intentionally separate from ``run_unseen_matrix.py``.  It adds
validation-ranked checkpoints (plus retained exploratory candidates) and
repeats only the original policy/scenario pairs whose first test was affected
by simulator retries.
S2 and S6 are excluded from this main-paper batch.
"""
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


BATCH_ID = "paper_draft4_20260903"
RESULT_ROOT = PROJECT_ROOT / "Test" / "results"
STATE_ROOT = PROJECT_ROOT / "Test" / "runtime"


@dataclass(frozen=True)
class Task:
    policy_name: str
    model: Path
    scenario: str
    reason: str
    force_once: bool = False

    @property
    def task_id(self) -> str:
        return f"{self.policy_name}/{self.scenario}"


def _model(relative_path: str) -> Path:
    return PROJECT_ROOT.parent / relative_path


DRAFT_SCENARIOS = ("s1", "s3", "s4", "s5")

CANDIDATE_POLICY_MODELS = {
    "joint_seed007_p30": PROJECT_ROOT / "runs_center_ego/models/policy_update_030.zip",
    # P17 ties P33 on validation macro, but P33 is the true runner-up under
    # validate.py's frozen collision-rate tie-break.
    "joint_seed027_p33": _model(
        "Aug24_ppo_OldOD_MultiSeed/runs_center_ego_old_od_seed27_r1/"
        "models/policy_update_033.zip"
    ),
    "joint_seed027_p17": _model(
        "Aug24_ppo_OldOD_MultiSeed/runs_center_ego_old_od_seed27_r1/"
        "models/policy_update_017.zip"
    ),
    "joint_seed037_p31": _model(
        "Aug24_ppo_OldOD_MultiSeed/runs_center_ego_old_od_seed37_r1/"
        "models/policy_update_031.zip"
    ),
    "road_arc_seed007_p21": PROJECT_ROOT
    / "baseline/baseline_ours/runs_center_ego_road_arc_r1/models/policy_update_021.zip",
    # P31 is the validation-selected Road-Arc best checkpoint.  P21 is #2;
    # P13 remains in this batch as an already-tested additional candidate.
    "road_arc_seed007_p31": PROJECT_ROOT
    / "baseline/baseline_ours/runs_center_ego_road_arc_r1/models/policy_update_031.zip",
    "road_arc_seed007_p13": PROJECT_ROOT
    / "baseline/baseline_ours/runs_center_ego_road_arc_r1/models/policy_update_013.zip",
    "s1to5_seed007_p32": _model(
        "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/"
        "s1to5_seed007_r1/models/policy_update_032.zip"
    ),
    "s1to5_seed027_p26": _model(
        "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/"
        "s1to5_seed027_r1/models/policy_update_026.zip"
    ),
    "s1to5_seed047_p25": _model(
        "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/"
        "s1to5_seed047_r1/models/policy_update_025.zip"
    ),
}

POLICY_ROLES = {
    "joint_seed007_p30": "validation runner-up",
    "joint_seed027_p33": "validation runner-up",
    "joint_seed027_p17": "exploratory; macro-tied checkpoint that loses the frozen collision tie-break",
    "joint_seed037_p31": "validation runner-up",
    "road_arc_seed007_p21": "validation runner-up",
    "road_arc_seed007_p31": "validation best",
    "road_arc_seed007_p13": "exploratory validation rank 4",
    "s1to5_seed007_p32": "validation runner-up",
    "s1to5_seed027_p26": "validation runner-up",
    "s1to5_seed047_p25": "validation runner-up",
}

RETRY_TASKS = (
    Task(
        "joint_seed027_p32",
        _model(
            "Aug24_ppo_OldOD_MultiSeed/runs_center_ego_old_od_seed27_r1/"
            "models/policy_update_032.zip"
        ),
        "s5",
        "repeat infrastructure-flagged result (9 SUMO retries)",
        True,
    ),
    Task(
        "joint_seed037_p26",
        _model(
            "Aug24_ppo_OldOD_MultiSeed/runs_center_ego_old_od_seed37_r1/"
            "models/policy_update_026.zip"
        ),
        "s5",
        "repeat infrastructure-flagged result (6 SUMO retries)",
        True,
    ),
    Task(
        "s1to5_seed027_p17",
        _model(
            "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/"
            "s1to5_seed027_r1/models/policy_update_017.zip"
        ),
        "s1",
        "repeat infrastructure-flagged result (6 SUMO retries)",
        True,
    ),
    Task(
        "s1to5_seed047_p19",
        _model(
            "Aug24_ppo_OldOD_MultiSeed/Aug29_S6_S1to5_Queue/runs/"
            "s1to5_seed047_r1/models/policy_update_019.zip"
        ),
        "s5",
        "repeat infrastructure-flagged result (14 SUMO retries)",
        True,
    ),
)

# A second run reproduced the original failure at the same origin and step
# across fresh SUMO PIDs.  Retrying until 50 surviving episodes would be
# rejection sampling, so this cell is retained as an explicit blocker rather
# than silently converted into a completed paper metric.
QUARANTINED_TASKS = (
    {
        "task_id": "joint_seed027_p32/s1",
        "status": "quarantined",
        "reason": (
            "policy/origin/trajectory-correlated SUMO 1.27.1 native crash "
            "(0xC0000005 near origin2 step 113); survivor-only success is invalid"
        ),
        "evidence": str(
            RESULT_ROOT
            / "joint_seed027_p32/s1/episodes_20260903_155747_765.csv"
        ),
    },
    {
        "task_id": "s1to5_seed007_p13/s5",
        "status": "quarantined",
        "reason": (
            "policy/origin/trajectory-correlated SUMO 1.27.1 native crash "
            "(0xC0000005 near origin2 steps 45-47); survivor-only success is invalid"
        ),
        "evidence": str(
            RESULT_ROOT
            / "s1to5_seed007_p13/s5/episodes_20260903_162236_786.csv"
        ),
    },
    {
        "task_id": "joint_seed027_p17/s1",
        "status": "quarantined",
        "reason": (
            "policy/origin/trajectory-correlated SUMO 1.27.1 native crash "
            "(0xC0000005 near origin1 step 74); survivor-only success is invalid"
        ),
        "evidence": str(
            RESULT_ROOT
            / "joint_seed027_p17/s1/episodes_20260903_162830_009.csv"
        ),
    },
    {
        "task_id": "s1to5_seed007_p32/s1",
        "status": "quarantined",
        "reason": (
            "policy/origin/trajectory-correlated SUMO 1.27.1 native crash "
            "(0xC0000005 near origin2 step 45); survivor-only success is invalid"
        ),
        "evidence": str(
            RESULT_ROOT
            / "s1to5_seed007_p32/s1/episodes_20260903_164841_830.csv"
        ),
    },
    {
        "task_id": "joint_seed037_p31/s5",
        "status": "quarantined",
        "reason": (
            "policy/origin/trajectory-correlated SUMO 1.27.1 native crash "
            "(0xC0000005 repeatedly near origin1 steps 45-47); "
            "survivor-only success is invalid"
        ),
        "evidence": str(
            RESULT_ROOT
            / "joint_seed037_p31/s5/episodes_20260903_165713_268.csv"
        ),
    },
    {
        "task_id": "road_arc_seed007_p13/s5",
        "status": "quarantined",
        "reason": (
            "policy/origin/trajectory-correlated SUMO 1.27.1 native crash "
            "(0xC0000005 repeatedly near origin2 steps 50-58, including "
            "three consecutive crashes in one episode); survivor-only success is invalid"
        ),
        "evidence": str(
            RESULT_ROOT
            / "road_arc_seed007_p13/s5/episodes_20260903_170751_755.csv"
        ),
    },
)

TOWN_SCENARIOS = {
    "town02": ("s5",),
    "town05": ("s1",),
    "town10hd": ("s3", "s4"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_valid_summary(task: Task, model_sha256: str, episodes_per_origin: int):
    result_dir = RESULT_ROOT / task.policy_name / task.scenario
    for path in sorted(result_dir.glob("summary_*.json"), reverse=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        protocol = value.get("protocol") or {}
        if (
            value.get("completed") is True
            and value.get("interrupted") is False
            and value.get("dataset_split") == "test"
            and value.get("deterministic") is True
            and value.get("manifest_status") == "ready"
            and value.get("policy_name") == task.policy_name
            and value.get("test_id") == task.scenario
            and int(value.get("seed", -1)) == 1007
            and value.get("model_sha256") == model_sha256
            and int(protocol.get("episodes_per_origin", -1)) == episodes_per_origin
            and int(protocol.get("origin_count", -1)) == 3
            and int(protocol.get("target_episodes", -1)) == 3 * episodes_per_origin
            and protocol.get("infrastructure_faults_excluded") is True
            and int(value.get("completed_episodes", -1)) == 3 * episodes_per_origin
        ):
            return path, value
    return None, None


def _load_resume_state(path: Path, episodes_per_origin: int) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if (
        state.get("batch_id") != BATCH_ID
        or int(state.get("episodes_per_origin", -1)) != episodes_per_origin
    ):
        return {}
    return {
        str(task.get("task_id")): task
        for task in state.get("tasks", [])
        if isinstance(task, dict) and task.get("task_id")
    }


def _write_state(path: Path, state: dict) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _tasks_for_town(town: str) -> list[Task]:
    scenarios = TOWN_SCENARIOS[town]
    tasks = [task for task in RETRY_TASKS if task.scenario in scenarios]
    tasks.extend(
        Task(name, model, scenario, POLICY_ROLES[name])
        for name, model in CANDIDATE_POLICY_MODELS.items()
        for scenario in DRAFT_SCENARIOS
        if scenario in scenarios
    )
    quarantined_ids = {str(item["task_id"]) for item in QUARANTINED_TASKS}
    return [task for task in tasks if task.task_id not in quarantined_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", required=True, choices=tuple(TOWN_SCENARIOS))
    parser.add_argument("--episodes-per-origin", type=int, default=50)
    args = parser.parse_args()
    if args.episodes_per_origin < 1:
        raise ValueError("--episodes-per-origin must be positive")

    state_path = STATE_ROOT / f"{BATCH_ID}_{args.town}.json"
    prior = _load_resume_state(state_path, args.episodes_per_origin)
    state = {
        "batch_id": BATCH_ID,
        "town": args.town,
        "episodes_per_origin": args.episodes_per_origin,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "quarantined_tasks": list(QUARANTINED_TASKS),
        "tasks": [],
    }
    _write_state(state_path, state)

    for task in _tasks_for_town(args.town):
        if not task.model.is_file():
            raise FileNotFoundError(task.model)
        model_sha = _sha256(task.model)
        old = prior.get(task.task_id, {})
        summary_path, summary = _latest_valid_summary(
            task, model_sha, args.episodes_per_origin
        )
        if (
            old.get("status") in {"complete", "complete_with_infra_faults"}
            and summary is not None
        ):
            print(f"[DRAFT4 {args.town}] resume-skip {task.task_id}", flush=True)
            infra_faults = summary.get("infrastructure_faults") or {}
            normalized = dict(old)
            normalized.update(
                {
                    "reason": task.reason,
                    "model": str(task.model),
                    "model_sha256": model_sha,
                    "summary": str(summary_path),
                    "infrastructure_faults": infra_faults,
                    "success_rate": (summary.get("overall") or {}).get("success_rate"),
                }
            )
            state["tasks"].append(normalized)
            _write_state(state_path, state)
            continue
        if summary is not None and not task.force_once:
            print(f"[DRAFT4 {args.town}] skip complete {task.task_id}", flush=True)
            state["tasks"].append(
                {
                    "task_id": task.task_id,
                    "status": "skipped_complete",
                    "reason": task.reason,
                    "model": str(task.model),
                    "model_sha256": model_sha,
                    "summary": str(summary_path),
                    "infrastructure_faults": summary.get("infrastructure_faults") or {},
                    "success_rate": (summary.get("overall") or {}).get("success_rate"),
                }
            )
            _write_state(state_path, state)
            continue

        record = {
            "task_id": task.task_id,
            "status": "running",
            "reason": task.reason,
            "model": str(task.model),
            "model_sha256": model_sha,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        state["tasks"].append(record)
        _write_state(state_path, state)
        spec = TEST_SPECS[task.scenario]
        command = [
            sys.executable,
            "-m",
            "Test.run_policy_test",
            "--scenario",
            task.scenario,
            "--model",
            str(task.model),
            "--policy-name",
            task.policy_name,
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
        print(f"[DRAFT4 {args.town}] start {task.task_id}", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            record["status"] = "failed"
            record["returncode"] = int(result.returncode)
            record["ended_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            state["status"] = "failed"
            state["ended_at"] = record["ended_at"]
            _write_state(state_path, state)
            return int(result.returncode)

        summary_path, summary = _latest_valid_summary(
            task, model_sha, args.episodes_per_origin
        )
        if summary is None:
            raise RuntimeError(f"no completed summary found after {task.task_id}")
        infra_faults = summary.get("infrastructure_faults") or {}
        record["status"] = (
            "complete_with_infra_faults" if sum(infra_faults.values()) else "complete"
        )
        record["summary"] = str(summary_path)
        record["infrastructure_faults"] = infra_faults
        record["success_rate"] = (summary.get("overall") or {}).get("success_rate")
        record["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_state(state_path, state)

    state["status"] = "complete"
    state["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_state(state_path, state)
    print(f"[DRAFT4 {args.town}] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
