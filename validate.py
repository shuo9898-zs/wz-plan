"""Evaluate one new policy on all six held-out validation cases."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

from baseline.PPO.global_rollout_checkpoint import atomic_write_json, file_digest
from logic.episode_termination_center_v2 import (
    CENTER_EGO_JUDGEMENT_V2,
    CENTER_TERMINATION_CONTRACT_VERSION_V2,
)
from logic.reward_v2 import REWARD_CONTRACT_VERSION_V2
from logic.termination_checker_v2 import TERMINATION_CONTRACT_VERSION_V2

from .bootstrap import BUNDLE_ROOT, portable_subprocess_environment
from .experiment_config import (
    SEED,
    TOWNS,
    VALIDATION_BEST_NAME,
    VALIDATION_DEVICE,
    VALIDATION_EPISODES_PER_ORIGIN,
    VALIDATION_TOTAL_EPISODES,
)


RATE_FIELDS = (
    "success_rate",
    "collision_rate",
    "workzone_violation_rate",
    "off_road_rate",
    "timeout_rate",
    "safety_failure_rate",
)
SWEPT_EGO_JUDGEMENT = "swept_ego_obb"
VALIDATION_TERMINATION_CONTRACTS = {
    SWEPT_EGO_JUDGEMENT: TERMINATION_CONTRACT_VERSION_V2,
    CENTER_EGO_JUDGEMENT_V2: CENTER_TERMINATION_CONTRACT_VERSION_V2,
}


def validation_selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Predeclared safety-aware checkpoint ordering."""
    return (
        float(metrics["macro_success_rate"]),
        -float(metrics["macro_collision_rate"]),
        -float(metrics["macro_workzone_violation_rate"]),
        -float(metrics["macro_off_road_rate"]),
        -float(metrics["macro_timeout_rate"]),
        float(metrics["macro_mean_episode_return"]),
    )


def aggregate_validation_summaries(
    summaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(summaries)
    if len(rows) != 6:
        raise ValueError(f"expected six validation summaries, got {len(rows)}")
    if any(not bool(row.get("completed")) for row in rows):
        raise ValueError("validation contains an incomplete scenario")
    contract_fields = (
        "ego_judgement",
        "reward_contract",
        "reward_source_sha256",
        "termination_contract",
        "model_sha256",
    )
    contracts: dict[str, str] = {}
    for field in contract_fields:
        values = {str(row.get(field, "")) for row in rows}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"validation contains mixed or missing {field}")
        contracts[field] = next(iter(values))
    expected_ids = {f"s{index}" for index in range(1, 7)}
    validation_ids = [str(row.get("validation_id", "")) for row in rows]
    if len(set(validation_ids)) != 6 or set(validation_ids) != expected_ids:
        raise ValueError(
            "validation must contain each of s1, s2, s3, s4, s5, and s6 exactly once"
        )
    expected_per_scenario = 3 * VALIDATION_EPISODES_PER_ORIGIN
    for row in rows:
        validation_id = str(row["validation_id"])
        if int(row["overall"]["episodes"]) != expected_per_scenario:
            raise ValueError(
                f"{validation_id} must contain exactly "
                f"{expected_per_scenario} completed episodes"
            )
        protocol = row.get("protocol") or {}
        if (
            int(protocol.get("origin_count", -1)) != 3
            or int(protocol.get("episodes_per_origin", -1))
            != VALIDATION_EPISODES_PER_ORIGIN
            or int(protocol.get("target_episodes", -1))
            != expected_per_scenario
        ):
            raise ValueError(f"{validation_id} used the wrong validation protocol")
    episodes = sum(int(row["overall"]["episodes"]) for row in rows)
    if episodes != VALIDATION_TOTAL_EPISODES:
        raise ValueError(
            f"expected {VALIDATION_TOTAL_EPISODES} validation episodes, got {episodes}"
        )
    metrics: dict[str, Any] = {
        "episodes": episodes,
        "scenario_count": len(rows),
        "scenario_metrics": {
            str(row["validation_id"]): dict(row["overall"]) for row in rows
        },
        "infrastructure_faults": {
            str(row["validation_id"]): dict(row.get("infrastructure_faults", {}))
            for row in rows
        },
        **contracts,
    }
    for field in RATE_FIELDS:
        metrics[f"macro_{field}"] = sum(
            float(row["overall"][field]) for row in rows
        ) / len(rows)
    metrics["macro_mean_episode_return"] = sum(
        float(row["overall"]["mean_episode_return"]) for row in rows
    ) / len(rows)
    metrics["macro_mean_episode_length"] = sum(
        float(row["overall"]["mean_episode_length"]) for row in rows
    ) / len(rows)
    return metrics


def _latest_summary(run_dir: Path) -> Path:
    paths = sorted(
        run_dir.glob("summary_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not paths:
        raise FileNotFoundError(f"validation produced no summary in {run_dir}")
    return paths[-1]


def _run_scenario(
    scenario: str,
    *,
    policy: Path,
    update: int,
    validation_root: Path,
    ego_judgement: str,
) -> dict[str, Any]:
    run_dir = validation_root / f"update_{update:03d}" / scenario
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        f"validation_debug.debug_{scenario}",
        "--model",
        str(policy),
        "--episodes-per-origin",
        str(VALIDATION_EPISODES_PER_ORIGIN),
        "--seed",
        str(SEED),
        "--device",
        VALIDATION_DEVICE,
        "--ego-judgement",
        ego_judgement,
        "--no-props",
        "--no-rendering",
        "--live-log-every",
        "0",
        "--run-dir",
        str(run_dir),
    ]
    print(
        f"VALIDATION_SCENARIO_START update={update} scenario={scenario}",
        flush=True,
    )
    completed = subprocess.run(
        command,
        check=False,
        cwd=str(BUNDLE_ROOT),
        env=portable_subprocess_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"validation {scenario} failed with return code {completed.returncode}"
        )
    summary = json.loads(_latest_summary(run_dir).read_text(encoding="utf-8"))
    if Path(str(summary.get("model", ""))).resolve() != policy.resolve():
        raise RuntimeError(f"validation {scenario} summarized the wrong policy path")
    if summary.get("model_sha256") != file_digest(policy):
        raise RuntimeError(f"validation {scenario} summarized the wrong policy bytes")
    print(
        f"VALIDATION_SCENARIO_END update={update} scenario={scenario} "
        f"success={100.0 * float(summary['overall']['success_rate']):.1f}%",
        flush=True,
    )
    return summary


def _run_town(
    scenarios: tuple[str, ...],
    *,
    policy: Path,
    update: int,
    validation_root: Path,
    ego_judgement: str,
) -> list[dict[str, Any]]:
    # The two cases assigned to one Town share one CARLA RPC port and must be
    # evaluated sequentially.  The three Town groups run concurrently.
    return [
        _run_scenario(
            scenario,
            policy=policy,
            update=update,
            validation_root=validation_root,
            ego_judgement=ego_judgement,
        )
        for scenario in scenarios
    ]


def _publish_alias(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{uuid.uuid4().hex}")
    try:
        shutil.copyfile(source, temporary)
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"file": str(target.resolve()), "sha256": file_digest(target)}


def _append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def validate_policy_after_update(
    policy: Path,
    *,
    update: int,
    run_root: Path,
    ego_judgement: str = SWEPT_EGO_JUDGEMENT,
) -> dict[str, Any]:
    if ego_judgement not in VALIDATION_TERMINATION_CONTRACTS:
        raise ValueError(f"unsupported ego judgement: {ego_judgement!r}")
    policy = policy.resolve()
    if not policy.is_file():
        raise FileNotFoundError(policy)
    training_state_path = run_root / "run_state.json"
    if not training_state_path.is_file():
        raise FileNotFoundError(
            f"validation requires the training run_state: {training_state_path}"
        )
    training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
    training_plan = training_state.get("plan") or {}
    training_rollout = training_plan.get("rollout_and_ppo") or {}
    expected_termination = VALIDATION_TERMINATION_CONTRACTS[ego_judgement]
    if training_rollout.get("reward_contract") != REWARD_CONTRACT_VERSION_V2:
        raise RuntimeError("training and validation reward contracts differ")
    if training_rollout.get("termination_contract") != expected_termination:
        raise RuntimeError("training and validation termination contracts differ")
    frozen_sources = {
        str(name).replace("\\", "/"): str(digest)
        for name, digest in (
            training_plan.get("frozen_source_sha256") or {}
        ).items()
    }
    expected_reward_hash = frozen_sources.get("logic/reward_v2.py")
    current_reward_hash = file_digest(BUNDLE_ROOT / "logic" / "reward_v2.py")
    if expected_reward_hash != current_reward_hash:
        raise RuntimeError(
            "reward source changed between training and validation"
        )
    validation_root = run_root / "validation"
    print(
        f"VALIDATION_START update={update} policy={policy} "
        f"episodes={VALIDATION_TOTAL_EPISODES}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                _run_town,
                town.validation_scenarios,
                policy=policy,
                update=update,
                validation_root=validation_root,
                ego_judgement=ego_judgement,
            )
            for town in TOWNS
        ]
        summaries = [item for future in futures for item in future.result()]
    metrics = aggregate_validation_summaries(summaries)
    if metrics["ego_judgement"] != ego_judgement:
        raise RuntimeError("validation worker used the wrong ego judgement")
    if metrics["reward_contract"] != REWARD_CONTRACT_VERSION_V2:
        raise RuntimeError("validation worker used the wrong reward contract")
    if metrics["reward_source_sha256"] != expected_reward_hash:
        raise RuntimeError("validation worker used the wrong reward source")
    if metrics["termination_contract"] != expected_termination:
        raise RuntimeError("validation worker used the wrong termination contract")
    if metrics["model_sha256"] != file_digest(policy):
        raise RuntimeError("validation workers used the wrong policy")

    state_path = validation_root / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"best_model": None, "completed_updates": []}
    )
    previous = state.get("best_model")
    if previous is not None and ego_judgement == CENTER_EGO_JUDGEMENT_V2:
        previous_metrics = previous.get("metrics") or {}
        expected_previous_contract = {
            "ego_judgement": ego_judgement,
            "reward_contract": REWARD_CONTRACT_VERSION_V2,
            "reward_source_sha256": expected_reward_hash,
            "termination_contract": expected_termination,
        }
        if any(
            previous_metrics.get(field) != expected
            for field, expected in expected_previous_contract.items()
        ):
            raise RuntimeError(
                "existing validation best_model belongs to a different contract"
            )
    best_updated = previous is None or validation_selection_key(metrics) > tuple(
        float(value) for value in previous["selection_key"]
    )
    if best_updated:
        alias = _publish_alias(policy, run_root / "models" / VALIDATION_BEST_NAME)
        best = {
            **alias,
            "source_model": str(policy),
            "source_sha256": file_digest(policy),
            "update": int(update),
            "selection_key": list(validation_selection_key(metrics)),
            "metrics": metrics,
        }
    else:
        best = dict(previous)

    result = {
        "update": int(update),
        "policy": str(policy),
        "policy_sha256": file_digest(policy),
        "ego_judgement": ego_judgement,
        "reward_contract": REWARD_CONTRACT_VERSION_V2,
        "reward_source_sha256": expected_reward_hash,
        "termination_contract": expected_termination,
        "best_updated": bool(best_updated),
        "best_model": best,
        "metrics": metrics,
    }
    update_dir = validation_root / f"update_{update:03d}"
    atomic_write_json(update_dir / "summary.json", result)
    completed = [
        item for item in state.get("completed_updates", [])
        if int(item.get("update", -1)) != int(update)
    ]
    completed.append(result)
    completed.sort(key=lambda item: int(item["update"]))
    atomic_write_json(
        state_path,
        {"best_model": best, "completed_updates": completed},
    )
    _append_event(
        run_root / "logs" / "events.jsonl",
        "validation_complete",
        **result,
    )
    print(
        f"VALIDATION_COMPLETE update={update} "
        f"success={100.0 * metrics['macro_success_rate']:.1f}% "
        f"collision={100.0 * metrics['macro_collision_rate']:.1f}% "
        f"violation={100.0 * metrics['macro_workzone_violation_rate']:.1f}% "
        f"timeout={100.0 * metrics['macro_timeout_rate']:.1f}% "
        f"best_update={best['update']} best_updated={best_updated}",
        flush=True,
    )
    return result


__all__ = [
    "aggregate_validation_summaries",
    "validate_policy_after_update",
    "validation_selection_key",
]
