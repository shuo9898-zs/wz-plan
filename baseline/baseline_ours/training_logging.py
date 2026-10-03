"""Readable terminal telemetry and persistent metrics for global PPO training."""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance

from config.scenario_labels import (
    training_display_scenario_id,
    training_display_setting_id,
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else float("nan")


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "n/a"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


class TrainingTelemetry:
    """Own the append-only run log, CSV metrics, and compact terminal output."""

    EPISODE_FIELDS = (
        "timestamp",
        "global_step",
        "town_step",
        "town",
        "scenario",
        "wz",
        "layout",
        "setting_id",
        "origin_index",
        "episode_return",
        "episode_length",
        "progress_reward_total",
        "speed_reward_total",
        "average_speed_mps",
        "step_cost_total",
        "terminal_base_reward",
        "terminal_reward",
        "acc_intervention_steps",
        "acc_intervention_rate",
        "reason",
        "success",
    )
    EPOCH_FIELDS = (
        "timestamp",
        "environment_steps",
        "epoch",
        "epochs_total",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
        "learning_rate",
        "elapsed_s",
    )

    def __init__(self, log_dir: Path, *, resume: bool) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"
        self.run_stamp = self._select_run_stamp(resume=resume)
        if self.run_stamp is None:
            # Compatibility when resuming a run created before timestamped CSVs.
            self.episodes_path = self.log_dir / "episodes.csv"
            self.epochs_path = self.log_dir / "ppo_epochs.csv"
        else:
            self.episodes_path = self.log_dir / f"episodes_{self.run_stamp}.csv"
            self.epochs_path = self.log_dir / f"ppo_epochs_{self.run_stamp}.csv"
        self.summary_path = self.log_dir / "summary.json"
        self.started_at = time.perf_counter()
        self.resume = bool(resume)
        self._ensure_csv(self.episodes_path, self.EPISODE_FIELDS)
        self._ensure_csv(self.epochs_path, self.EPOCH_FIELDS)

    def _select_run_stamp(self, *, resume: bool) -> str | None:
        timestamped = sorted(
            self.log_dir.glob("episodes_*.csv"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if resume and timestamped:
            return timestamped[-1].stem[len("episodes_"):]
        if resume and (self.log_dir / "episodes.csv").exists():
            return None
        return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:19]

    @staticmethod
    def _ensure_csv(path: Path, fields: Sequence[str]) -> None:
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=list(fields)).writeheader()

    @staticmethod
    def banner(title: str, detail: str = "") -> None:
        width = 88
        print("=" * width, flush=True)
        print(f"{title}", flush=True)
        if detail:
            print(detail, flush=True)
        print("=" * width, flush=True)

    def event(self, event: str, **payload: Any) -> None:
        record = {"timestamp": _now(), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    @staticmethod
    def _append_csv(path: Path, fields: Sequence[str], record: Mapping[str, Any]) -> None:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields))
            if write_header:
                writer.writeheader()
            writer.writerow({field: record.get(field, "") for field in fields})

    def episode(self, record: Mapping[str, Any]) -> None:
        self._append_csv(self.episodes_path, self.EPISODE_FIELDS, record)
        self.event("episode_end", **dict(record))

    def train_epoch(
        self,
        record: Mapping[str, Any],
        *,
        console: bool = True,
    ) -> None:
        self._append_csv(self.epochs_path, self.EPOCH_FIELDS, record)
        self.event("ppo_epoch", **dict(record))
        if not console:
            return
        print(
            f"[PPO] e={int(record['epoch']):02d}/{int(record['epochs_total']):02d} "
            f"pi={float(record['policy_loss']):.4f} "
            f"vf={float(record['value_loss']):.3f} "
            f"KL={float(record['approx_kl']):.5f} "
            f"clip={100.0 * float(record['clip_fraction']):.1f}% "
            f"t={float(record['elapsed_s']):.1f}s",
            flush=True,
        )

    def write_summary(self, payload: Mapping[str, Any]) -> None:
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.summary_path)


class RolloutTelemetryCallback(BaseCallback):
    """Track reward/step/context distributions while leaving selection untouched."""

    def __init__(
        self,
        selector: Any,
        telemetry: TrainingTelemetry,
        *,
        town: str,
        target_steps: int,
        every_steps: int,
        step_offset: int = 0,
    ) -> None:
        super().__init__()
        self.selector = selector
        self.telemetry = telemetry
        self.town = town
        self.target_steps = int(target_steps)
        self.every_steps = max(5_000, int(every_steps))
        self.step_offset = int(step_offset)
        self.outcomes: Counter[Tuple[str, str]] = Counter()
        self.episode_records: List[Dict[str, Any]] = []
        self.context_steps: Counter[Tuple[str, str]] = Counter()
        self._returns: Dict[int, float] = defaultdict(float)
        self._lengths: Dict[int, int] = defaultdict(int)
        self._progress_rewards: Dict[int, float] = defaultdict(float)
        self._speed_rewards: Dict[int, float] = defaultdict(float)
        self._step_costs: Dict[int, float] = defaultdict(float)
        self._acc_interventions: Dict[int, int] = defaultdict(int)
        self.infrastructure_faults: Counter[str] = Counter()
        self._console_episode_start = 0
        self._phase_start = time.perf_counter()

    def flush_console_episodes(self) -> None:
        """Print one compact reward/outcome line for each episode batch."""
        records = self.episode_records[self._console_episode_start:]
        if not records:
            return
        first = self._console_episode_start + 1
        last = len(self.episode_records)
        returns = [float(record["episode_return"]) for record in records]
        reasons = Counter(str(record["reason"]) for record in records)
        reason_text = ",".join(
            f"{reason}:{count}" for reason, count in sorted(reasons.items())
        )
        print(
            f"[EP {self.town}] n={first}-{last} "
            f"Rmean={float(np.mean(returns)):.2f} Rlast={returns[-1]:.2f} "
            f"end={reason_text}",
            flush=True,
        )
        self._console_episode_start = last

    def _on_training_end(self) -> None:
        self.flush_console_episodes()

    def record_infrastructure_fault(self, info: Mapping[str, Any]) -> None:
        """Persist a skipped simulator fault without treating it as PPO data."""
        reason = str(info.get("reason", "unknown_infrastructure_fault"))
        self.infrastructure_faults[reason] += 1
        # The VecEnv auto-reset starts a fresh retry of the same coverage
        # ticket.  Drop telemetry accumulated for the aborted attempt exactly
        # as the valid-rollout collector drops its non-MDP transition.
        discarded_steps = int(self._lengths[0])
        discarded_return = float(self._returns[0])
        self.telemetry.event(
            "infrastructure_fault_skipped",
            town=self.town,
            setting_id=str(info.get("setting_id", "unknown/unknown/unknown")),
            origin_index=info.get("origin_index", ""),
            reason=reason,
            exception_type=info.get("exception_type", ""),
            exception=info.get("exception", ""),
            discarded_episode_steps=discarded_steps,
            discarded_episode_return=discarded_return,
        )
        print(
            f"[INFRA {self.town}] skipped=true  "
            f"setting={training_display_setting_id(str(info.get('setting_id', 'unknown')))}  "
            f"reason={reason}",
            flush=True,
        )
        self._returns[0] = 0.0
        self._lengths[0] = 0
        self._progress_rewards[0] = 0.0
        self._speed_rewards[0] = 0.0
        self._step_costs[0] = 0.0
        self._acc_interventions[0] = 0

    @staticmethod
    def _identity(info: Mapping[str, Any]) -> Tuple[str, str, str, str]:
        setting = str(info.get("setting_id", "unknown/unknown/unknown"))
        pieces = setting.split("/")
        scenario = pieces[0] if pieces else "unknown"
        wz = pieces[1] if len(pieces) > 1 else "unknown"
        layout = pieces[2] if len(pieces) > 2 else "unknown"
        return setting, scenario, wz, layout

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for index, info in enumerate(infos):
            reward = float(rewards[index]) if index < len(rewards) else 0.0
            done = bool(dones[index]) if index < len(dones) else False
            setting, scenario, wz, layout = self._identity(info)
            self._returns[index] += reward
            self._lengths[index] += 1
            self._progress_rewards[index] += float(
                info.get("reward_progress_bonus", 0.0)
            )
            self._speed_rewards[index] += float(
                info.get("reward_speed_bonus", 0.0)
            )
            self._step_costs[index] += float(info.get("reward_step_cost", 0.0))
            self._acc_interventions[index] += int(
                bool(info.get("control_acc_intervened", False))
            )
            self.context_steps[(scenario, wz)] += 1
            if not done:
                continue
            reason = str(info.get("reason", "unknown"))
            record = {
                "timestamp": _now(),
                "global_step": self.step_offset + self.n_calls,
                "town_step": self.n_calls,
                "town": self.town,
                "scenario": scenario,
                "wz": wz,
                "layout": layout,
                "setting_id": setting,
                "origin_index": info.get("origin_index", ""),
                "episode_return": self._returns[index],
                "episode_length": self._lengths[index],
                "progress_reward_total": self._progress_rewards[index],
                "speed_reward_total": self._speed_rewards[index],
                "average_speed_mps": float(
                    info.get("reward_average_speed_mps", 0.0)
                ),
                "step_cost_total": self._step_costs[index],
                "terminal_base_reward": float(
                    info.get("reward_terminal_base", reward)
                ),
                "terminal_reward": reward,
                "acc_intervention_steps": self._acc_interventions[index],
                "acc_intervention_rate": (
                    self._acc_interventions[index]
                    / max(self._lengths[index], 1)
                ),
                "reason": reason,
                "success": int(
                    bool(
                        info.get(
                            "success",
                            reason in {"goal_reached", "finish_line_crossed"},
                        )
                    )
                ),
            }
            self.episode_records.append(record)
            self.outcomes[(setting, reason)] += 1
            self.telemetry.episode(record)
            if len(self.episode_records) % 10 == 0:
                self.flush_console_episodes()
            self._returns[index] = 0.0
            self._lengths[index] = 0
            self._progress_rewards[index] = 0.0
            self._speed_rewards[index] = 0.0
            self._step_costs[index] = 0.0
            self._acc_interventions[index] = 0

        if self.n_calls == 1 or self.n_calls % self.every_steps == 0:
            returns = [float(record["episode_return"]) for record in self.episode_records]
            successes = sum(int(record["success"]) for record in self.episode_records)
            elapsed = max(time.perf_counter() - self._phase_start, 1e-9)
            current = (
                training_display_setting_id(str(infos[0].get("setting_id", "n/a")))
                if infos else "n/a"
            )
            samples_per_s = self.n_calls / elapsed
            remaining_steps = max(0, self.target_steps - self.n_calls)
            eta_s = remaining_steps / max(samples_per_s, 1e-9)
            print(
                f"[ROLL {self.town}] {self.n_calls:,}/{self.target_steps:,} "
                f"{100.0 * self.n_calls / self.target_steps:.1f}% "
                f"rate={samples_per_s:.1f}/s ETA={_duration(eta_s)} "
                f"ep={len(returns)} Rmean="
                f"{_fmt(float(np.mean(returns)) if returns else float('nan'), 2)} "
                f"success={100.0 * successes / max(len(returns), 1):.1f}% "
                f"now={current}",
                flush=True,
            )
        return True

    def context_summary(self) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
        for record in self.episode_records:
            grouped[(str(record["scenario"]), str(record["wz"]))].append(record)
        rows = []
        for key in sorted(set(grouped) | set(self.context_steps)):
            records = grouped.get(key, [])
            returns = [float(record["episode_return"]) for record in records]
            success = sum(int(record["success"]) for record in records)
            rows.append(
                {
                    "scenario": key[0],
                    "wz": key[1],
                    "steps": int(self.context_steps[key]),
                    "episodes": len(records),
                    "reward_mean": float(np.mean(returns)) if returns else None,
                    "reward_p50": _percentile(returns, 50) if returns else None,
                    "reward_min": min(returns) if returns else None,
                    "reward_max": max(returns) if returns else None,
                    "success_rate": success / len(records) if records else None,
                    "reasons": dict(
                        Counter(str(record["reason"]) for record in records)
                    ),
                }
            )
        return rows


def print_context_table(rows: Sequence[Mapping[str, Any]]) -> None:
    print("  scenario/wz        steps   episodes     Rmean       R50   success", flush=True)
    print("  ----------------  -------  --------  --------  --------  --------", flush=True)
    for row in rows:
        success = row.get("success_rate")
        print(
            f"  {training_display_scenario_id(str(row['scenario'])) + '/' + str(row['wz']):<16} "
            f"{int(row['steps']):>7,}  {int(row['episodes']):>8,}  "
            f"{_fmt(float(row['reward_mean']) if row['reward_mean'] is not None else float('nan'), 2):>8}  "
            f"{_fmt(float(row['reward_p50']) if row['reward_p50'] is not None else float('nan'), 2):>8}  "
            f"{(100.0 * float(success)) if success is not None else 0.0:>7.1f}%",
            flush=True,
        )


def train_ppo_with_telemetry(model: Any, telemetry: TrainingTelemetry) -> Dict[str, float]:
    """SB3 PPO.train equivalent with one compact real-time line per epoch."""
    model.policy.set_training_mode(True)
    model._update_learning_rate(model.policy.optimizer)
    clip_range = model.clip_range(model._current_progress_remaining)
    clip_range_vf = (
        model.clip_range_vf(model._current_progress_remaining)
        if model.clip_range_vf is not None
        else None
    )
    all_entropy: List[float] = []
    all_policy: List[float] = []
    all_value: List[float] = []
    all_clip: List[float] = []
    all_kl: List[float] = []
    last_loss = float("nan")
    train_start = time.perf_counter()

    for epoch in range(model.n_epochs):
        epoch_start = time.perf_counter()
        epoch_entropy: List[float] = []
        epoch_policy: List[float] = []
        epoch_value: List[float] = []
        epoch_clip: List[float] = []
        epoch_kl: List[float] = []
        for rollout_data in model.rollout_buffer.get(model.batch_size):
            actions = rollout_data.actions
            if isinstance(model.action_space, spaces.Discrete):
                actions = actions.long().flatten()
            if model.use_sde:
                model.policy.reset_noise(model.batch_size)

            values, log_prob, entropy = model.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            values = values.flatten()
            advantages = rollout_data.advantages
            if model.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(
                ratio, 1 - clip_range, 1 + clip_range
            )
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            clip_fraction = th.mean(
                (th.abs(ratio - 1) > clip_range).float()
            ).item()

            if clip_range_vf is None:
                values_pred = values
            else:
                values_pred = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values,
                    -clip_range_vf,
                    clip_range_vf,
                )
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            entropy_loss = (
                -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
            )
            loss = (
                policy_loss
                + model.ent_coef * entropy_loss
                + model.vf_coef * value_loss
            )

            with th.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl = th.mean(
                    (th.exp(log_ratio) - 1) - log_ratio
                ).cpu().item()

            epoch_policy.append(policy_loss.item())
            epoch_value.append(value_loss.item())
            epoch_entropy.append(entropy_loss.item())
            epoch_clip.append(clip_fraction)
            epoch_kl.append(approx_kl)
            last_loss = loss.item()

            model.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()

        model._n_updates += 1
        all_policy.extend(epoch_policy)
        all_value.extend(epoch_value)
        all_entropy.extend(epoch_entropy)
        all_clip.extend(epoch_clip)
        all_kl.extend(epoch_kl)
        epoch_record = {
                "timestamp": _now(),
                "environment_steps": int(model.num_timesteps),
                "epoch": epoch + 1,
                "epochs_total": model.n_epochs,
                "policy_loss": float(np.mean(epoch_policy)),
                "value_loss": float(np.mean(epoch_value)),
                "entropy_loss": float(np.mean(epoch_entropy)),
                "approx_kl": float(np.mean(epoch_kl)),
                "clip_fraction": float(np.mean(epoch_clip)),
                "learning_rate": float(model.lr_schedule(model._current_progress_remaining)),
                "elapsed_s": time.perf_counter() - epoch_start,
            }
        telemetry.train_epoch(
            epoch_record,
            console=(
                epoch == 0
                or (epoch + 1) % 5 == 0
                or epoch + 1 == model.n_epochs
            ),
        )

    explained_var = explained_variance(
        model.rollout_buffer.values.flatten(), model.rollout_buffer.returns.flatten()
    )
    summary = {
        "policy_loss": float(np.mean(all_policy)),
        "value_loss": float(np.mean(all_value)),
        "entropy_loss": float(np.mean(all_entropy)),
        "approx_kl": float(np.mean(all_kl)),
        "clip_fraction": float(np.mean(all_clip)),
        "loss": float(last_loss),
        "explained_variance": float(explained_var),
        "training_seconds": time.perf_counter() - train_start,
    }
    model.logger.record("train/entropy_loss", summary["entropy_loss"])
    model.logger.record("train/policy_gradient_loss", summary["policy_loss"])
    model.logger.record("train/value_loss", summary["value_loss"])
    model.logger.record("train/approx_kl", summary["approx_kl"])
    model.logger.record("train/clip_fraction", summary["clip_fraction"])
    model.logger.record("train/loss", summary["loss"])
    model.logger.record("train/explained_variance", summary["explained_variance"])
    if hasattr(model.policy, "log_std"):
        model.logger.record("train/std", th.exp(model.policy.log_std).mean().item())
    model.logger.record("train/n_updates", model._n_updates, exclude="tensorboard")
    model.logger.record("train/clip_range", clip_range)
    if clip_range_vf is not None:
        model.logger.record("train/clip_range_vf", clip_range_vf)
    return summary


__all__ = [
    "RolloutTelemetryCallback",
    "TrainingTelemetry",
    "print_context_table",
    "train_ppo_with_telemetry",
]
