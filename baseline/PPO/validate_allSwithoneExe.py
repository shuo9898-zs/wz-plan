"""Train one PPO on one global all-scenario rollout through one CARLA server.

The CARLA executable keeps listening on one RPC port.  This entry point loads
Town02, Town05, and Town10HD in sequence, rebuilding the single Gym environment
at Town boundaries while retaining one frozen policy during data collection.
PPO is optimized exactly once, after the Town02, Town05, and Town10HD rollout
fragments have been merged into one on-policy buffer.

Episode order is fixed and explicit::

    Town02: s1 -> s6
    Town05: s2 -> s5
    Town10HD: s3 -> s4

Inside each scenario the manifest order is preserved (wz1, wz2, ...; then
layout a, b, c).  Every layout visits origin 0, 1, and 2 before advancing.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from baseline.PPO.global_rollout_checkpoint import (
    GlobalRolloutCheckpoint,
    file_digest,
    load_rollout_buffer,
    policy_digest,
    save_model_atomic,
)
from baseline.PPO.encoder_v2 import (
    ENCODER_ATTENTION_HEADS_V2,
    ENCODER_BRANCH_HIDDEN_DIM_V2,
    ENCODER_CONTRACT_VERSION_V2,
    ENCODER_FEATURES_DIM_V2,
    ENCODER_TOKEN_DIM_V2,
)
from baseline.PPO.training_logging import (
    RolloutTelemetryCallback,
    TrainingTelemetry,
    print_context_table,
    train_ppo_with_telemetry,
)
from baseline.PPO.valid_rollout import collect_valid_rollouts
from baseline.PPO.rollout_plan import (
    RolloutPlan,
    choose_batch_size,
    describe_plan,
    plan_single,
)
from baseline.PPO.runtime_v2 import (
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    V2_DEFAULT_PPO_EPOCHS,
    build_or_load_ppo,
    configure_initial_config,
)
from config.scenario_catalog import (
    iter_settings,
    list_setting_ids,
    load_manifest,
    scenario_root,
)
from config.scenario_config import EgoSpawnPointConfig, ScenarioConfig, load_scenario
from config.scenario_selector import CoverageSelector, ScenarioSelector
from baseline.controllers_v2 import CONTROLLER_CONTRACT_VERSION_V2
from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_DIM_V2,
    DEFAULT_OBSERVATION_SPEC_V2,
    OBSERVATION_CONTRACT_VERSION_V2,
)
from logic.reward_v2 import REWARD_CONTRACT_VERSION_V2
from logic.termination_checker_v2 import TERMINATION_CONTRACT_VERSION_V2
from tools.preflight import validate_scenario
from baseline.PPO.training_config_v2 import DEFAULT_THREE_SERVER_TRAINING_V2


OBSERVATION_CONTRACT_VERSION = OBSERVATION_CONTRACT_VERSION_V2
OBSERVATION_DIM = DEFAULT_OBSERVATION_DIM_V2
REWARD_CONTRACT_VERSION = REWARD_CONTRACT_VERSION_V2


PHASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    DEFAULT_THREE_SERVER_TRAINING_V2.phases
)
DEFAULT_MAP_LOADER = DEFAULT_THREE_SERVER_TRAINING_V2.map_loader
DEFAULT_TOWN_STEPS = {"Town02": 72000, "Town05": 84000, "Town10HD": 60000}


@dataclass(frozen=True)
class EpisodeTicket:
    setting_id: str
    origin_index: int

    @property
    def ticket_id(self) -> str:
        return f"{self.setting_id}#origin{self.origin_index}"


@dataclass(frozen=True)
class TownPhase:
    town: str
    scenarios: Tuple[str, ...]
    settings: Tuple[str, ...]
    configs: Tuple[ScenarioConfig, ...]
    worst_case_steps: int


class OrderedOriginSelector(ScenarioSelector):
    """Cover setting/origin tickets without changing shared selector code."""

    INFRASTRUCTURE_REASONS = CoverageSelector.INFRASTRUCTURE_REASONS

    def __init__(
        self,
        settings: Sequence[str],
        origin_counts: Mapping[str, int],
        repeats: int = 1,
    ) -> None:
        if not settings:
            raise ValueError("OrderedOriginSelector requires at least one setting")
        if repeats < 1:
            raise ValueError("episodes per origin must be positive")
        if len(set(settings)) != len(settings):
            raise ValueError("settings must be unique")

        self.settings = list(settings)
        self.repeats = int(repeats)
        self.origin_counts = {name: int(origin_counts[name]) for name in settings}
        for name, count in self.origin_counts.items():
            if count < 1:
                raise ValueError(f"{name} has no manual origins")

        base_tickets = [
            EpisodeTicket(setting, origin_index)
            for setting in self.settings
            for origin_index in range(self.origin_counts[setting])
        ]
        self._base_tickets = tuple(base_tickets)
        self._queue = deque(
            ticket
            for _ in range(self.repeats)
            for ticket in self._base_tickets
        )
        self._completed: Counter[EpisodeTicket] = Counter()
        self._current: EpisodeTicket | None = None
        self._overflow_index = 0

    def next(self) -> str:
        if self._queue:
            self._current = self._queue.popleft()
        else:
            # SB3 must finish a fixed-size on-policy rollout after coverage.
            # Fill the tail by cycling all tickets in the same stable order.
            self._current = self._base_tickets[
                self._overflow_index % len(self._base_tickets)
            ]
            self._overflow_index += 1
        return self._current.setting_id

    @property
    def current_ticket(self) -> EpisodeTicket:
        if self._current is None:
            raise RuntimeError("No origin ticket is active")
        return self._current

    def on_episode_end(self, info: dict) -> None:
        if self._current is None:
            return
        ticket = self._current
        info.setdefault("origin_index", ticket.origin_index)
        info.setdefault("ticket_id", ticket.ticket_id)
        if info.get("reason") in self.INFRASTRUCTURE_REASONS:
            self._queue.appendleft(ticket)
        elif self._completed[ticket] < self.repeats:
            self._completed[ticket] += 1
        self._current = None

    @property
    def complete(self) -> bool:
        return all(
            self._completed[ticket] >= self.repeats
            for ticket in self._base_tickets
        )

    @property
    def completed_counts(self) -> Dict[str, int]:
        return {
            setting: sum(
                self._completed[EpisodeTicket(setting, origin_index)]
                for origin_index in range(self.origin_counts[setting])
            )
            for setting in self.settings
        }

    @property
    def completed_ticket_counts(self) -> Dict[str, int]:
        return {
            ticket.ticket_id: self._completed[ticket]
            for ticket in self._base_tickets
        }

    @property
    def ticket_count(self) -> int:
        return len(self._base_tickets) * self.repeats


class OriginBinder:
    """Bind one owner-authored origin without modifying the environment API.

    ``CarlaSumoGymEnv`` invokes this object after applying the selected setting
    and before ``engine.reset()``.  Replacing the runtime config's candidate
    tuple with one element preserves the existing OD sampler: its historical
    ``random.choice`` can only select the requested point.
    """

    def __init__(
        self,
        selector: OrderedOriginSelector,
        origins: Mapping[str, Tuple[EgoSpawnPointConfig, ...]],
    ) -> None:
        self.selector = selector
        self.origins = origins
        self.gym_env = None

    def attach(self, gym_env) -> None:
        self.gym_env = gym_env

    def bind(self, setting_id: str) -> None:
        if self.gym_env is None:
            raise RuntimeError("OriginBinder has not been attached to the Gym env")
        ticket = self.selector.current_ticket
        if ticket.setting_id != setting_id:
            raise RuntimeError(
                f"Ticket/config mismatch: {ticket.setting_id} != {setting_id}"
            )
        points = self.origins[setting_id]
        point = points[ticket.origin_index]
        cfg = self.gym_env.engine.cfg
        if cfg.setting_id != setting_id:
            raise RuntimeError(
                f"Engine/config mismatch: {cfg.setting_id} != {setting_id}"
            )
        cfg.origin.spawn_points = (point,)


def _ordered_settings(scenarios: Sequence[str]) -> List[str]:
    settings: List[str] = []
    for scenario in scenarios:
        settings.extend(list_setting_ids(scenario, runnable_only=True))
    return settings


def build_phase_plans(episodes_per_origin: int) -> List[TownPhase]:
    phases: List[TownPhase] = []
    for town, scenarios in PHASES:
        settings = _ordered_settings(scenarios)
        if not settings:
            raise ValueError(f"{town} has no runnable settings")
        configs = tuple(load_scenario(setting) for setting in settings)
        for config in configs:
            if config.carla.town != town:
                raise ValueError(
                    f"{config.setting_id} belongs to {config.carla.town}, expected {town}"
                )
            if len(config.origin.spawn_points) != 3:
                raise ValueError(
                    f"{config.setting_id} must expose exactly three origins, found "
                    f"{len(config.origin.spawn_points)}"
                )
        budget = episodes_per_origin * sum(
            len(config.origin.spawn_points) * config.episode.max_steps
            for config in configs
        )
        phases.append(
            TownPhase(
                town=town,
                scenarios=tuple(scenarios),
                settings=tuple(settings),
                configs=configs,
                worst_case_steps=budget,
            )
        )
    return phases


def print_file_plan(phases: Iterable[TownPhase]) -> None:
    """Print the exact manifest/config/net/route/origin traversal."""
    for phase_index, phase in enumerate(phases, start=1):
        print(
            f"PHASE {phase_index} town={phase.town} "
            f"scenarios={'->'.join(phase.scenarios)} "
            f"settings={len(phase.settings)}",
            flush=True,
        )
        for scenario in phase.scenarios:
            manifest_path = scenario_root(scenario) / "manifest.json"
            manifest = load_manifest(scenario)
            print(
                f"  SCENARIO {scenario} manifest={manifest_path} "
                f"backend={manifest['traffic_backend']}",
                flush=True,
            )
            records = {
                record.setting_id: record for record in iter_settings(scenario)
            }
            for setting in list_setting_ids(scenario, runnable_only=True):
                record = records[setting]
                config = load_scenario(setting)
                origins = ", ".join(
                    f"o{i}=({p.x:.2f},{p.y:.2f},{p.z:.2f},yaw={p.yaw_deg:.1f})"
                    for i, p in enumerate(config.origin.spawn_points)
                )
                network = str(record.network_path) if record.network_path else "CARLA_ONLY"
                route = str(record.route_path) if record.route_path else "CARLA_ONLY"
                print(
                    f"    {setting} config={record.config_path} "
                    f"net={network} route={route} origins=[{origins}]",
                    flush=True,
                )


def print_compact_plan(
    phases: Sequence[TownPhase], town_quotas: Mapping[str, int]
) -> None:
    print("  Town       scenarios   settings   tickets   worst-case   quota", flush=True)
    print("  ---------  ----------  --------   -------   ----------   -------", flush=True)
    for phase in phases:
        tickets = sum(len(config.origin.spawn_points) for config in phase.configs)
        print(
            f"  {phase.town:<9}  {' -> '.join(phase.scenarios):<10}  "
            f"{len(phase.settings):>8}   {tickets:>7}   "
            f"{phase.worst_case_steps:>10,}   {town_quotas[phase.town]:>7,}",
            flush=True,
        )


def _checkpoint_plan_payload(
    phases: Sequence[TownPhase],
    town_quotas: Mapping[str, int],
    *,
    episodes_per_origin: int,
    ppo_epochs: int,
    seed: int,
    batch_size: int = DEFAULT_THREE_SERVER_TRAINING_V2.ppo.batch_size,
    target_kl: float | None = DEFAULT_THREE_SERVER_TRAINING_V2.ppo.target_kl,
) -> Dict[str, object]:
    """Fingerprint the exact rollout order, quotas, and referenced inputs."""
    files: Dict[str, str] = {}
    observation_specs = {
        (
            tuple(DEFAULT_OBSERVATION_SPEC_V2.history_lags),
            DEFAULT_OBSERVATION_SPEC_V2.max_workzone_elements,
            DEFAULT_OBSERVATION_SPEC_V2.max_other_agents,
            DEFAULT_OBSERVATION_SPEC_V2.max_lane_segments,
            DEFAULT_OBSERVATION_SPEC_V2.perception_radius_m,
            DEFAULT_OBSERVATION_SPEC_V2.ego_position_scale_m,
            DEFAULT_OBSERVATION_SPEC_V2.history_distance_scale_m,
            DEFAULT_OBSERVATION_SPEC_V2.max_ego_speed_mps,
            DEFAULT_OBSERVATION_SPEC_V2.lane_length_scale_m,
            DEFAULT_OBSERVATION_SPEC_V2.lane_width_scale_m,
            DEFAULT_OBSERVATION_SPEC_V2.local_id_bits,
            DEFAULT_OBSERVATION_SPEC_V2.missing_actor_ttl_steps,
            DEFAULT_OBSERVATION_SPEC_V2.control_dt_s,
        )
    }
    for phase in phases:
        for scenario in phase.scenarios:
            manifest_path = scenario_root(scenario) / "manifest.json"
            files[str(manifest_path.resolve())] = file_digest(manifest_path)
            for record in iter_settings(scenario):
                for path in (record.config_path, record.network_path, record.route_path):
                    if path is not None:
                        files[str(path.resolve())] = file_digest(path)
    return {
        "phases": [
            {
                "town": phase.town,
                "scenarios": list(phase.scenarios),
                "settings": list(phase.settings),
                "quota_steps": int(town_quotas[phase.town]),
                "worst_case_steps": int(phase.worst_case_steps),
            }
            for phase in phases
        ],
        "episodes_per_origin": int(episodes_per_origin),
        "ppo_epochs": int(ppo_epochs),
        "ppo_batch_size": int(batch_size),
        "ppo_target_kl": None if target_kl is None else float(target_kl),
        "ppo_gamma": PPO_GAMMA,
        "ppo_gae_lambda": PPO_GAE_LAMBDA,
        "ppo_learning_rate": DEFAULT_THREE_SERVER_TRAINING_V2.ppo.learning_rate,
        "ppo_clip_range": DEFAULT_THREE_SERVER_TRAINING_V2.ppo.clip_range,
        "ppo_clip_range_vf": DEFAULT_THREE_SERVER_TRAINING_V2.ppo.clip_range_vf,
        "ppo_normalize_advantage": (
            DEFAULT_THREE_SERVER_TRAINING_V2.ppo.normalize_advantage
        ),
        "ppo_entropy_coefficient": (
            DEFAULT_THREE_SERVER_TRAINING_V2.ppo.entropy_coefficient
        ),
        "ppo_value_function_coefficient": (
            DEFAULT_THREE_SERVER_TRAINING_V2.ppo.value_function_coefficient
        ),
        "ppo_max_gradient_norm": (
            DEFAULT_THREE_SERVER_TRAINING_V2.ppo.max_gradient_norm
        ),
        "observation_contract": OBSERVATION_CONTRACT_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "encoder_contract": ENCODER_CONTRACT_VERSION_V2,
        "encoder_token_dim": ENCODER_TOKEN_DIM_V2,
        "encoder_branch_hidden_dim": ENCODER_BRANCH_HIDDEN_DIM_V2,
        "encoder_features_dim": ENCODER_FEATURES_DIM_V2,
        "encoder_attention_heads": ENCODER_ATTENTION_HEADS_V2,
        "actor_critic_share_encoder": False,
        "observation_normalization": [
            {
                "history_lags": list(spec[0]),
                "max_workzone_elements": spec[1],
                "max_other_agents": spec[2],
                "max_lane_segments": spec[3],
                "perception_radius_m": spec[4],
                "ego_position_scale_m": spec[5],
                "history_distance_scale_m": spec[6],
                "max_ego_speed_mps": spec[7],
                "lane_length_scale_m": spec[8],
                "lane_width_scale_m": spec[9],
                "local_id_bits": spec[10],
                "missing_actor_ttl_steps": spec[11],
                "control_dt_s": spec[12],
            }
            for spec in sorted(observation_specs)
        ],
        "action_contract": CONTROLLER_CONTRACT_VERSION_V2,
        "action_dim": 2,
        "action_max_speed_mps": 13.89,
        "action_max_yaw_rate_deg_s": 30.0,
        "reward_contract": REWARD_CONTRACT_VERSION,
        "termination_contract": TERMINATION_CONTRACT_VERSION_V2,
        "seed": int(seed),
        "input_files": files,
    }


def _outcome_records(outcomes: Counter[tuple[str, str]]) -> List[Dict[str, object]]:
    return [
        {"setting_id": setting, "reason": reason, "count": int(count)}
        for (setting, reason), count in sorted(outcomes.items())
    ]


def _restore_outcomes(
    outcomes: Counter[tuple[str, str]], fragments: Sequence[Mapping[str, object]]
) -> None:
    for fragment in fragments:
        for record in fragment.get("outcomes", []):
            outcomes[(str(record["setting_id"]), str(record["reason"]))] += int(
                record["count"]
            )


def _restore_telemetry(
    fragments: Iterable[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    episodes: List[Dict[str, object]] = []
    contexts: List[Dict[str, object]] = []
    for fragment in fragments:
        episodes.extend(dict(item) for item in fragment.get("episodes", []))
        contexts.extend(dict(item) for item in fragment.get("context_summary", []))
    return episodes, contexts


def _load_checkpoint_buffers(
    checkpoint: GlobalRolloutCheckpoint,
    state: Mapping[str, object],
    model,
) -> List[RolloutBuffer]:
    if policy_digest(model) != state.get("frozen_policy_sha256"):
        raise ValueError("Loaded model does not match the frozen rollout policy")
    buffers = []
    for fragment in sorted(
        state.get("fragments", []), key=lambda item: int(item["phase_index"])
    ):
        if fragment.get("policy_sha256") != state.get("frozen_policy_sha256"):
            raise ValueError("A rollout fragment was collected by another policy")
        buffers.append(
            load_rollout_buffer(
                checkpoint.root / str(fragment["file"]), fragment, model
            )
        )
    return buffers


def _load_map(
    map_loader: Path,
    town: str,
    *,
    host: str,
    carla_port: int,
    rpc_timeout: float,
) -> None:
    if not map_loader.is_file():
        raise FileNotFoundError(f"CARLA map loader not found: {map_loader}")
    command = [
        sys.executable,
        str(map_loader),
        town,
        "--host",
        host,
        "--port",
        str(carla_port),
        "--rpc-timeout",
        str(rpc_timeout),
    ]
    print("MAP_LOAD", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _town_step_quotas(
    phases: Sequence[TownPhase],
    manual_n_steps: int | None,
    town_overrides: Mapping[str, int | None],
    buffer_margin: float | None,
    require_worst_case: bool = True,
) -> Dict[str, int]:
    """Resolve one fixed fragment size per Town and validate coverage capacity."""
    if manual_n_steps is not None:
        if manual_n_steps < 1:
            raise ValueError("--n-steps must be positive")
        if any(value is not None for value in town_overrides.values()):
            raise ValueError("--n-steps cannot be combined with per-Town step options")
        if buffer_margin is not None:
            raise ValueError("--n-steps cannot be combined with --buffer-margin")
        quotas = {phase.town: int(manual_n_steps) for phase in phases}
    elif buffer_margin is not None:
        if any(value is not None for value in town_overrides.values()):
            raise ValueError(
                "--buffer-margin cannot be combined with per-Town step options"
            )
        if not math.isfinite(buffer_margin) or buffer_margin < 1.0:
            raise ValueError("--buffer-margin must be finite and at least 1.0")
        quotas = {
            phase.town: int(math.ceil(phase.worst_case_steps * buffer_margin))
            for phase in phases
        }
    else:
        quotas = {
            phase.town: int(
                town_overrides.get(phase.town)
                or DEFAULT_TOWN_STEPS[phase.town]
            )
            for phase in phases
        }

    for phase in phases:
        quota = quotas[phase.town]
        if require_worst_case and quota < phase.worst_case_steps:
            raise ValueError(
                f"{phase.town} step quota {quota} is below its worst-case "
                f"coverage budget {phase.worst_case_steps}"
            )
    return quotas


def _make_plan(phase: TownPhase, episodes_per_origin: int, n_steps: int) -> RolloutPlan:
    # Each setting has three owner-authored origins.  ``repeats`` includes all
    # three origins and any user-requested repetitions of each origin.
    repeats = 3 * episodes_per_origin
    return plan_single(
        [config.episode.max_steps for config in phase.configs],
        repeats,
        n_steps_override=n_steps,
    )


def _global_plan(
    phases: Sequence[TownPhase],
    episodes_per_origin: int,
    town_quotas: Mapping[str, int],
) -> RolloutPlan:
    """Describe the single PPO buffer assembled from all Town fragments."""
    total_steps = sum(int(town_quotas[phase.town]) for phase in phases)
    return RolloutPlan(
        n_envs=1,
        n_steps_per_env=total_steps,
        buffer_size_steps=total_steps,
        setting_count=sum(len(phase.settings) for phase in phases),
        repeats=3 * int(episodes_per_origin),
        group_step_budgets=(sum(phase.worst_case_steps for phase in phases),),
    )


def _new_rollout_buffer(model, n_steps: int) -> RolloutBuffer:
    """Create an SB3-compatible temporary buffer for one Town."""
    return RolloutBuffer(
        int(n_steps),
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1,
    )


def merge_rollout_buffers(buffers: Sequence[RolloutBuffer]) -> RolloutBuffer:
    """Merge processed Town buffers without leaking GAE across maps."""
    if not buffers:
        raise ValueError("At least one Town rollout buffer is required")
    if any(not buffer.full for buffer in buffers):
        raise ValueError("Only full Town rollout buffers can be merged")

    first = buffers[0]
    merged = RolloutBuffer(
        sum(buffer.buffer_size for buffer in buffers),
        first.observation_space,
        first.action_space,
        device=first.device,
        gamma=first.gamma,
        gae_lambda=first.gae_lambda,
        n_envs=first.n_envs,
    )
    for name in (
        "observations",
        "actions",
        "rewards",
        "returns",
        "episode_starts",
        "values",
        "log_probs",
        "advantages",
    ):
        setattr(
            merged,
            name,
            np.concatenate([getattr(buffer, name) for buffer in buffers], axis=0),
        )
    merged.pos = merged.buffer_size
    merged.full = True
    merged.generator_ready = False
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-per-origin",
        type=int,
        default=DEFAULT_THREE_SERVER_TRAINING_V2.episodes_per_origin,
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Legacy override applying the same fragment size to every Town",
    )
    parser.add_argument("--town02-steps", type=int, default=None)
    parser.add_argument("--town05-steps", type=int, default=None)
    parser.add_argument("--town10hd-steps", type=int, default=None)
    parser.add_argument(
        "--buffer-margin",
        type=float,
        default=None,
        help="Optional multiplier applied separately to each Town worst-case budget",
    )
    parser.add_argument("--ppo-epochs", type=int, default=V2_DEFAULT_PPO_EPOCHS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_THREE_SERVER_TRAINING_V2.ppo.batch_size,
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=DEFAULT_THREE_SERVER_TRAINING_V2.ppo.target_kl,
    )
    parser.add_argument(
        "--estimated-coverage",
        action="store_true",
        help=(
            "Use fixed valid-step rollout boundaries even when the diagnostic "
            "setting/origin repeat target has not completed"
        ),
    )
    parser.add_argument(
        "--append-logs",
        action="store_true",
        help="Append to the newest timestamped telemetry run",
    )
    parser.add_argument(
        "--global-step-offset",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--live-log-every",
        type=int,
        default=DEFAULT_THREE_SERVER_TRAINING_V2.live_log_every_steps,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_THREE_SERVER_TRAINING_V2.device,
        help="PPO optimization device; rollout simulation remains CPU-bound",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None,
        help="Persistent CSV/JSON log directory (default: <output>_logs)",
    )
    parser.add_argument(
        "--verbose-file-plan", action="store_true",
        help="Print every config/net/route/origin path before collection",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_THREE_SERVER_TRAINING_V2.seed)
    parser.add_argument("--carla-host", default=DEFAULT_THREE_SERVER_TRAINING_V2.carla_host)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--sumo-port", type=int, default=8813)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--show-traffic-cones", action="store_true")
    parser.add_argument("--cones-only", action="store_true")
    parser.add_argument("--map-loader", type=Path, default=DEFAULT_MAP_LOADER)
    parser.add_argument("--map-rpc-timeout", type=float, default=60.0)
    parser.add_argument("--teardown-sleep", type=float, default=3.0)
    parser.add_argument("--map-ready-sleep", type=float, default=30.0)
    parser.add_argument("--model", help="Optional PPO .zip checkpoint to continue")
    parser.add_argument("--output", default="runs/allS_one_exe")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Town-boundary rollout checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same frozen-policy rollout from --checkpoint-dir",
    )
    args = parser.parse_args(argv)

    if args.episodes_per_origin < 1:
        parser.error("--episodes-per-origin must be positive")
    if args.teardown_sleep < 0.0 or args.map_ready_sleep < 0.0:
        parser.error("sleep durations must be non-negative")
    if args.map_rpc_timeout <= 0.0:
        parser.error("--map-rpc-timeout must be positive")
    if args.resume and args.model:
        parser.error("--resume and --model are mutually exclusive")
    if args.live_log_every < 1:
        parser.error("--live-log-every must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.target_kl is not None and (
        not math.isfinite(args.target_kl) or args.target_kl <= 0.0
    ):
        parser.error("--target-kl must be finite and positive")
    if args.global_step_offset < 0:
        parser.error("--global-step-offset must be non-negative")

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("ERROR --device cuda requested, but CUDA is unavailable", flush=True)
            return 2

    # Use the same scenario preflight as validate_single; do not add a second,
    # stricter interpretation of owner-authored SUMO routes here.
    for _, scenarios in PHASES:
        for scenario in scenarios:
            result = validate_scenario(scenario)
            if not result.ok:
                for error in result.errors:
                    print(f"ERROR {scenario}: {error}", flush=True)
                return 2

    try:
        phases = build_phase_plans(args.episodes_per_origin)
        town_quotas = _town_step_quotas(
            phases,
            args.n_steps,
            {
                "Town02": args.town02_steps,
                "Town05": args.town05_steps,
                "Town10HD": args.town10hd_steps,
            },
            args.buffer_margin,
            require_worst_case=not args.estimated_coverage,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR {error}", flush=True)
        return 2

    global_plan = _global_plan(phases, args.episodes_per_origin, town_quotas)
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        output_path = Path(args.output)
        checkpoint_dir = output_path.with_name(f"{output_path.name}_checkpoint")
    checkpoint_plan = _checkpoint_plan_payload(
        phases,
        town_quotas,
        episodes_per_origin=args.episodes_per_origin,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        target_kl=args.target_kl,
    )
    checkpoint = GlobalRolloutCheckpoint(checkpoint_dir, checkpoint_plan)
    resume_state = None
    if args.resume:
        try:
            resume_state = checkpoint.load()
        except (OSError, TypeError, ValueError) as error:
            print(f"ERROR cannot resume: {error}", flush=True)
            return 2
        if resume_state["state"] == "iteration_complete":
            print(
                f"RESUME_ALREADY_COMPLETE final_model="
                f"{resume_state['final_model']['file']}",
                flush=True,
            )
            return 0
    elif checkpoint.state_path.exists():
        print(
            f"ERROR checkpoint already exists: {checkpoint.state_path}; "
            f"use --resume or choose another --checkpoint-dir",
            flush=True,
        )
        return 2

    output_path = Path(args.output)
    log_dir = args.log_dir or output_path.with_name(f"{output_path.name}_logs")
    telemetry = TrainingTelemetry(
        log_dir, resume=args.resume or args.append_logs
    )
    TrainingTelemetry.banner(
        "GLOBAL PPO — ONE POLICY, THREE TOWNS, ONE UPDATE",
        f"steps={global_plan.buffer_size_steps:,} | epochs={args.ppo_epochs} | "
        f"gamma={PPO_GAMMA} | lambda={PPO_GAE_LAMBDA} | "
        f"device={args.device} | logs={log_dir}",
    )
    print_compact_plan(phases, town_quotas)
    if args.verbose_file_plan:
        print_file_plan(phases)
    telemetry.event(
        "run_start", resume=args.resume, output=str(output_path),
        total_steps=global_plan.buffer_size_steps, town_quotas=town_quotas,
        ppo_epochs=args.ppo_epochs, requested_device=args.device,
        batch_size=args.batch_size, target_kl=args.target_kl,
        gamma=PPO_GAMMA, gae_lambda=PPO_GAE_LAMBDA,
        observation_contract=OBSERVATION_CONTRACT_VERSION,
        observation_dim=OBSERVATION_DIM,
        reward_contract=REWARD_CONTRACT_VERSION,
    )

    model = None
    vec_env = None
    visualizer = None
    all_outcomes: Counter[tuple[str, str]] = Counter()
    all_episode_records: List[Dict[str, object]] = []
    all_context_rows: List[Dict[str, object]] = []
    town_timings: Dict[str, float] = {}
    town_physical_steps: Dict[str, int] = {}
    town_infrastructure_faults: Dict[str, int] = {}
    collected_buffers: List[RolloutBuffer] = []
    learning_initialized = False
    next_phase_index = int((resume_state or {}).get("next_phase_index", 0))
    effective_model_path = args.model
    if resume_state is not None:
        effective_model_path = str(checkpoint.root / resume_state["model"]["file"])
        _restore_outcomes(all_outcomes, resume_state.get("fragments", []))
        restored_episodes, restored_contexts = _restore_telemetry(
            resume_state.get("fragments", [])
        )
        all_episode_records.extend(restored_episodes)
        all_context_rows.extend(restored_contexts)
        for fragment in resume_state.get("fragments", []):
            town = str(fragment.get("town", "unknown"))
            town_physical_steps[town] = int(
                fragment.get("physical_env_steps", fragment.get("n_steps", 0))
            )
            town_infrastructure_faults[town] = int(
                fragment.get("infrastructure_faults", 0)
            )
    try:
        for phase_index, phase in enumerate(phases):
            if phase_index < next_phase_index:
                print(
                    f"RESUME_SKIP town={phase.town} "
                    f"fragment_already_committed=true",
                    flush=True,
                )
                continue
            if vec_env is not None:
                vec_env.close()
                vec_env = None
                if visualizer is not None:
                    visualizer.close()
                    visualizer = None
                time.sleep(args.teardown_sleep)

            _load_map(
                args.map_loader,
                phase.town,
                host=args.carla_host,
                carla_port=args.carla_port,
                rpc_timeout=args.map_rpc_timeout,
            )
            time.sleep(args.map_ready_sleep)
            phase_started = time.perf_counter()
            TrainingTelemetry.banner(
                f"ROLLOUT {phase_index + 1}/3 — {phase.town}",
                f"scenarios={' -> '.join(phase.scenarios)} | "
                f"quota={town_quotas[phase.town]:,} steps",
            )

            origins = {
                config.setting_id: tuple(config.origin.spawn_points)
                for config in phase.configs
            }
            selector = OrderedOriginSelector(
                phase.settings,
                {setting: len(origins[setting]) for setting in phase.settings},
                repeats=args.episodes_per_origin,
            )
            binder = OriginBinder(selector, origins)
            initial = load_scenario(phase.settings[0])
            configure_initial_config(
                initial,
                carla_port=args.carla_port,
                tm_port=args.tm_port,
                sumo_port=args.sumo_port,
                no_rendering=args.no_rendering,
            )

            if args.show_traffic_cones:
                from tools.show_traffic_cones import WorkZonePropVisualizer

                visualizer = WorkZonePropVisualizer(
                    phase.settings[0],
                    host=args.carla_host,
                    carla_port=args.carla_port,
                    cones_only=args.cones_only,
                    load_town=False,
                    move_spectator=False,
                )

            def episode_setup(setting_id: str) -> None:
                binder.bind(setting_id)
                if visualizer is not None:
                    visualizer.switch(setting_id)

            def make_env():
                from env.gym_wrapper_v2 import CarlaSumoGymEnv

                gym_env = CarlaSumoGymEnv(
                    scenario=phase.settings[0],
                    config=initial,
                    worker_id=0,
                    no_rendering_mode=args.no_rendering,
                    scenario_selector=selector,
                    episode_setup_callback=episode_setup,
                )
                binder.attach(gym_env)
                return gym_env

            vec_env = DummyVecEnv([make_env])
            n_steps = town_quotas[phase.town]
            rollout_plan = _make_plan(phase, args.episodes_per_origin, n_steps)
            print(
                f"TOWN_START town={phase.town} "
                f"scenarios={'->'.join(phase.scenarios)} "
                f"tickets={selector.ticket_count} "
                f"worst_case_steps={phase.worst_case_steps} "
                f"{describe_plan(rollout_plan)}",
                flush=True,
            )

            if model is None:
                model = build_or_load_ppo(
                    vec_env,
                    global_plan,
                    model_path=effective_model_path,
                    ppo_epochs=args.ppo_epochs,
                    seed=args.seed,
                    device=args.device,
                    batch_size=args.batch_size,
                    target_kl=args.target_kl,
                )
                print(f"COMPUTE PPO_device={model.device} rollout_device=CPU/simulators", flush=True)
                if resume_state is not None:
                    collected_buffers.extend(
                        _load_checkpoint_buffers(checkpoint, resume_state, model)
                    )
                    print(
                        f"RESUME_LOADED fragments={len(collected_buffers)} "
                        f"steps={sum(b.buffer_size for b in collected_buffers)}",
                        flush=True,
                    )
            else:
                model.set_env(vec_env, force_reset=True)

            callback = RolloutTelemetryCallback(
                selector, telemetry, town=phase.town,
                target_steps=n_steps, every_steps=args.live_log_every,
                step_offset=sum(
                    town_quotas[prior.town] for prior in phases[:phase_index]
                ) + args.global_step_offset,
            )
            if not learning_initialized:
                _, callback = model._setup_learn(
                    global_plan.buffer_size_steps,
                    callback,
                    reset_num_timesteps=not (args.resume or args.model),
                    tb_log_name="all_scenarios_one_buffer",
                )
                learning_initialized = True
            else:
                # Town changes clear _last_obs. Reset the new environment,
                # while deliberately keeping the policy frozen.
                if model._last_obs is None:
                    model._last_obs = vec_env.reset()
                    model._last_episode_starts = np.ones(
                        (vec_env.num_envs,), dtype=bool
                    )
                callback = model._init_callback(callback)

            town_buffer = _new_rollout_buffer(model, n_steps)
            print(
                f"FROZEN_COLLECTION_START town={phase.town} "
                f"target_steps={n_steps} policy_epoch_updates={model._n_updates}",
                flush=True,
            )
            callback.on_training_start(locals(), globals())
            complete_rollout = collect_valid_rollouts(
                model,
                vec_env,
                callback,
                town_buffer,
                n_rollout_steps=n_steps,
            )
            callback.on_training_end()
            if not complete_rollout or not town_buffer.full:
                raise RuntimeError(f"Incomplete frozen rollout for {phase.town}")
            collected_buffers.append(town_buffer)
            all_outcomes.update(callback.outcomes)
            all_episode_records.extend(callback.episode_records)
            context_rows = callback.context_summary()
            all_context_rows.extend(context_rows)
            town_timings[phase.town] = time.perf_counter() - phase_started
            town_physical_steps[phase.town] = int(town_buffer.physical_env_steps)
            town_infrastructure_faults[phase.town] = int(
                town_buffer.infrastructure_faults
            )
            print(
                f"FROZEN_COLLECTION_END town={phase.town} "
                f"actual_steps={town_buffer.buffer_size} "
                f"physical_steps={town_buffer.physical_env_steps} "
                f"infrastructure_faults={town_buffer.infrastructure_faults} "
                f"coverage_complete={selector.complete} global_collected_steps="
                f"{sum(buffer.buffer_size for buffer in collected_buffers)} "
                f"policy_epoch_updates={model._n_updates}",
                flush=True,
            )

            print_context_table(context_rows)
            telemetry.event(
                "town_complete", town=phase.town, steps=town_buffer.buffer_size,
                physical_steps=town_buffer.physical_env_steps,
                infrastructure_faults=town_buffer.infrastructure_faults,
                elapsed_s=town_timings[phase.town], coverage_complete=selector.complete,
                context_distribution=context_rows,
            )
            if not selector.complete:
                message = (
                    f"{phase.town} did not cover every diagnostic setting/origin "
                    f"repeat target within its fixed {n_steps}-step quota"
                )
                if not args.estimated_coverage:
                    print(f"ERROR {message}", flush=True)
                    return 3
                print(
                    f"COVERAGE_DIAGNOSTIC incomplete=true action=continue "
                    f"detail={message}",
                    flush=True,
                )
                telemetry.event(
                    "coverage_diagnostic",
                    town=phase.town,
                    complete=False,
                    action="continue_fixed_step_rollout",
                    target_repeats=args.episodes_per_origin,
                    completed_ticket_counts=selector.completed_ticket_counts,
                )
            resume_state = checkpoint.commit_town(
                model=model,
                town=phase.town,
                phase_index=phase_index,
                phase_count=len(phases),
                buffer=town_buffer,
                previous_state=resume_state,
                coverage=selector.completed_ticket_counts,
                outcomes=_outcome_records(callback.outcomes),
                episodes=callback.episode_records,
                context_summary=context_rows,
            )
            print(
                f"ROLLOUT_CHECKPOINT_SAVED town={phase.town} "
                f"state={resume_state['state']} path={checkpoint.state_path}",
                flush=True,
            )
        if model is None:
            if resume_state is None or resume_state.get("state") != "ready_to_update":
                raise RuntimeError("No PPO model was created")
            from stable_baselines3 import PPO
            from stable_baselines3.common.utils import configure_logger

            model = PPO.load(
                effective_model_path,
                env=None,
                n_steps=global_plan.buffer_size_steps,
                batch_size=args.batch_size,
                n_epochs=args.ppo_epochs,
                gamma=PPO_GAMMA,
                gae_lambda=PPO_GAE_LAMBDA,
                device=args.device,
                target_kl=args.target_kl,
            )
            model.set_logger(
                configure_logger(
                    model.verbose,
                    model.tensorboard_log,
                    "all_scenarios_one_buffer_resume",
                    False,
                )
            )
            collected_buffers.extend(
                _load_checkpoint_buffers(checkpoint, resume_state, model)
            )
            print(
                f"RESUME_READY_TO_UPDATE fragments={len(collected_buffers)} "
                f"steps={sum(b.buffer_size for b in collected_buffers)}",
                flush=True,
            )

        merged_buffer = merge_rollout_buffers(collected_buffers)
        if merged_buffer.buffer_size != global_plan.buffer_size_steps:
            raise RuntimeError(
                f"Global buffer has {merged_buffer.buffer_size} steps, expected "
                f"{global_plan.buffer_size_steps}"
            )
        if resume_state is None or resume_state.get("state") != "ready_to_update":
            raise RuntimeError("All Town fragments were not checkpointed for update")
        if policy_digest(model) != resume_state.get("frozen_policy_sha256"):
            raise RuntimeError("Policy changed before the global PPO update")
        model.rollout_buffer = merged_buffer
        model.n_steps = merged_buffer.buffer_size
        model.batch_size = args.batch_size
        model._total_timesteps = model.num_timesteps
        model._update_current_progress_remaining(
            model.num_timesteps,
            model._total_timesteps,
        )
        TrainingTelemetry.banner(
            "PPO OPTIMIZATION",
            f"samples={merged_buffer.buffer_size:,} | batch={model.batch_size} | "
            f"epochs={model.n_epochs} | device={model.device}",
        )
        updates_before = model._n_updates
        train_summary = train_ppo_with_telemetry(model, telemetry)
        model.logger.dump(step=model.num_timesteps)
        print(
            f"GLOBAL_PPO_UPDATE_END epoch_updates="
            f"{model._n_updates - updates_before}",
            flush=True,
        )
        final_model_info = save_model_atomic(model, Path(args.output))
        final_model_path = Path(args.output).with_suffix(".zip")
        resume_state = checkpoint.mark_complete(resume_state, final_model_path)
        print(
            f"FINAL_CHECKPOINT_SAVED file={final_model_info['file']} "
            f"resume_state={checkpoint.state_path}",
            flush=True,
        )
        returns = [float(record["episode_return"]) for record in all_episode_records]
        reward_summary = {
            "count": len(returns),
            "mean": float(np.mean(returns)) if returns else None,
            "std": float(np.std(returns)) if returns else None,
            "min": min(returns) if returns else None,
            "p25": float(np.percentile(returns, 25)) if returns else None,
            "p50": float(np.percentile(returns, 50)) if returns else None,
            "p75": float(np.percentile(returns, 75)) if returns else None,
            "max": max(returns) if returns else None,
        }
        summary = {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "output": str(final_model_path),
            "total_steps": merged_buffer.buffer_size,
            "ppo_device": str(model.device),
            "ppo_epochs": model.n_epochs,
            "ppo_gamma": model.gamma,
            "ppo_gae_lambda": model.gae_lambda,
            "observation_contract": OBSERVATION_CONTRACT_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "reward_contract": REWARD_CONTRACT_VERSION,
            "batch_size": model.batch_size,
            "town_collection_seconds": town_timings,
            "town_physical_env_steps": town_physical_steps,
            "town_infrastructure_faults": town_infrastructure_faults,
            "reward_distribution": reward_summary,
            "training": train_summary,
            "scenario_wz_distribution": all_context_rows,
            "outcomes": _outcome_records(all_outcomes),
            "wall_seconds": time.perf_counter() - telemetry.started_at,
        }
        telemetry.write_summary(summary)
        telemetry.event("run_complete", **summary)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR {error}", flush=True)
        return 2
    finally:
        if vec_env is not None:
            vec_env.close()
        if visualizer is not None:
            visualizer.close()

    TrainingTelemetry.banner(
        "TRAINING COMPLETE",
        f"model={Path(args.output).with_suffix('.zip')} | logs={log_dir} | "
        f"wall={time.perf_counter() - telemetry.started_at:.1f}s",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
