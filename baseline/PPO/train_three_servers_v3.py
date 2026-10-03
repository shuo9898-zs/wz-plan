"""PPO V3: smaller synchronized buffers with shuffled fixed-setting tickets.

V3 deliberately keeps the V2 environment, observation, encoder, reward,
controller, PPO optimizer and three-server barrier unchanged.  Its only
training changes are:

* each Town fragment is approximately one third of the V2 size;
* every update independently shuffles the fixed ``(setting, origin)`` tickets;
* sampling is without replacement until one ticket sweep is exhausted;
* an infrastructure failure retries the identical ticket;
* the default run uses 60 updates (about 2.048M valid environment steps).

This is sampling-order randomization, not work-zone geometry randomization.
The smaller buffer is allowed to end before a complete 162-unit coverage
round; coverage remains visible in the existing worker descriptors and logs.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import baseline.PPO.train_three_servers_v2 as _v2
from baseline.PPO.global_rollout_checkpoint import atomic_write_json, file_digest
from baseline.PPO.training_logging import TrainingTelemetry
from baseline.PPO.validate_allSwithoneExe import EpisodeTicket
from config.scenario_selector import CoverageSelector


SCHEMA_VERSION = 1
THREE_SERVER_CONTRACT = "ppo_v3_three_fixed_town_servers_small_shuffled_v1"
TASK_SAMPLING_CONTRACT = "fixed_setting_origin_shuffled_prefix_v1"
DEFAULT_TOWN_STEPS = {
    "Town02": 10_800,
    "Town05": 14_334,
    "Town10HD": 9_000,
}
DEFAULT_TOTAL_UPDATES = 60


_ACTIVE_SEQUENCE_SEED: int | None = None
_CONFIGURED = False
_ORIGINAL_WORKER_SPEC = _v2._worker_spec
_ORIGINAL_WORKER_COMMAND = _v2._worker_command
_ORIGINAL_WORKER_MAIN = _v2._worker_main
_ORIGINAL_TRAINING_PLAN = _v2._training_plan
_ORIGINAL_BANNER = TrainingTelemetry.banner


def sequence_seed(master_seed: int, worker_id: int, update_index: int) -> int:
    """Derive a task-order seed independent from policy/environment RNG."""
    return int(master_seed + worker_id * 2_000_033 + update_index * 100_003)


class ShuffledOriginSelectorV3:
    """Yield random, non-repeating setting/origin tickets within each sweep."""

    INFRASTRUCTURE_REASONS = CoverageSelector.INFRASTRUCTURE_REASONS

    def __init__(
        self,
        settings: Sequence[str],
        origin_counts: Mapping[str, int],
        repeats: int = 1,
        *,
        seed: int | None = None,
    ) -> None:
        if not settings:
            raise ValueError("ShuffledOriginSelectorV3 requires at least one setting")
        if repeats < 1:
            raise ValueError("episodes per origin must be positive")
        if len(set(settings)) != len(settings):
            raise ValueError("settings must be unique")
        resolved_seed = _ACTIVE_SEQUENCE_SEED if seed is None else seed
        if resolved_seed is None:
            raise ValueError("V3 selector requires an explicit sequence seed")

        self.settings = list(settings)
        self.repeats = int(repeats)
        self.seed = int(resolved_seed)
        self.origin_counts = {name: int(origin_counts[name]) for name in settings}
        for name, count in self.origin_counts.items():
            if count < 1:
                raise ValueError(f"{name} has no manual origins")

        self._base_tickets = tuple(
            EpisodeTicket(setting, origin_index)
            for setting in self.settings
            for origin_index in range(self.origin_counts[setting])
        )
        self._rng = random.Random(self.seed)
        self._queue: deque[EpisodeTicket] = deque()
        self._completed: Counter[EpisodeTicket] = Counter()
        self._current: EpisodeTicket | None = None
        self._sweeps_started = 0
        for _ in range(self.repeats):
            self._append_sweep()

    def _append_sweep(self) -> None:
        sweep = list(self._base_tickets)
        self._rng.shuffle(sweep)
        self._queue.extend(sweep)
        self._sweeps_started += 1

    def next(self) -> str:
        if not self._queue:
            self._append_sweep()
        self._current = self._queue.popleft()
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

    @property
    def sweeps_started(self) -> int:
        return self._sweeps_started


def build_parser():
    parser = _v2.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        run_root=Path("runs/ppo_v3_small_shuffled_60u"),
        total_updates=DEFAULT_TOTAL_UPDATES,
        town02_steps=DEFAULT_TOWN_STEPS["Town02"],
        town05_steps=DEFAULT_TOWN_STEPS["Town05"],
        town10hd_steps=DEFAULT_TOWN_STEPS["Town10HD"],
    )
    return parser


def _worker_spec_v3(*args, **kwargs):
    spec = _ORIGINAL_WORKER_SPEC(*args, **kwargs)
    namespace = args[0] if args else kwargs.get("args")
    assignment = kwargs["assignment"]
    update_index = int(kwargs["update_index"])
    spec.update(
        task_sampling_contract=TASK_SAMPLING_CONTRACT,
        sequence_seed=sequence_seed(
            int(namespace.seed), int(assignment.worker_id), update_index
        ),
    )
    return spec


def _worker_command_v3(spec_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "baseline.PPO.train_three_servers_v3",
        "--worker-spec",
        str(spec_path),
    ]


def _training_plan_v3(*args, **kwargs):
    plan = _ORIGINAL_TRAINING_PLAN(*args, **kwargs)
    workspace_root = Path(__file__).resolve().parents[2]
    plan["task_sampling"] = {
        "contract": TASK_SAMPLING_CONTRACT,
        "unit": "setting_origin",
        "without_replacement_within_sweep": True,
        "complete_coverage_required_per_update": False,
        "new_seed_per_town_and_update": True,
        "infrastructure_fault_consumes_ticket": False,
    }
    plan["frozen_source_sha256"][
        str(Path(__file__).resolve().relative_to(workspace_root))
    ] = file_digest(Path(__file__).resolve())
    return plan


def _v3_banner(title: str, detail: str = "") -> None:
    _ORIGINAL_BANNER(
        title.replace("PPO V2", "PPO V3"),
        f"{detail} | sampling=shuffled-fixed-tickets" if detail else detail,
    )


def _configure_v3_runtime() -> None:
    """Inject V3 policy-neutral sampling into the isolated V3 process only."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _v2.SCHEMA_VERSION = SCHEMA_VERSION
    _v2.THREE_SERVER_CONTRACT = THREE_SERVER_CONTRACT
    _v2.DEFAULT_TOWN_STEPS = dict(DEFAULT_TOWN_STEPS)
    _v2.OrderedOriginSelector = ShuffledOriginSelectorV3
    _v2._worker_spec = _worker_spec_v3
    _v2._worker_command = _worker_command_v3
    _v2._training_plan = _training_plan_v3
    TrainingTelemetry.banner = staticmethod(_v3_banner)
    _CONFIGURED = True


def _worker_main_v3(spec_path: Path) -> int:
    global _ACTIVE_SEQUENCE_SEED
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("contract") != THREE_SERVER_CONTRACT:
            raise ValueError("V3 worker specification contract mismatch")
        if spec.get("task_sampling_contract") != TASK_SAMPLING_CONTRACT:
            raise ValueError("V3 task-sampling contract mismatch")
        _ACTIVE_SEQUENCE_SEED = int(spec["sequence_seed"])
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"WORKER_ERROR spec={spec_path} error={error}", flush=True)
        return 3

    result = _ORIGINAL_WORKER_MAIN(spec_path)
    if result == 0:
        descriptor_path = Path(spec["descriptor_path"])
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        descriptor.update(
            task_sampling_contract=TASK_SAMPLING_CONTRACT,
            sequence_seed=_ACTIVE_SEQUENCE_SEED,
            coverage_scope="this_update_only",
        )
        atomic_write_json(descriptor_path, descriptor)
    return result


def main(argv: List[str] | None = None) -> int:
    _configure_v3_runtime()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--worker-spec"]:
        if len(arguments) != 2:
            print("ERROR --worker-spec requires exactly one path", flush=True)
            return 2
        return _worker_main_v3(Path(arguments[1]))
    parser = build_parser()
    args = parser.parse_args(arguments)
    _v2._validate_args(parser, args)
    return _v2._run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

