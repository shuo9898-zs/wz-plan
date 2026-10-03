"""
CarlaSumoEnv — six-scenario CARLA-SUMO RL environment (10 Hz closed-loop)
============================================================================

Closed-loop step order for SUMO-backed scenarios (spec §4 / §8):
    1.  build_observation()        ← previous state for the policy
    2.  action = policy(obs)       ← external caller
    3.  apply_ego_control(action)
    4.  update_carla_pedestrians()
    5.  carla_world.tick()         ← CARLA advances dt = 0.1 s
    6.  ego_proxy.sync(ego)        ← CARLA Ego → SUMO proxy
    7.  traci.simulationStep()     ← SUMO advances dt = 0.1 s
    8.  bg_traffic.sync()          ← SUMO background vehicles → CARLA
    9.  check_termination()
       → return obs, reward, terminated, truncated, info

S4 is deliberately CARLA-only. Its step stops after CARLA advances and then
checks termination; it never launches SUMO and never creates a pedestrian
proxy. Its three jaywalkers are controlled directly by CARLA WalkerControl.

Actor ownership (spec §3):
    CARLA-controlled : Ego, Pedestrians
    SUMO-controlled  : Background vehicles

Coordinate rule (spec §4 note):
    Use CarlaSumoCoordinateBridge for all CARLA ↔ SUMO transforms.
    Never copy x, y, or yaw directly between the two simulators.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import time
from typing import Any, Callable

import carla
import numpy as np
import traci

# ── resolve package root so imports work regardless of CWD ─────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.scenario_config import ScenarioConfig, load_scenario
from env.agent_history       import AgentPoint, TrackedAgentEncoder
from env.corridor_heading     import CorridorProgressTracker
from env.observation_v2       import (
    EGO_STATE_DIM,
    OBSERVATION_DIM,
    encode_ego_state,
)
from logic.od_sampler        import OriginDestinationSampler
from logic.reward            import BoundedProgressReward, RewardWeights, compute_reward
from logic.termination_checker import EpisodeTerminationChecker
from scenarios.S4_Town10HD_Jaywalker.jaywalker_controller import JaywalkerController
from sync.bridge             import CarlaSumoCoordinateBridge
from sync.ego_proxy          import EgoProxySynchronizer
# PedestrianProxySynchronizer is intentionally disconnected.  S4 is
# CARLA-only, so its jaywalker must not be mirrored into SUMO.
# from sync.pedestrian_proxy import PedestrianProxySynchronizer
from sync.background_traffic import BackgroundTrafficSynchronizer, MirrorSpawnExhaustedError
from monitoring.runtime_monitor import (
    RuntimeMonitor,
    close_process_logging,
    configure_process_logging,
    pid_exists,
    pid_listening_on_port,
    process_identity_for_pid,
    terminate_pid,
)

logger = logging.getLogger(__name__)

# ── action indices ──────────────────────────────────────────────────────────
_IDX_STEER    = 0   # ∈ [−1, +1]
_IDX_THROTTLE = 1   # ∈ [ 0, +1]
_IDX_BRAKE    = 2   # ∈ [ 0, +1]

# ── observation dimension ────────────────────────────────────────────────────
# 0-6    normalized ego/controller state (no world pose, route, or OD)
# 7-70   8 stable cone/vehicle/walker slots in the ego frame
_MAX_OBS_AGENTS = 8
_OBS_DIM = OBSERVATION_DIM
assert _OBS_DIM == EGO_STATE_DIM + (
    _MAX_OBS_AGENTS * TrackedAgentEncoder.AGENT_DIM
)

# ── reward constants (simple, tunable) ───────────────────────────────────────

_OWNED_ROLE_NAMES = {
    "ego",             # current explicit training ego role
    "hero",            # legacy runs used this role
    "sumo_background", # SUMO-to-CARLA mirror actors
    "episode_sensor",
    "episode_pedestrian",
}
_SUMO_TRANSPORT_EXCEPTIONS = (
    traci.exceptions.FatalTraCIError,
    traci.exceptions.TraCIException,
    ConnectionError,
    BrokenPipeError,
    OSError,
)


class EpisodeInitializationError(RuntimeError):
    """A recoverable failure while building one episode."""


class SpawnExhaustedError(EpisodeInitializationError):
    """Every collision-only actor-spawn attempt for an episode was occupied."""


class CarlaRuntimeFault(RuntimeError):
    """CARLA RPC/tick failure that must not escape a vector-env worker."""


class SumoRuntimeFault(RuntimeError):
    """SUMO/TraCI transport failure that requires a fresh SUMO process."""


class CarlaSumoEnv:
    """
    CARLA-SUMO co-simulation environment for RL training.

    Usage
    -----
    env = CarlaSumoEnv("wz1")     # or "wz2" / "wz3" — see config/scenarios/
    env.connect()                 # once per process

    obs = env.reset(mode="train")
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()

    Parameters
    ----------
    scenario : str
        Name of the scenario JSON in config/scenarios/ (default "wz1").
        Ignored if `config` is given explicitly.
    config : ScenarioConfig | None
        Pass a custom config to bypass the JSON loader entirely.
    worker_id : int
        Numeric identity of this worker (default 0) — used only for logging
        and to namespace deterministic background-vehicle SUMO IDs
        (`w{worker_id}_ep{episode_id}_bg{n}`) so two workers can never
        generate the same vehicle ID even though each owns its own
        independent CARLA server / SUMO process / ports.
    """

    def __init__(self, scenario: str = "wz1", config: ScenarioConfig | None = None,
                 worker_id: int = 0, no_rendering_mode: bool | None = None) -> None:
        self.cfg = config or load_scenario(scenario)
        self._scenario_name = scenario
        self._worker_id      = worker_id
        self._no_rendering_mode = no_rendering_mode
        self._observation_base_heading_provider: (
            Callable[[Any], float] | None
        ) = None
        self._agent_encoder = self._new_agent_encoder()

        # The CARLA server this worker attaches to.  For a single-EXE
        # single-map-rotate run every scenario must share this same server
        # (and port), even though each scenario JSON declares its own
        # carla.port/tm_port.  Remember the first config's server identity and
        # force it onto every switched scenario so the worker never tries to
        # attach to a different port that has no EXE behind it.
        self._fixed_carla_port: int = self.cfg.carla.port
        self._fixed_carla_tm_port: int = self.cfg.carla.tm_port
        self._fixed_carla_host: str = self.cfg.carla.host
        self._fixed_sumo_port: int | None = self.cfg.sumo.port if self.cfg.sumo else None

        # CARLA handles
        self._client: carla.Client | None      = None
        self._world:  carla.World  | None      = None
        self._cmap:   carla.Map    | None      = None
        self._ego:    carla.Actor  | None      = None
        self._pedestrians: list[carla.Actor]   = []
        self._jaywalker_controller: JaywalkerController | None = None

        # Synchronisation components
        self._bridge:      CarlaSumoCoordinateBridge   | None = None
        self._ego_proxy:   EgoProxySynchronizer         | None = None
        # self._ped_proxy: PedestrianProxySynchronizer | None = None
        self._bg_traffic:  BackgroundTrafficSynchronizer| None = None
        self._termination: EpisodeTerminationChecker    | None = None
        self._od_sampler:  OriginDestinationSampler     | None = None

        self._destination_transform: carla.Transform | None = None
        self._episode_step   = 0
        self._episode_id     = 0
        self._prev_dist_to_goal: float | None = None
        self._corridor_progress_tracker: CorridorProgressTracker | None = None
        self._prev_corridor_progress: float | None = None
        self._bounded_progress_reward: BoundedProgressReward | None = None
        self._reward_progress_source = "uninitialized"
        self._reported_reward_fallback_settings: set[str] = set()
        self._sumo_running   = False
        self._carla_connected = False
        self._episode_actors: dict[int, dict[str, str | int]] = {}
        self._last_observation = np.zeros(_OBS_DIM, dtype=np.float32)
        self._observation_previous_speed_mps: float | None = None
        self._observation_previous_yaw_deg: float | None = None
        self._observation_previous_action = np.zeros(2, dtype=np.float32)
        # Curved work-zone polygon (None when only the legacy AABB is
        # configured).  Built once after CARLA map is ready; stable across
        # episodes.  Used for forbidden_polygon / safe_corridor geometry modes.
        self._wz_polygon = None
        # Polygon construction walks CARLA waypoints and is much more expensive
        # than selecting a cached Shapely result.  A worker commonly rotates
        # through the same Town/settings, so retain one result per setting.
        # The cache is instance-local because CARLA waypoint objects must not be
        # shared across servers/workers.
        self._wz_polygon_cache: dict[tuple[str, str], Any] = {}
        self._wz_polygon_cache_map_name: str | None = None
        self._last_monitor_actor_refresh = 0.0
        self._last_monitor_pid_refresh = 0.0
        self._cached_external_carla_pid: int | None = None
        self._observed_external_carla_pid: int | None = None
        self._observed_external_carla_command: list[str] | None = None
        self._observed_external_carla_create_time: float | None = None
        # A captured external command is launchable only after this worker has
        # safely stopped its verified predecessor (or confirmed it exited).
        self._captured_external_launch_authorized = False
        self._last_traci_command: dict[str, Any] | None = None
        self._initialization_failures = 0

        # This worker's persistent SUMO process/connection. Owned entirely
        # by this instance — created once (first reset) and reused across
        # episodes via conn.load(); only killed + relaunched if found dead.
        self._sumo_proc:        subprocess.Popen | None = None
        self._sumo_stderr_file: Any             | None = None
        self._sumo_stderr_path: str             | None = None
        self._conn:             traci.connection.Connection | None = None
        self._sumo_label = f"w{worker_id}_{scenario.replace('/', '_')}"
        self._sumo_generation = 0
        self._last_sumo_returncode: int | None = None

        # CARLA is often launched externally. A healthy initial attachment
        # captures the exact process serving this worker's port; recovery
        # validates it again before restarting only that server. An explicit
        # server_command is the deterministic fallback.
        self._carla_proc: subprocess.Popen | None = None
        self._carla_log_file: Any | None = None
        self._carla_log_path: str | None = None
        self._carla_generation = 0
        self._carla_restart_count = 0

        log_dir = os.path.join(_PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._worker_log_path = os.path.join(log_dir, f"{self._sumo_label}_worker.log")
        configure_process_logging(self._worker_log_path)
        self._monitor = RuntimeMonitor(
            log_path=os.path.join(log_dir, f"{self._sumo_label}_monitor.jsonl"),
            worker=worker_id,
            scenario=scenario,
            interval_s=5.0,
            # One worker is enough to sample global GPU state in a multi-worker run.
            include_gpu=(worker_id == 0),
        )
        self._update_monitor_state(refresh_actors=False)

    # ================================================================== #
    #  Lifecycle                                                           #
    # ================================================================== #

    def connect(self) -> None:
        """
        Connect to CARLA and configure synchronous mode.
        Call once per process before the first reset().
        """
        try:
            self._attach_carla()
            self._require_carla_restart_source()
            return
        except Exception as first_error:
            self._monitor.record_exception("carla_connect", first_error)
            self._carla_connected = False
            logger.warning(
                "Initial CARLA attachment failed | worker=%d scenario=%s port=%d error=%s",
                self._worker_id, self._scenario_name, self.cfg.carla.port, first_error,
            )

        if not self._format_carla_command():
            # A healthy externally started server is captured on the initial
            # attach and can then be restarted only by this worker.  If that
            # capture was unavailable, do not guess a PID or executable.
            raise CarlaRuntimeFault(
                "CARLA is unavailable and this worker has neither a configured server_command "
                "nor a captured command for the server bound to this worker's port; "
                "will retry the connection without terminating the worker."
            )

        self._launch_carla_process()
        deadline = time.monotonic() + self.cfg.carla.server_start_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._attach_carla()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(self.cfg.carla.server_retry_interval_s)
        raise CarlaRuntimeFault(
            f"Worker-owned CARLA did not become ready on port {self.cfg.carla.port} "
            f"within {self.cfg.carla.server_start_timeout_s:.0f}s (last error: {last_error})"
        )

    def _attach_carla(self) -> None:
        """Attach to an already-running CARLA server and configure this world."""
        cfg = self.cfg.carla
        logger.info("Connecting to CARLA %s:%d ...", cfg.host, cfg.port)
        client = carla.Client(cfg.host, cfg.port)
        client.set_timeout(cfg.timeout)

        world = client.get_world()
        current_map = world.get_map().name.split("/")[-1]
        # CARLA appends an "_Opt" suffix to optimized map names (e.g.
        # "Town02_Opt", "Town10HD_Opt").  Strip it so "Town02_Opt" compares
        # equal to the configured "Town02" and we don't trigger a full
        # load_world() map reload on every reconnect.
        if current_map.endswith("_Opt"):
            current_map = current_map[:-4]
        map_was_reloaded = current_map != cfg.town
        if map_was_reloaded:
            logger.info("CARLA has '%s' loaded, switching to '%s' ...", current_map, cfg.town)
            world = client.load_world(cfg.town)

        attached_map = _canonical_town_name(world.get_map().name.split("/")[-1])
        cached_map = getattr(self, "_wz_polygon_cache_map_name", None)
        if map_was_reloaded or (cached_map is not None and cached_map != attached_map):
            self._invalidate_wz_polygon_cache(
                reason=f"map_change:{cached_map or 'none'}->{attached_map}"
            )
        self._wz_polygon_cache_map_name = attached_map

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg.fixed_delta_s
        if hasattr(settings, "no_rendering_mode"):
            settings.no_rendering_mode = self._effective_no_rendering_mode()
        world.apply_settings(settings)

        # Traffic Manager must also be synchronous so it doesn't advance on its own.
        # Port must differ across scenarios so parallel CARLA instances don't share a TM.
        tm = client.get_trafficmanager(cfg.tm_port)
        tm.set_synchronous_mode(True)

        self._client = client
        self._world = world
        self._cmap = world.get_map()
        self._carla_connected = True
        self._capture_external_carla_restart_command()
        self._monitor.record_event(
            "carla_connected",
            port=cfg.port,
            map=cfg.town,
            no_rendering_mode=self._effective_no_rendering_mode(),
        )
        self._update_monitor_state(refresh_actors=True)
        logger.info("CARLA connected (sync, dt=%.2f s, map=%s, no_rendering=%s)",
                    cfg.fixed_delta_s, cfg.town, self._effective_no_rendering_mode())

    def _effective_no_rendering_mode(self) -> bool:
        return self.cfg.carla.no_rendering_mode if self._no_rendering_mode is None else self._no_rendering_mode

    def _capture_external_carla_restart_command(self) -> None:
        """Capture only the command of the CARLA process serving this worker's port."""
        if self._carla_proc is not None:
            return
        pid = pid_listening_on_port(self.cfg.carla.port)
        identity = process_identity_for_pid(pid)
        command = identity["command"] if identity else None
        executable = os.path.basename(command[0]).lower() if command else ""
        if pid is None or not command or "carla" not in executable:
            self._monitor.record_event(
                "external_carla_restart_capture_unavailable",
                port=self.cfg.carla.port,
                pid=pid,
                executable=executable or None,
            )
            return
        self._observed_external_carla_pid = pid
        self._observed_external_carla_command = command
        self._observed_external_carla_create_time = identity["create_time"]
        self._cached_external_carla_pid = pid
        self._captured_external_launch_authorized = False
        self._monitor.record_event(
            "external_carla_restart_command_captured",
            port=self.cfg.carla.port,
            pid=pid,
            command=command,
            create_time=identity["create_time"],
        )

    def _require_carla_restart_source(self) -> None:
        """Record whether this worker can restart an externally managed CARLA server.

        Process inspection is optional (for example, ``psutil`` may not be
        installed in the CARLA conda environment). A healthy server must still
        be usable in that case. If it later becomes unhealthy, recovery first
        attempts a connection rebuild and keeps the worker alive; automatic
        server restart requires either PID capture or ``server_command``.
        """
        if self.cfg.carla.server_command or self._observed_external_carla_command:
            return
        self._monitor.record_event("carla_restart_source_missing", port=self.cfg.carla.port)
        logger.warning(
            "CARLA connected but automatic server restart is unavailable | worker=%d "
            "scenario=%s port=%d. Configure carla.server_command or install psutil "
            "for PID capture; connection-only recovery remains enabled.",
            self._worker_id, self._scenario_name, self.cfg.carla.port,
        )

    def reset(self, mode: str = "train") -> np.ndarray:
        """
        Reset the environment for a new episode.

        All failure boundaries (spawn exhaustion, CARLA timeout, SUMO crash,
        orphan-actor leakage) are handled inside the resilient lifecycle:
        the worker process survives and simply retries a fresh episode.
        """
        return self._resilient_reset(mode)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the environment by one 10 Hz step.

        Any recoverable fault (CARLA tick timeout, SUMO transport error,
        mirror-spawn collision, unexpected runtime exception) is caught and
        returned as a truncated episode.  SB3 will then invoke reset(),
        which re-initializes the episode in the same worker process.
        """
        return self._resilient_step(action)

    def close(self) -> None:
        """Release all CARLA and SUMO resources, including the monitor thread."""
        self._resilient_close()

    def apply_scenario(self, name: str) -> None:
        """Switch this environment to a different work-zone scenario.

        Replaces ``self.cfg`` and ``self._scenario_name`` with the named
        scenario's config and resets episode-local state.  When both settings
        use SUMO on the same CARLA Town, the current SUMO process and TraCI
        connection stay alive; the next ``reset()`` uses ``conn.load()`` to
        load the selected WZ's route file into that same process.

        Because every work-zone scenario in this project shares the same
        CARLA map (``carla.town``), the existing CARLA connection is reused
        as-is — no reconnect and no ``load_world`` is needed.  For scenarios
        that *did* target a different town this method's caller is expected
        to coordinate a CARLA ``load_world`` separately.
        """
        if name == self._scenario_name:
            return

        next_cfg = load_scenario(name)
        current_town = _canonical_town_name(self.cfg.carla.town)
        next_town = _canonical_town_name(next_cfg.carla.town)
        if next_town != current_town:
            raise ValueError(
                f"Cannot switch one CARLA worker from {self.cfg.carla.town} "
                f"({self.cfg.setting_id}) to {next_cfg.carla.town} "
                f"({next_cfg.setting_id}). Group settings by Town and use a "
                "different CARLA server/worker for each Town."
            )

        current_uses_sumo = self.cfg.uses_sumo
        next_uses_sumo = next_cfg.uses_sumo

        # Clean the old episode while its synchronizers still have their live
        # TraCI connection.  Closing SUMO first makes ego-proxy cleanup send a
        # command through an already-closed connection.
        try:
            self._episode_cleanup()
        except Exception as exc:
            self._monitor.record_exception("apply_scenario_cleanup", exc, scenario=name)

        # A SUMO-backed setting can switch net/route files with TraCI load()
        # without restarting the OS process.  Only tear SUMO down when the next
        # setting is deliberately CARLA-only.  A CARLA-only -> SUMO switch has
        # no process yet and will launch normally in _resilient_reload_sumo().
        if current_uses_sumo and not next_uses_sumo:
            try:
                self._terminate_sumo(reason=f"switch_to_carla_only_{name}")
            except Exception as exc:
                self._monitor.record_exception("apply_scenario_terminate", exc, scenario=name)

        # Swap config and reset episode-local state that is rebuilt on reset.
        self.cfg = next_cfg
        # In a single-EXE rotation every scenario must share the server this
        # worker first attached to; otherwise switching to a scenario whose
        # JSON declares a different port would make us try to attach to a
        # server that isn't running.
        self.cfg.carla.port = self._fixed_carla_port
        self.cfg.carla.tm_port = self._fixed_carla_tm_port
        self.cfg.carla.host = self._fixed_carla_host
        if self.cfg.sumo is not None and self._fixed_sumo_port is not None:
            self.cfg.sumo.port = self._fixed_sumo_port
        self._scenario_name = name
        self._agent_encoder = self._new_agent_encoder()
        self._jaywalker_controller = None
        self._od_sampler = None
        self._destination_transform = None
        self._episode_step = 0
        self._corridor_progress_tracker = None
        self._prev_corridor_progress = None
        self._bounded_progress_reward = None
        self._reward_progress_source = "uninitialized"
        # conn.load() preserves the TraCI connection but the selected net and
        # route definitions may have changed. Rebuild all synchronizers after
        # the load so their bridge/config always belong to the new setting.
        self._bridge = None
        self._ego_proxy = None
        # self._ped_proxy = None  # pedestrian proxy disabled
        self._bg_traffic = None
        self._termination = None
        # Select the next setting's cached geometry (or build it) on reset.
        self._wz_polygon = None
        self._observation_previous_speed_mps = None
        self._observation_previous_yaw_deg = None
        self._observation_previous_action = np.zeros(2, dtype=np.float32)


    # ================================================================== #
    #  Internal — episode lifecycle                                        #
    # ================================================================== #

    def _episode_cleanup(self) -> None:
        """Destroy known and orphaned actors, then verify the actor set is clean.

        Targets every CARLA actor whose role_name matches the owned set
        (ego/hero/sumo_background/episode_sensor/episode_pedestrian). Uses
        batch commands and re-queries the world until no owned actors remain,
        raising CarlaRuntimeFault if cleanup cannot be verified.
        """
        for label, cleanup in (
            ("collision_sensor", lambda: self._termination.destroy() if self._termination else None),
            ("jaywalker", lambda: self._jaywalker_controller.destroy() if self._jaywalker_controller else None),
            ("ego_proxy", lambda: self._ego_proxy.destroy() if self._ego_proxy else None),
            # ("pedestrian_proxy", lambda: self._ped_proxy.destroy() if self._ped_proxy else None),
            ("background_mirrors", lambda: self._bg_traffic.destroy_all() if self._bg_traffic else None),
        ):
            try:
                cleanup()
            except Exception as exc:
                self._monitor.record_exception(f"episode_cleanup_{label}", exc)

        # The registry can include actors whose Python handles were lost; the
        # role scan catches actors leaked by an earlier worker episode.  A
        # destroy command is not considered successful until a re-query says
        # every owned actor is gone.
        # Component cleanup unregisters actors it already destroyed, although
        # the current synchronous frame's role scan may still return their old
        # IDs.  A repeated destroy for such an ID is harmless; the committed
        # snapshot below, rather than that stale handle, decides what remains.
        targets = set(self._episode_actors) | self._find_owned_actor_ids()
        remaining: set[int] = set()
        for cleanup_attempt in range(1, 4):
            targets |= self._find_owned_actor_ids()
            self._destroy_actor_ids(targets)

            # CARLA's Python Actor handles and get_actors() result describe the
            # current synchronous frame.  Immediately after actor.destroy() or
            # apply_batch_sync(DestroyActor), that frame can still report the
            # actor as alive even though the server has already accepted its
            # destruction.  Advance one frame first, then verify against the
            # new WorldSnapshot.  Verifying before this tick was the source of
            # one false cleanup failure (and CARLA reconnect) per episode.
            self._world_tick("cleanup_verification")
            remaining = self._verify_actor_removal(targets)
            if not remaining:
                break
            self._monitor.record_event(
                "actor_cleanup_retry",
                attempt=cleanup_attempt,
                actor_ids=sorted(remaining),
            )
            # Never retain the original target set.  It may contain IDs that
            # the component destructors already removed and that only remain
            # visible through stale Python handles.  Retry the snapshot-proven
            # live actors, plus any newly discovered owned orphan.
            targets = remaining

        if remaining:
            self._monitor.record_event("actor_cleanup_incomplete", actor_ids=sorted(remaining))
            self._episode_actors = {
                actor_id: meta for actor_id, meta in self._episode_actors.items() if actor_id in remaining
            }
            self._termination = None
            self._pedestrians.clear()
            self._ego = None
            self._destination_transform = None
            self._corridor_progress_tracker = None
            self._prev_corridor_progress = None
            self._bounded_progress_reward = None
            self._reward_progress_source = "uninitialized"
            raise CarlaRuntimeFault(
                f"Actor cleanup could not verify removal after 3 attempts: {sorted(remaining)}"
            )

        self._monitor.record_event("actor_cleanup_verified", destroyed_actor_ids=sorted(targets))

        self._episode_actors = {
            actor_id: meta for actor_id, meta in self._episode_actors.items() if actor_id in remaining
        }
        self._termination = None
        self._jaywalker_controller = None
        self._pedestrians.clear()
        self._ego = None
        self._destination_transform = None
        self._corridor_progress_tracker = None
        self._prev_corridor_progress = None
        self._bounded_progress_reward = None
        self._reward_progress_source = "uninitialized"

    def _sumo_option_args(self) -> list[str]:
        """SUMO CLI option list (net/route/collision/etc.), shared by both
        the initial process launch and every conn.load() reload — this is
        exactly what changes when a worker switches which scenario config
        it is currently running.

        ``net_file`` / ``route_file`` in the scenario JSON are resolved
        relative to ``sumo_files/`` (e.g. ``"s2/Town05_net_counterClock.xml"``
        or a bare ``"Town02_edit.net.xml"``).  Absolute paths pass through.
        """
        if self.cfg.sumo is None:
            raise SumoRuntimeFault(f"{self.cfg.setting_id} is CARLA-only")
        net_file   = self.cfg.sumo.net_file
        route_file = self.cfg.sumo.route_file
        if not os.path.isabs(net_file):
            net_file = os.path.join(_PROJECT_ROOT, net_file)
        if not os.path.isabs(route_file):
            route_file = os.path.join(_PROJECT_ROOT, route_file)
        return [
            "-n", net_file,
            "-r", route_file,
            "--step-length",               str(self.cfg.sumo.step_length),
            "--lateral-resolution",      "1.6",
            # Traffic-light compliance is outside the work-zone task.  Keep
            # this in the shared option list so it applies both at process
            # launch and after every conn.load() scenario/episode reload.
            "--tls.all-off",
            "--collision.action",        "none",
            "--collision.mingap-factor", "0",
            "--collision.check-junctions", "false",
            "--no-step-log",             "true",
            "--no-warnings",             "true",
            "--ignore-route-errors",     "true",
            "--time-to-teleport",        "-1",
            "--time-to-teleport.highways", "-1",
        ]

    def _launch_sumo_process(self) -> None:
        """Spawn this worker's SUMO subprocess ourselves and capture stderr.

        We launch it directly (instead of letting traci.start() hide the
        Popen handle) so we can log the real PID, confirm the exit code on
        restart, preserve the last TraCI command, and capture stderr to a
        per-generation log file.
        """
        if self.cfg.sumo is None:
            raise SumoRuntimeFault(f"{self.cfg.setting_id} is CARLA-only")
        sumo_bin = "sumo-gui" if self.cfg.sumo.sumo_gui else "sumo"
        self._sumo_generation += 1
        log_dir = os.path.join(_PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._sumo_stderr_path = os.path.join(
            log_dir, f"{self._sumo_label}_sumo_g{self._sumo_generation:04d}.log",
        )
        command = [sumo_bin, "--remote-port", str(self.cfg.sumo.port)] + self._sumo_option_args()
        try:
            self._sumo_stderr_file = open(self._sumo_stderr_path, "a", encoding="utf-8", buffering=1)
            self._sumo_stderr_file.write(
                f"\n===== SUMO START {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                f"worker={self._worker_id} scenario={self._scenario_name} command={command!r} =====\n"
            )
            self._sumo_proc = subprocess.Popen(
                command,
                stdout=self._sumo_stderr_file,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            if self._sumo_stderr_file is not None:
                try:
                    self._sumo_stderr_file.write(f"===== SUMO START FAILED error={exc!r} =====\n")
                    self._sumo_stderr_file.close()
                except Exception:
                    pass
            self._sumo_stderr_file = None
            self._sumo_proc = None
            self._sumo_running = False
            self._monitor.record_exception("sumo_process_launch", exc, command=command)
            raise SumoRuntimeFault(f"Could not launch SUMO command {command!r}: {exc}") from exc
        self._monitor.increment("sumo_process_starts")
        self._monitor.record_event(
            "sumo_process_started", pid=self._sumo_proc.pid, command=command,
            stderr_log=self._sumo_stderr_path, generation=self._sumo_generation,
        )
        logger.info(
            "SUMO process launched | worker=%d scenario=%s pid=%d carla_port=%d tm_port=%d "
            "sumo_port=%d log=%s",
            self._worker_id, self._scenario_name, self._sumo_proc.pid,
            self.cfg.carla.port, self.cfg.carla.tm_port, self.cfg.sumo.port, self._sumo_stderr_path,
        )

    def _connect_traci(self) -> None:
        """Attach an explicit TraCI Connection and record every low-level command."""
        if self.cfg.sumo is None:
            raise SumoRuntimeFault(f"{self.cfg.setting_id} is CARLA-only")
        try:
            self._conn = traci.connect(
                port=self.cfg.sumo.port,
                numRetries=20,
                proc=self._sumo_proc,
            )
        except Exception as exc:
            self._record_sumo_failure("traci_connect", exc)
            raise SumoRuntimeFault(f"Could not connect TraCI on port {self.cfg.sumo.port}: {exc}") from exc
        self._install_traci_command_recorder(self._conn)
        self._sumo_running = True
        self._monitor.record_event("traci_connected", sumo_port=self.cfg.sumo.port)
        logger.info("TraCI connected | worker=%d scenario=%s sumo_port=%d",
                    self._worker_id, self._scenario_name, self.cfg.sumo.port)

    def _sumo_is_alive(self) -> bool:
        """Cheap liveness check: process running AND connection responds."""
        if self._sumo_proc is None or self._sumo_proc.poll() is not None or self._conn is None:
            return False
        try:
            self._conn.simulation.getTime()
            return True
        except Exception:
            return False

    def _terminate_sumo(self, reason: str) -> None:
        """Stop this worker's SUMO process and confirm it actually exited."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as exc:
                self._monitor.record_exception("traci_close", exc, reason=reason)
            self._conn = None

        proc = self._sumo_proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
                self._last_sumo_returncode = proc.returncode
                details = _returncode_details(proc.returncode)
                self._monitor.record_event(
                    "sumo_process_stopped", pid=proc.pid, reason=reason, **details,
                    last_traci_command=self._last_traci_command,
                    stderr_tail=self._stderr_tail(),
                )
                logger.warning(
                    "SUMO process stopped | worker=%d scenario=%s pid=%d reason=%s %s",
                    self._worker_id, self._scenario_name, proc.pid, reason, details,
                )
            except Exception as exc:
                self._monitor.record_exception("sumo_process_terminate", exc, reason=reason, pid=proc.pid)
        self._sumo_proc = None
        self._sumo_running = False
        if self._sumo_stderr_file is not None:
            try:
                self._sumo_stderr_file.write(
                    f"===== SUMO END {time.strftime('%Y-%m-%dT%H:%M:%S%z')} reason={reason} "
                    f"last_traci={self._last_traci_command!r} =====\n"
                )
                self._sumo_stderr_file.close()
            except Exception:
                pass
            self._sumo_stderr_file = None

    def _build_wz_polygon(self):
        """Build the curved work-zone polygon from config; raises on failure."""
        from logic.workzone_geometry import build_workzone_polygon
        pc = self.cfg.workzone.polygon_config
        result = build_workzone_polygon(
            self._cmap,
            carla.Location(x=pc.head_x, y=pc.head_y, z=pc.head_z),
            carla.Location(x=pc.tail_x, y=pc.tail_y, z=pc.tail_z),
            sample_spacing_m=pc.sample_spacing_m,
            half_width_m=pc.half_width_m,
            margin_m=pc.margin_m,
            expected_road_id=pc.expected_road_id,
            expected_lane_id=pc.expected_lane_id,
        )
        self._monitor.record_event(
            "workzone_polygon_built",
            road_id=result.road_id,
            lane_id=result.lane_id,
            waypoints=len(result.centerline_pts),
            half_width_m=result.actual_half_width_m,
            bounds=list(result.polygon.bounds),
        )
        return result

    def _invalidate_wz_polygon_cache(self, *, reason: str) -> None:
        """Drop all cached geometry after a CARLA map change."""
        cache = getattr(self, "_wz_polygon_cache", None)
        if cache is not None:
            cache.clear()
        self._wz_polygon = None
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            monitor.record_event("workzone_polygon_cache_cleared", reason=reason)

    def _get_or_build_wz_polygon(self):
        """Return cached geometry for the current Town/setting, if configured."""
        if self.cfg.workzone.polygon_config is None:
            return None
        town = _canonical_town_name(self.cfg.carla.town)
        key = (town, self.cfg.setting_id)
        cache = getattr(self, "_wz_polygon_cache", None)
        if cache is None:
            cache = {}
            self._wz_polygon_cache = cache
        result = cache.get(key)
        if result is None:
            result = self._build_wz_polygon()
            cache[key] = result
            monitor = getattr(self, "_monitor", None)
            if monitor is not None:
                monitor.record_event(
                    "workzone_polygon_cache_store", town=town,
                    setting_id=self.cfg.setting_id,
                )
        else:
            monitor = getattr(self, "_monitor", None)
            if monitor is not None:
                monitor.record_event(
                    "workzone_polygon_cache_hit", town=town,
                    setting_id=self.cfg.setting_id,
                )
        return result

    # ================================================================== #
    #  Internal — per-step helpers                                        #
    # ================================================================== #

    def _apply_ego_control(self, action: np.ndarray) -> None:
        ctrl          = carla.VehicleControl()
        ctrl.steer    = float(np.clip(action[_IDX_STEER],    -1.0, 1.0))
        ctrl.throttle = float(np.clip(action[_IDX_THROTTLE],  0.0, 1.0))
        ctrl.brake    = float(np.clip(action[_IDX_BRAKE],     0.0, 1.0))
        self._ego.apply_control(ctrl)

    def _update_pedestrians(self) -> None:
        """
        Update CARLA-controlled pedestrian actors.
        Placeholder — implement scripted paths or CARLA Walker AI here.
        """
        if self._jaywalker_controller is not None and self._ego is not None:
            self._jaywalker_controller.update(self._ego)
            self._pedestrians = self._jaywalker_controller.active_actors()

    def _build_observation(self) -> np.ndarray:
        """Return ego state plus detected cone/vehicle/walker objects."""
        if self._ego is None:
            return np.zeros(_OBS_DIM, dtype=np.float32)
        transform = self._ego.get_transform()
        velocity = self._ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        observation_cfg = self.cfg.observation
        provider = self._observation_base_heading_provider
        base_heading = (
            float(provider(transform))
            if provider is not None
            else float(self.cfg.carla.road_heading_deg)
        )
        dt = max(float(self.cfg.episode.sim_dt), 1e-6)
        previous_speed = self._observation_previous_speed_mps
        acceleration = (
            0.0 if previous_speed is None else (speed - previous_speed) / dt
        )
        current_yaw = float(transform.rotation.yaw)
        previous_yaw = self._observation_previous_yaw_deg
        yaw_delta = (
            0.0
            if previous_yaw is None
            else (current_yaw - previous_yaw + 180.0) % 360.0 - 180.0
        )
        yaw_rate = yaw_delta / dt
        ego_state = encode_ego_state(
            ego_yaw_deg=transform.rotation.yaw,
            ego_speed_mps=speed,
            acceleration_mps2=acceleration,
            yaw_rate_deg_s=yaw_rate,
            previous_speed_action=float(self._observation_previous_action[0]),
            previous_heading_action=float(self._observation_previous_action[1]),
            base_heading_deg=base_heading,
            max_ego_speed_mps=observation_cfg.max_ego_speed_mps,
            max_acceleration_mps2=observation_cfg.max_acceleration_mps2,
            yaw_rate_limit_deg_s=observation_cfg.yaw_rate_limit_deg_s,
        )
        dynamic_agents = self._dynamic_agent_observation(transform, velocity)
        observation = np.concatenate((ego_state, dynamic_agents)).astype(
            np.float32, copy=False
        )
        self._observation_previous_speed_mps = float(speed)
        self._observation_previous_yaw_deg = current_yaw
        if observation.shape != (_OBS_DIM,) or not np.all(np.isfinite(observation)):
            raise RuntimeError(
                f"Invalid observation-v2 shape/content: {observation.shape}"
            )
        return np.clip(observation, -1.0, 1.0)

    def _dynamic_agent_observation(
        self, ego_transform: Any, ego_velocity: Any
    ) -> np.ndarray:
        """Encode visible cones, SUMO vehicles and CARLA walkers."""
        background_speeds = self.get_background_speed_map()
        agents: list[AgentPoint] = []
        for index, (cone_x, cone_y) in enumerate(self.cfg.workzone.traffic_cones):
            agents.append(AgentPoint(
                actor_id=f"cone:{self.cfg.setting_id}:{index}",
                kind="cone", x=float(cone_x), y=float(cone_y), vx=0.0, vy=0.0,
            ))
        for sumo_id, actor in self.get_background_actor_map().items():
            try:
                if not actor.is_alive:
                    continue
                actor_transform = actor.get_transform()
                scalar_speed = background_speeds.get(sumo_id)
                if scalar_speed is not None:
                    actor_yaw = math.radians(float(actor_transform.rotation.yaw))
                    actor_vx = float(scalar_speed) * math.cos(actor_yaw)
                    actor_vy = float(scalar_speed) * math.sin(actor_yaw)
                else:
                    actor_velocity = actor.get_velocity()
                    actor_vx, actor_vy = actor_velocity.x, actor_velocity.y
                agents.append(AgentPoint(
                    actor_id=f"vehicle:{sumo_id}", kind="vehicle",
                    x=float(actor_transform.location.x),
                    y=float(actor_transform.location.y),
                    vx=float(actor_vx), vy=float(actor_vy),
                ))
            except Exception:
                continue
        for actor in self._pedestrians:
            try:
                if not actor.is_alive:
                    continue
                actor_transform = actor.get_transform()
                actor_velocity = actor.get_velocity()
                agents.append(AgentPoint(
                    actor_id=f"walker:{actor.id}", kind="walker",
                    x=float(actor_transform.location.x),
                    y=float(actor_transform.location.y),
                    vx=float(actor_velocity.x), vy=float(actor_velocity.y),
                ))
            except Exception:
                continue
        return self._agent_encoder.encode(
            ego_x=float(ego_transform.location.x),
            ego_y=float(ego_transform.location.y),
            ego_yaw_deg=float(ego_transform.rotation.yaw),
            ego_vx=float(ego_velocity.x),
            ego_vy=float(ego_velocity.y),
            agents=agents,
        )

    def _new_agent_encoder(self) -> TrackedAgentEncoder:
        observation = self.cfg.observation
        return TrackedAgentEncoder(
            max_agents=observation.max_agents,
            radius_m=observation.perception_radius_m,
            relative_speed_scale_mps=observation.relative_speed_scale_mps,
            missing_ttl_steps=observation.missing_actor_ttl_steps,
        )

    def _compute_reward(self,
                         terminated: bool,
                         truncated:  bool,
                         success:    bool,
                         info:       dict[str, Any]) -> float:
        """Apply terminal precedence and anti-farming bounded progress."""
        tracker = getattr(self, "_bounded_progress_reward", None)
        source = getattr(self, "_reward_progress_source", "uninitialized")
        progress_reward = 0.0
        if not (terminated or truncated) and tracker is not None:
            if source == "s3_legal_corridor_arc":
                metric = info.get("corridor_progress_m")
            else:
                goal_distance = info.get("dist_to_goal")
                metric = -float(goal_distance) if goal_distance is not None else None
            if metric is not None:
                progress_reward = tracker.advance(float(metric))

        weights = RewardWeights(
            failure=self.cfg.reward.failure,
            success=self.cfg.reward.success,
            timeout=self.cfg.reward.timeout,
            progress_budget=self.cfg.reward.progress_budget,
            step_cost=self.cfg.reward.step_cost,
        )
        reward = compute_reward(
            terminated=terminated,
            truncated=truncated,
            success=success,
            reason=info.get("reason"),
            progress_reward=progress_reward,
            weights=weights,
        )
        info["reward_progress_source"] = source
        info["reward_progress_bonus"] = float(progress_reward)
        info["reward_progress_fraction"] = (
            float(tracker.high_water_fraction) if tracker is not None else 0.0
        )
        info["reward_step_cost"] = (
            float(weights.step_cost) if not (terminated or truncated) else 0.0
        )
        info["reward_total"] = float(reward)
        return reward

    def _initialize_reward_progress(self) -> None:
        """Create one episode-local normalized progress high-water tracker."""
        if self._ego is None or self._destination_transform is None:
            raise RuntimeError("Cannot initialize reward progress before Ego and destination")

        if self._corridor_progress_tracker is not None:
            workzone = self.cfg.workzone
            left = workzone.corridor_left_boundary_points
            right = workzone.corridor_right_boundary_points
            if not left or not right or self._prev_corridor_progress is None:
                raise RuntimeError("S3 legal corridor progress is missing its ordered boundaries")
            target_tracker = CorridorProgressTracker(left, right)
            destination = self._destination_transform.location
            target_progress = target_tracker.progress_at(destination.x, destination.y)
            if target_progress <= self._prev_corridor_progress:
                raise RuntimeError("S3 finish must be forward of its origin on the legal corridor arc")
            self._bounded_progress_reward = BoundedProgressReward(
                self._prev_corridor_progress,
                target_progress,
                budget=self.cfg.reward.progress_budget,
            )
            self._reward_progress_source = "s3_legal_corridor_arc"
            return

        ego_location = self._ego.get_location()
        destination = self._destination_transform.location
        initial_distance = math.hypot(
            destination.x - ego_location.x,
            destination.y - ego_location.y,
        )
        self._bounded_progress_reward = BoundedProgressReward(
            -max(initial_distance, 1e-6),
            0.0,
            budget=self.cfg.reward.progress_budget,
        )
        if self.cfg.scenario_id == "s2":
            self._reward_progress_source = (
                "s2_goal_distance_bounded_fallback_no_legal_route"
            )
            reported = getattr(self, "_reported_reward_fallback_settings", set())
            if self.cfg.setting_id not in reported:
                logger.warning(
                    "REWARD_PROGRESS_FALLBACK setting=%s source=%s "
                    "unresolved=owner_authored_legal_s2_route",
                    self.cfg.setting_id,
                    self._reward_progress_source,
                )
                reported.add(self.cfg.setting_id)
                self._reported_reward_fallback_settings = reported
        else:
            self._reward_progress_source = "goal_distance_bounded"

    # ================================================================== #
    #  Public utilities                                                    #
    # ================================================================== #

    @property
    def ego(self) -> carla.Actor | None:
        """The current Ego CARLA actor."""
        return self._ego

    def set_observation_base_heading_provider(
        self, provider: Callable[[Any], float] | None
    ) -> None:
        """Inject a pure heading query matching the high-level controller."""
        self._observation_base_heading_provider = provider

    def set_observation_previous_action(
        self, speed_action: float, heading_action: float
    ) -> None:
        """Store the last normalized high-level action for the next frame."""
        self._observation_previous_action = np.asarray(
            [
                np.clip(float(speed_action), -1.0, 1.0),
                np.clip(float(heading_action), -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    @property
    def observation_actor_ids(self) -> tuple[str | None, ...]:
        """Stable slots for diagnostics only; IDs are absent from policy input."""
        return self._agent_encoder.actor_ids

    def driving_heading_at(self, location: carla.Location) -> float | None:
        """Return the exact CARLA Driving-lane heading at ``location``.

        Projection is deliberately disabled: after a road departure, snapping
        to a nearby lane could select an opposite-direction heading.
        """
        if self._cmap is None:
            return None
        waypoint = self._cmap.get_waypoint(
            location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return None
        return float(waypoint.transform.rotation.yaw)

    @property
    def world(self) -> carla.World | None:
        """The CARLA world handle."""
        return self._world

    def get_background_actor_map(self) -> dict[str, carla.Actor]:
        """Return the current SUMO-id → CARLA-actor mapping."""
        return self._bg_traffic.get_actor_map() if self._bg_traffic else {}

    def get_background_speed_map(self) -> dict[str, float]:
        """Return the current SUMO-id → speed (m/s) mapping. Use this
        instead of actor.get_velocity() — background actors have physics
        disabled, so their own velocity state is not meaningful."""
        return self._bg_traffic.get_speed_map() if self._bg_traffic else {}

    # ================================================================== #
    #  Resilient worker lifecycle                                         #
    # ================================================================== #

    def _resilient_reset(self, mode: str) -> np.ndarray:
        """Never forward a recoverable initialization failure to SB3.

        A failed spawn attempt or simulator transport fault aborts the partial
        episode, verifies cleanup, and immediately starts a fresh episode in
        the same worker process.  This is intentionally a loop: returning an
        exception here would terminate a SubprocVecEnv worker.
        """
        while True:
            attempt_started = time.perf_counter()
            self._episode_id += 1
            self._episode_step = 0
            self._monitor.update_state(episode=self._episode_id, environment_step=0)
            try:
                observation = self._reset_once(mode)
                self._last_observation = observation.copy()
                self._initialization_failures = 0
                self._monitor.record_event("episode_reset_ready", episode=self._episode_id)
                self._update_monitor_state(refresh_actors=True)
                return observation
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                self._initialization_failures += 1
                self._monitor.record_exception(
                    "episode_initialization",
                    exc,
                    episode=self._episode_id,
                    failure_count=self._initialization_failures,
                )
                logger.exception(
                    "Episode initialization aborted | worker=%d scenario=%s episode=%d failures=%d",
                    self._worker_id, self._scenario_name, self._episode_id,
                    self._initialization_failures,
                )
                cleanup_started_recovery = self._abort_episode_initialization(exc)
                if not cleanup_started_recovery and (isinstance(exc, CarlaRuntimeFault) or _looks_like_carla_fault(exc)):
                    self._recover_carla("reset_failure", exc)
                elif not cleanup_started_recovery and (isinstance(exc, SumoRuntimeFault) or isinstance(exc, _SUMO_TRANSPORT_EXCEPTIONS)):
                    self._terminate_sumo(reason="reset_failure")
                # Back off just enough to avoid a busy loop while retaining the
                # worker and training run for an external server to recover.
                time.sleep(min(2.0, 0.1 * self._initialization_failures))
            finally:
                self._monitor.record_timing("reset", time.perf_counter() - attempt_started)

    def _reset_once(self, mode: str) -> np.ndarray:
        if not self._carla_connected:
            self.connect()

        # Destruction plus verification occurs before every episode, not only
        # after normal terminal states.  This also removes role-tagged orphans.
        self._episode_cleanup()
        self._agent_encoder.reset()
        if self.cfg.uses_sumo:
            reset_reason = self._resilient_reload_sumo()
            if not self._bg_traffic or not self._conn:
                raise SumoRuntimeFault("SUMO synchronization components are unavailable after reload")
            self._bg_traffic.start_episode(self._episode_id)
        else:
            self._terminate_sumo(reason="carla_only_reset")
            self._bridge = None
            self._ego_proxy = None
            # self._ped_proxy = None  # pedestrian proxy disabled
            self._bg_traffic = None
            reset_reason = "carla_only"
        logger.info(
            "Episode reset begin | worker=%d scenario=%s episode=%d reason=%s "
            "carla_port=%d tm_port=%d sumo_port=%s sumo_pid=%s",
            self._worker_id, self._scenario_name, self._episode_id, reset_reason,
            self.cfg.carla.port, self.cfg.carla.tm_port,
            self.cfg.sumo.port if self.cfg.sumo else None,
            self._sumo_proc.pid if self._sumo_proc else None,
        )

        if self._od_sampler is None:
            self._od_sampler = OriginDestinationSampler(self.cfg)
        carla_map = self._world.get_map() if self._world else None
        # Origin is a manual absolute transform from the materialized setting.
        # A layout may override its Work-zone-level OD; do not project or
        # rotate either form using the CARLA map.
        origin_t = self._od_sampler.sample_origin(mode)
        self._destination_transform = self._od_sampler.sample_destination(mode)
        self._ego, origin_t = self._spawn_ego(mode, origin_t)

        self._world_tick("reset_settle")

        self._termination = EpisodeTerminationChecker(self.cfg, carla_map=self._cmap)
        self._termination.attach_collision_sensor(
            self._world,
            self._ego,
            actor_register=self._track_episode_actor,
            actor_unregister=self._untrack_episode_actor,
        )
        # Build curved polygon once per Town/setting and inject it into the
        # episode checker. Switching ABC/WZ settings reuses their cached result.
        if self._wz_polygon is None and self.cfg.workzone.polygon_config is not None:
            self._wz_polygon = self._get_or_build_wz_polygon()
        if self._wz_polygon is not None:
            self._termination.set_forbidden_polygon(self._wz_polygon.polygon)

        workzone = self.cfg.workzone
        # Only S3's centerline represents legal drivable space.  S2's sampled
        # polygon centerline represents the forbidden work-zone lane, so using
        # it for path progress would reward the ego for entering the closure.
        # Until S2 has a separately owner-authored legal reference path, it
        # deliberately retains the generic destination-distance progress
        # reward below rather than fabricating an unsafe arc-length target.
        if (
            self.cfg.scenario_id == "s3"
            and workzone.geometry_mode == "safe_corridor"
            and workzone.corridor_open_ends
            and workzone.corridor_left_boundary_points
            and workzone.corridor_right_boundary_points
        ):
            self._corridor_progress_tracker = CorridorProgressTracker(
                workzone.corridor_left_boundary_points,
                workzone.corridor_right_boundary_points,
            )
            ego_location = self._ego.get_location()
            self._prev_corridor_progress = self._corridor_progress_tracker.progress_at(
                ego_location.x, ego_location.y
            )
        else:
            self._corridor_progress_tracker = None
            self._prev_corridor_progress = None
        self._initialize_reward_progress()

        if self.cfg.uses_sumo:
            assert self.cfg.sumo is not None and self._conn is not None and self._bg_traffic is not None
            for lane_id in self.cfg.workzone.closed_lanes:
                try:
                    # An empty allowed-list means "no restriction" in SUMO; it
                    # does not close the lane.  Explicitly disallow every class.
                    self._conn.lane.setDisallowed(lane_id, ["all"])
                except _SUMO_TRANSPORT_EXCEPTIONS as exc:
                    raise SumoRuntimeFault(f"Could not close SUMO lane {lane_id}: {exc}") from exc
            if not self._ego_proxy:
                raise SumoRuntimeFault("Ego proxy is unavailable after SUMO reload")
            # Required co-simulation initialization order:
            # CARLA Ego -> SUMO Ego proxy -> SUMO traffic -> CARLA mirrors.
            self._ego_proxy.reset(self._ego)
            if self.cfg.sumo.traffic_pattern == "platoon":
                self._bg_traffic.start_platoon_traffic(
                    routes=self.cfg.sumo.bg_routes,
                    vtype=self.cfg.sumo.bg_vtype,
                    max_vehicles=self.cfg.sumo.max_background_vehicles,
                    size_min=self.cfg.sumo.platoon_size_min,
                    size_max=self.cfg.sumo.platoon_size_max,
                    headway_min_s=self.cfg.sumo.platoon_headway_min_s,
                    headway_max_s=self.cfg.sumo.platoon_headway_max_s,
                    gap_min_s=self.cfg.sumo.platoon_gap_min_s,
                    gap_max_s=self.cfg.sumo.platoon_gap_max_s,
                )
            else:
                n_init = min(
                    self.cfg.sumo.max_background_vehicles,
                    self.cfg.sumo.initial_background_vehicles,
                )
                self._bg_traffic.spawn_initial_traffic(
                    n_init, self.cfg.sumo.bg_routes, self.cfg.sumo.bg_vtype
                )
            # TraCI vehicle.add() schedules a departure; one SUMO step is
            # required before newly-added traffic appears in getIDList().
            # Re-anchor the proxy immediately before that bootstrap step so
            # SUMO traffic reacts to the stationary CARLA Ego's real pose.
            # Do not advance SUMO alone for several warm-up steps: that lets
            # the proxy drive away while CARLA is still at the reset pose.
            self._ego_proxy.sync(self._ego)
            self._simulation_step("reset_background_materialize")
            if self.cfg.sumo.traffic_pattern == "platoon":
                # Confirm the first request only after SUMO reports its actual
                # departure; all later platoon timing is anchored the same way.
                self._bg_traffic.tick_platoon_spawn()
            # Initial SUMO vehicles are not visible to the policy until they
            # are mirrored into CARLA.  Synchronize once before constructing
            # the episode's first observation so its traffic state is valid.
            self._bg_traffic.sync()

        if self.cfg.jaywalker is not None:
            self._jaywalker_controller = JaywalkerController(
                self._world,
                self.cfg.jaywalker,
                road_heading_deg=self.cfg.carla.road_heading_deg,
                actor_register=self._track_episode_actor,
                actor_unregister=self._untrack_episode_actor,
            )
            self._jaywalker_controller.start_episode(self._ego)
            self._pedestrians = self._jaywalker_controller.active_actors()

        self._episode_step = 0
        self._prev_dist_to_goal = None
        logger.info(
            "Episode reset done | worker=%d scenario=%s episode=%d origin=%s dest=%s mode=%s",
            self._worker_id, self._scenario_name, self._episode_id,
            _loc_str(origin_t.location), _loc_str(self._destination_transform.location), mode,
        )
        return self._build_observation()

    def _spawn_ego(self, mode: str, origin_t: carla.Transform) -> tuple[carla.Actor, carla.Transform]:
        """Retry only collision-rejected spawn operations with a new transform."""
        if not self._world:
            raise CarlaRuntimeFault("Cannot spawn ego without a CARLA world")
        bp = self._world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", self.cfg.ego_role or "ego")

        for attempt in range(1, 11):
            # CARLA returns None only for an occupied transform.  RPC errors
            # are deliberately not swallowed as if they were spawn collisions.
            actor = self._world.try_spawn_actor(bp, origin_t)
            self._monitor.increment("spawn_attempts")
            if actor is not None:
                self._track_episode_actor(actor, "ego")
                self._monitor.record_event("spawn_attempt", actor_kind="ego", attempt=attempt, success=True)
                return actor, origin_t

            self._monitor.increment("spawn_collisions")
            self._monitor.record_event(
                "spawn_attempt", actor_kind="ego", attempt=attempt, success=False, reason="collision",
            )
            logger.warning(
                "Ego spawn rejected | worker=%d scenario=%s episode=%d attempt=%d/10 "
                "transform=(%.2f, %.2f, %.2f, yaw=%.1f); resampling transform.",
                self._worker_id, self._scenario_name, self._episode_id, attempt,
                origin_t.location.x, origin_t.location.y, origin_t.location.z,
                origin_t.rotation.yaw,
            )
            origin_t = self._od_sampler.sample_origin(mode)

        raise SpawnExhaustedError(
            f"Ego spawn occupied for all 10 attempts in episode {self._episode_id}; starting a new episode."
        )

    def _resilient_step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        started = time.perf_counter()
        try:
            return self._step_once(np.asarray(action, dtype=np.float32))
        except (KeyboardInterrupt, SystemExit):
            raise
        except CarlaRuntimeFault as exc:
            return self._recoverable_step_failure("carla_tick_or_rpc_failure", exc, fault_kind="carla")
        except SumoRuntimeFault as exc:
            return self._recoverable_step_failure("sumo_connection_lost", exc, fault_kind="sumo")
        except MirrorSpawnExhaustedError as exc:
            return self._recoverable_step_failure("sumo_mirror_spawn_collision", exc, fault_kind="generic")
        except _SUMO_TRANSPORT_EXCEPTIONS as exc:
            return self._recoverable_step_failure("sumo_connection_lost", exc, fault_kind="sumo")
        except Exception as exc:
            # Unexpected runtime faults are logged with a traceback and made
            # terminal for this episode, rather than killing a worker process.
            kind = "carla" if _looks_like_carla_fault(exc) else "generic"
            return self._recoverable_step_failure("worker_runtime_fault", exc, fault_kind=kind)
        finally:
            self._monitor.record_timing("env.step", time.perf_counter() - started)
            self._update_monitor_state(refresh_actors=False)

    def _step_once(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self._ego or not self._termination or not self._destination_transform:
            raise CarlaRuntimeFault("step() called without a fully initialized CARLA episode")
        if self.cfg.uses_sumo and (not self._ego_proxy or not self._bg_traffic):
            raise SumoRuntimeFault("step() called without synchronized SUMO components")

        self._apply_ego_control(action)
        self._update_pedestrians()
        self._world_tick("step")

        if self.cfg.uses_sumo:
            assert self.cfg.sumo is not None
            assert self._ego_proxy is not None and self._bg_traffic is not None
            self._ego_proxy.sync(self._ego)
            # self._ped_proxy.sync(self._pedestrians)  # S4 has no SUMO proxy
            if self.cfg.sumo.traffic_pattern == "poisson":
                self._bg_traffic.tick_poisson_spawn(
                    rate_veh_s=self.cfg.sumo.bg_spawn_rate_veh_s,
                    dt=self.cfg.episode.sim_dt,
                    routes=self.cfg.sumo.bg_routes,
                    vtype=self.cfg.sumo.bg_vtype,
                    max_vehicles=self.cfg.sumo.max_background_vehicles,
                )
            self._simulation_step("step")
            if self.cfg.sumo.traffic_pattern == "platoon":
                self._bg_traffic.tick_platoon_spawn()
            self._bg_traffic.sync()

        self._episode_step += 1
        terminated, truncated, success, info = self._termination.tick(
            self._ego,
            self._destination_transform,
            bg_actor_map=self._bg_traffic.get_actor_map() if self._bg_traffic else {},
        )
        info["success"] = success
        info["episode_step"] = self._episode_step
        info["scenario_id"] = self.cfg.scenario_id
        info["wz_id"] = self.cfg.wz_id
        info["layout_id"] = self.cfg.layout_id
        info["setting_id"] = self.cfg.setting_id
        info["traffic_backend"] = self.cfg.traffic_backend
        ego_transform = self._ego.get_transform()
        ego_velocity = self._ego.get_velocity()
        info["ego_x"] = float(ego_transform.location.x)
        info["ego_y"] = float(ego_transform.location.y)
        info["ego_speed_mps"] = float(math.sqrt(
            ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2
        ))
        info["sumo_mirrors_in_carla"] = (
            len(self._bg_traffic.get_actor_map()) if self._bg_traffic else 0
        )
        if self._corridor_progress_tracker is not None:
            info["corridor_progress_m"] = self._corridor_progress_tracker.progress_at(
                ego_transform.location.x, ego_transform.location.y
            )
        if self._jaywalker_controller is not None:
            walker_status = self._jaywalker_controller.status()
            for key, value in walker_status.items():
                if key != "trigger_distances_m":
                    info[f"jaywalker_{key}"] = value
        else:
            info["jaywalker_active_count"] = 0
        observation = self._build_observation()
        info["observation_actor_ids"] = self.observation_actor_ids
        # Cache the current observation for recoverable step failures, when no new observation can be built.
        self._last_observation = observation.copy()
        self._monitor.update_state(episode=self._episode_id, environment_step=self._episode_step)
        reward = self._compute_reward(terminated, truncated, success, info)
        # Track the previous goal distance for the within-episode differential progress reward.
        if "dist_to_goal" in info:
            self._prev_dist_to_goal = info["dist_to_goal"]
        if "corridor_progress_m" in info:
            self._prev_corridor_progress = info["corridor_progress_m"]
        return observation, reward, terminated, truncated, info

    def _recoverable_step_failure(
        self,
        reason: str,
        exc: BaseException,
        *,
        fault_kind: str,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._monitor.record_exception(
            "environment_step",
            exc,
            reason=reason,
            episode=self._episode_id,
            environment_step=self._episode_step,
        )
        logger.exception(
            "Recoverable worker fault | worker=%d scenario=%s episode=%d step=%d reason=%s",
            self._worker_id, self._scenario_name, self._episode_id, self._episode_step, reason,
        )
        self._episode_step += 1
        try:
            self._episode_cleanup()
        except Exception as cleanup_error:
            self._monitor.record_exception("step_fault_cleanup", cleanup_error)

        if fault_kind == "carla":
            self._recover_carla(reason, exc)
        elif fault_kind == "sumo":
            self._record_sumo_failure(reason, exc)
            self._terminate_sumo(reason=reason)

        info = {
            "success": False,
            "episode_step": self._episode_step,
            "reason": reason,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "scenario_id": self.cfg.scenario_id,
            "wz_id": self.cfg.wz_id,
            "layout_id": self.cfg.layout_id,
            "setting_id": self.cfg.setting_id,
            "traffic_backend": self.cfg.traffic_backend,
        }
        self._monitor.update_state(episode=self._episode_id, environment_step=self._episode_step)
        # SB3's vector worker will now invoke reset() itself.  Returning the
        # last valid observation preserves the Gym terminal-observation contract.
        return self._last_observation.copy(), 0.0, False, True, info

    def _resilient_close(self) -> None:
        try:
            try:
                self._episode_cleanup()
            except Exception as cleanup_error:
                # Destruction in synchronous mode is committed by a world
                # tick, which may not have happened yet at close time.  Mirrors
                # the graceful handling in step/reset: a residual cleanup
                # failure here must not crash the run — we still shut down
                # SUMO and restore the world settings.
                self._monitor.record_exception("close_cleanup", cleanup_error)
            self._terminate_sumo(reason="env_close")
            if self._carla_connected and self._world:
                try:
                    settings = self._world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    self._world.apply_settings(settings)
                except Exception as exc:
                    self._monitor.record_exception("carla_restore_settings", exc)
            self._carla_connected = False
            self._terminate_carla_process(reason="env_close")
            self._monitor.record_event("environment_closed")
        finally:
            self._monitor.close()
            close_process_logging(self._worker_log_path)
        logger.info("CarlaSumoEnv closed | worker=%d scenario=%s", self._worker_id, self._scenario_name)

    # ------------------------------------------------------------------ #
    #  Episode actor ownership and cleanup                                #
    # ------------------------------------------------------------------ #

    def _track_episode_actor(self, actor: carla.Actor, kind: str) -> None:
        try:
            role_name = actor.attributes.get("role_name", "")
        except Exception:
            role_name = ""
        self._episode_actors[actor.id] = {
            "kind": kind,
            "role_name": role_name,
            "type_id": getattr(actor, "type_id", "unknown"),
        }
        self._monitor.record_event(
            "actor_registered", actor_id=actor.id, actor_kind=kind,
            role_name=role_name, type_id=getattr(actor, "type_id", "unknown"),
        )

    def _untrack_episode_actor(self, actor_id: int) -> None:
        if self._episode_actors.pop(actor_id, None) is not None:
            self._monitor.record_event("actor_unregistered", actor_id=actor_id)

    def _record_mirror_spawn_attempt(
        self,
        actor_kind: str,
        attempt: int,
        success: bool,
        reason: str | None,
    ) -> None:
        self._monitor.increment("spawn_attempts")
        if not success:
            self._monitor.increment("spawn_collisions")
        self._monitor.record_event(
            "spawn_attempt", actor_kind=actor_kind, attempt=attempt, success=success, reason=reason,
        )

    def _owned_role_names(self) -> set[str]:
        return set(_OWNED_ROLE_NAMES) | {self.cfg.ego_role}

    def _find_owned_actor_ids(self) -> set[int]:
        if not self._world or not self._carla_connected:
            return set()
        try:
            owned_roles = self._owned_role_names()
            return {
                actor.id for actor in self._world.get_actors()
                if actor.is_alive and actor.attributes.get("role_name", "") in owned_roles
            }
        except Exception as exc:
            self._monitor.record_exception("orphan_actor_scan", exc)
            return set()

    def _destroy_actor_ids(self, actor_ids: set[int]) -> None:
        if not actor_ids or not self._client:
            return
        try:
            commands = [carla.command.DestroyActor(actor_id) for actor_id in sorted(actor_ids)]
            responses = self._client.apply_batch_sync(commands, False)
            errors = [
                response.error
                for response in responses
                if getattr(response, "error", None)
                and "actor: not found" not in response.error.lower()
            ]
            if errors:
                self._monitor.record_event("actor_destroy_batch_errors", errors=errors[:10])
                logger.warning("CARLA destroy batch reported %d errors: %s", len(errors), errors[:3])
        except Exception as exc:
            self._monitor.record_exception("actor_destroy_batch", exc, actor_ids=sorted(actor_ids))

    def _verify_actor_removal(self, actor_ids: set[int]) -> set[int]:
        if not actor_ids or not self._world or not self._carla_connected:
            return set(actor_ids) if actor_ids and not self._carla_connected else set()
        try:
            # WorldSnapshot is tied to the committed synchronous frame.  Actor
            # objects returned before a destroy command can keep is_alive=True
            # for a stale frame and must not be used as destruction evidence.
            snapshot_actor_ids = {
                actor_snapshot.id for actor_snapshot in self._world.get_snapshot()
            }
            remaining = set(actor_ids) & snapshot_actor_ids
            remaining |= self._find_owned_actor_ids() & snapshot_actor_ids
        except Exception as exc:
            self._monitor.record_exception("actor_removal_verification", exc)
            return set(actor_ids)
        return remaining

    def _abort_episode_initialization(self, exc: BaseException) -> bool:
        self._monitor.record_event(
            "episode_initialization_aborted", episode=self._episode_id,
            exception_type=type(exc).__name__, exception=str(exc),
        )
        try:
            self._episode_cleanup()
        except Exception as cleanup_error:
            self._monitor.record_exception("episode_initialization_cleanup", cleanup_error)
            self._recover_carla("initialization_cleanup", cleanup_error)
            return True
        return False

    def _actor_counts(self) -> dict[str, dict[str, int]]:
        by_role: dict[str, int] = {}
        by_type: dict[str, int] = {}
        if not self._world or not self._carla_connected:
            return {"by_role_name": by_role, "by_type_id": by_type}
        try:
            for actor in self._world.get_actors():
                role = actor.attributes.get("role_name", "") or "(none)"
                by_role[role] = by_role.get(role, 0) + 1
                type_id = actor.type_id
                by_type[type_id] = by_type.get(type_id, 0) + 1
        except Exception as exc:
            self._monitor.record_exception("actor_count_snapshot", exc)
        return {"by_role_name": by_role, "by_type_id": by_type}

    def _update_monitor_state(self, *, refresh_actors: bool) -> None:
        try:
            now = time.monotonic()
            if self._carla_proc is not None:
                carla_pid = self._carla_proc.pid
            else:
                # psutil.net_connections() is a global socket scan.  Cache
                # externally managed CARLA's PID at the monitor cadence, not
                # on every 10 Hz environment step.
                if now - self._last_monitor_pid_refresh >= 5.0:
                    self._cached_external_carla_pid = pid_listening_on_port(self.cfg.carla.port)
                    self._last_monitor_pid_refresh = now
                carla_pid = self._cached_external_carla_pid
            state: dict[str, Any] = {
                "episode": self._episode_id,
                "environment_step": self._episode_step,
                "processes": {
                    "python_pid": os.getpid(),
                    "carla_pid": carla_pid,
                    "sumo_pid": self._sumo_proc.pid if self._sumo_proc else None,
                },
                "ports": {
                    "carla_rpc": self.cfg.carla.port,
                    "carla_streaming": self.cfg.carla.port + 1,
                    "traffic_manager": self.cfg.carla.tm_port,
                    "sumo_traci": self.cfg.sumo.port if self.cfg.sumo else None,
                },
                "scenario": self.cfg.scenario_id,
                "wz": self.cfg.wz_id,
                "layout": self.cfg.layout_id,
                "setting": self.cfg.setting_id,
                "traffic_backend": self.cfg.traffic_backend,
            }
            if refresh_actors or now - self._last_monitor_actor_refresh >= 5.0:
                state["actor_counts"] = self._actor_counts()
                self._last_monitor_actor_refresh = now
            self._monitor.update_state(**state)
        except Exception as exc:
            self._monitor.record_exception("monitor_state_update", exc)

    def runtime_diagnostics(self) -> dict[str, Any]:
        """Small worker-side snapshot safe to fetch from a vector environment."""
        self._update_monitor_state(refresh_actors=True)
        return {
            "worker": self._worker_id,
            "scenario": self._scenario_name,
            "episode": self._episode_id,
            "environment_step": self._episode_step,
            "monitor_path": self._monitor.path,
            "actor_counts": self._actor_counts(),
            "last_traci_command": self._last_traci_command,
        }

    # ------------------------------------------------------------------ #
    #  Worker-owned CARLA server recovery                                 #
    # ------------------------------------------------------------------ #

    def _format_carla_command(self) -> list[str]:
        if not self.cfg.carla.server_command:
            # Merely knowing an externally launched server's command is not
            # authority to start another instance.  This becomes true only
            # after we have stopped the captured server safely (or verified it
            # already exited and the worker port is clear).
            if self._captured_external_launch_authorized:
                return list(self._observed_external_carla_command or [])
            return []
        substitutions = {
            "port": self.cfg.carla.port,
            "tm_port": self.cfg.carla.tm_port,
            "scenario": self._scenario_name,
            "town": self.cfg.carla.town,
        }
        try:
            return [str(part).format(**substitutions) for part in self.cfg.carla.server_command]
        except KeyError as exc:
            raise CarlaRuntimeFault(
                f"Invalid CARLA server_command placeholder {exc}; allowed: {sorted(substitutions)}"
            ) from exc

    def _restart_observed_external_carla(self, reason: str) -> bool:
        """Stop only the previously verified CARLA process for this worker port.

        We capture the PID and command while the server is healthy.  At
        recovery, the port owner, creation time, and complete command line
        must still match before this method will terminate anything. The same
        captured command becomes launchable only after a safe stop or a
        confirmed already-exited process with a clear worker port.
        """
        expected_pid = self._observed_external_carla_pid
        expected_command = self._observed_external_carla_command
        expected_create_time = self._observed_external_carla_create_time
        self._captured_external_launch_authorized = False
        if expected_pid is None or not expected_command or expected_create_time is None:
            self._monitor.record_event(
                "external_carla_restart_refused",
                reason=reason,
                refusal="no_verified_capture",
            )
            return False

        listener_pid = pid_listening_on_port(self.cfg.carla.port)
        if listener_pid not in (None, expected_pid):
            self._monitor.record_event(
                "external_carla_restart_refused",
                reason=reason,
                expected_pid=expected_pid,
                listener_pid=listener_pid,
            )
            return False

        current_identity = process_identity_for_pid(expected_pid)
        if current_identity is None:
            # The verified external process is gone. A cleared worker port
            # makes starting the previously captured command safe; do not act
            # if process inspection itself is unavailable.
            if listener_pid is None and pid_exists(expected_pid) is False:
                self._cached_external_carla_pid = None
                self._captured_external_launch_authorized = True
                self._monitor.record_event(
                    "external_carla_restart_authorized",
                    reason=reason,
                    expected_pid=expected_pid,
                    basis="captured_process_already_exited_port_clear",
                )
                return True
            self._monitor.record_event(
                "external_carla_restart_refused",
                reason=reason,
                expected_pid=expected_pid,
                refusal="process_identity_unavailable",
            )
            return False

        command_matches = current_identity["command"] == expected_command
        create_time_matches = current_identity["create_time"] == expected_create_time
        if not command_matches or not create_time_matches:
            self._monitor.record_event(
                "external_carla_restart_refused",
                reason=reason,
                expected_pid=expected_pid,
                command_matches=command_matches,
                create_time_matches=create_time_matches,
            )
            return False

        result = terminate_pid(expected_pid)
        self._monitor.record_event("external_carla_process_stopped", reason=reason, **result)
        listener_after_stop = pid_listening_on_port(self.cfg.carla.port)
        if result.get("terminated") and listener_after_stop is None:
            self._cached_external_carla_pid = None
            self._captured_external_launch_authorized = True
            self._monitor.record_event(
                "external_carla_restart_authorized",
                reason=reason,
                expected_pid=expected_pid,
                basis="verified_process_stopped_port_clear",
            )
            return True
        self._monitor.record_event(
            "external_carla_restart_refused",
            reason=reason,
            expected_pid=expected_pid,
            refusal="termination_or_port_clearance_failed",
            listener_after_stop=listener_after_stop,
        )
        return False

    def _launch_carla_process(self) -> None:
        command = self._format_carla_command()
        if not command:
            raise CarlaRuntimeFault("Cannot launch CARLA without configured or captured server command")
        if self._carla_proc and self._carla_proc.poll() is None:
            return
        listener_pid = pid_listening_on_port(self.cfg.carla.port)
        if listener_pid is not None:
            raise CarlaRuntimeFault(
                f"Refusing to launch CARLA: PID {listener_pid} is already listening on "
                f"worker port {self.cfg.carla.port}."
            )

        self._carla_generation += 1
        log_dir = os.path.join(_PROJECT_ROOT, "logs")
        self._carla_log_path = os.path.join(
            log_dir, f"{self._sumo_label}_carla_g{self._carla_generation:04d}.log",
        )
        self._carla_log_file = open(self._carla_log_path, "a", encoding="utf-8", buffering=1)
        self._carla_log_file.write(
            f"\n===== CARLA START {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
            f"worker={self._worker_id} scenario={self._scenario_name} command={command!r} =====\n"
        )
        try:
            self._carla_proc = subprocess.Popen(
                command,
                stdout=self._carla_log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            self._carla_log_file.close()
            self._carla_log_file = None
            raise CarlaRuntimeFault(f"Could not launch worker-owned CARLA: {exc}") from exc

        self._monitor.increment("carla_process_starts")
        self._monitor.record_event(
            "carla_process_started", pid=self._carla_proc.pid, command=command,
            log_path=self._carla_log_path, generation=self._carla_generation,
        )
        logger.warning(
            "Worker-owned CARLA launched | worker=%d scenario=%s pid=%d port=%d log=%s",
            self._worker_id, self._scenario_name, self._carla_proc.pid,
            self.cfg.carla.port, self._carla_log_path,
        )

    def _terminate_carla_process(self, reason: str) -> None:
        proc = self._carla_proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=20)
            details = _returncode_details(proc.returncode)
            self._monitor.record_event(
                "carla_process_stopped", pid=proc.pid, reason=reason, **details,
            )
            logger.warning(
                "Worker-owned CARLA stopped | worker=%d scenario=%s pid=%d reason=%s %s",
                self._worker_id, self._scenario_name, proc.pid, reason, details,
            )
        except Exception as exc:
            self._monitor.record_exception("carla_process_terminate", exc, reason=reason, pid=proc.pid)
        finally:
            self._carla_proc = None
            self._cached_external_carla_pid = None
            if self._carla_log_file is not None:
                try:
                    self._carla_log_file.write(
                        f"===== CARLA END {time.strftime('%Y-%m-%dT%H:%M:%S%z')} reason={reason} =====\n"
                    )
                    self._carla_log_file.close()
                except Exception:
                    pass
                self._carla_log_file = None

    def _recover_carla(self, reason: str, exc: BaseException) -> bool:
        """Reconnect/restart this worker's CARLA only; never kill an external PID."""
        self._carla_restart_count += 1
        self._monitor.increment("carla_restarts")
        self._monitor.record_event(
            "carla_recovery_begin", reason=reason, restart_count=self._carla_restart_count,
            exception_type=type(exc).__name__, exception=str(exc),
        )
        self._carla_connected = False
        self._client = None
        self._world = None
        self._cmap = None
        if self._carla_proc is not None:
            self._terminate_carla_process(reason=f"recovery:{reason}")
        else:
            restart_authorized = self._restart_observed_external_carla(reason=f"recovery:{reason}")
            if not restart_authorized:
                self._monitor.record_event(
                    "carla_recovery_restart_blocked",
                    reason=reason,
                    port=self.cfg.carla.port,
                )
        try:
            self.connect()
            # A CARLA RPC fault invalidates actor handles.  The subsequent
            # reset does a role-based orphan sweep, which is safer than using
            # stale numeric IDs against a reconnected/restarted server.
            self._episode_actors.clear()
            if self._conn is not None:
                self._resilient_build_sync_components()
            self._monitor.record_event("carla_recovery_complete", reason=reason)
            self._update_monitor_state(refresh_actors=True)
            return True
        except Exception as recovery_error:
            self._monitor.record_exception("carla_recovery", recovery_error, reason=reason)
            logger.error(
                "CARLA recovery pending | worker=%d scenario=%s port=%d error=%s",
                self._worker_id, self._scenario_name, self.cfg.carla.port, recovery_error,
            )
            return False

    # ------------------------------------------------------------------ #
    #  SUMO process, TraCI capture, and restart                           #
    # ------------------------------------------------------------------ #

    def _install_traci_command_recorder(self, conn: Any) -> None:
        """Record the final low-level TraCI command without a growing trace file."""
        original = getattr(conn, "_sendCmd", None)
        if not callable(original) or getattr(conn, "_rl_command_recorder", False):
            return

        def recorded_send_cmd(*args: Any, **kwargs: Any) -> Any:
            command = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "method": "Connection._sendCmd",
                "args": _short_repr(args),
                "kwargs": _short_repr(kwargs),
            }
            self._last_traci_command = command
            self._monitor.set_last_traci_command(command)
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                self._monitor.record_exception("traci_send_cmd", exc, command=command)
                raise

        try:
            setattr(conn, "_sendCmd", recorded_send_cmd)
            setattr(conn, "_rl_command_recorder", True)
        except Exception as exc:
            self._monitor.record_exception("traci_command_recorder_install", exc)

        original_simulation_step = getattr(conn, "simulationStep", None)
        if not callable(original_simulation_step):
            return

        def recorded_simulation_step(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_simulation_step(*args, **kwargs)
            finally:
                self._monitor.record_timing("simulationStep", time.perf_counter() - started)

        try:
            setattr(conn, "simulationStep", recorded_simulation_step)
        except Exception as exc:
            self._monitor.record_exception("traci_simulation_step_timer_install", exc)

    def _stderr_tail(self, limit_bytes: int = 8192) -> str:
        if not self._sumo_stderr_path:
            return ""
        try:
            if self._sumo_stderr_file is not None:
                self._sumo_stderr_file.flush()
            with open(self._sumo_stderr_path, "rb") as source:
                source.seek(0, os.SEEK_END)
                start = max(0, source.tell() - limit_bytes)
                source.seek(start)
                return source.read().decode("utf-8", errors="replace")
        except Exception as tail_error:
            return f"<could not read SUMO stderr tail: {tail_error}>"

    def _record_sumo_failure(self, phase: str, exc: BaseException | None) -> None:
        proc = self._sumo_proc
        returncode = proc.poll() if proc is not None else self._last_sumo_returncode
        details = _returncode_details(returncode)
        stderr_tail = self._stderr_tail()
        event = {
            "phase": phase,
            "sumo_pid": proc.pid if proc else None,
            "stderr_log": self._sumo_stderr_path,
            "stderr_tail": stderr_tail,
            "last_traci_command": self._last_traci_command,
            **details,
        }
        if exc is not None:
            event.update({"exception_type": type(exc).__name__, "exception": str(exc)})
        self._monitor.record_event("sumo_failure", **event)
        logger.error("SUMO failure diagnostics | %s", event)

    def _resilient_reload_sumo(self) -> str:
        if self._sumo_proc is None:
            self._launch_sumo_process()
            self._connect_traci()
            self._resilient_build_sync_components()
            return "initial_launch"

        if not self._sumo_is_alive():
            self._record_sumo_failure("connection_broken_before_reload", None)
            self._terminate_sumo(reason="connection_broken_before_reload")
            self._monitor.increment("sumo_restarts")
            self._launch_sumo_process()
            self._connect_traci()
            self._resilient_rebind_sync_components()
            return "sumo_connection_broken"

        try:
            started = time.perf_counter()
            self._conn.load(self._sumo_option_args())
            self._monitor.record_timing("sumo.load", time.perf_counter() - started)
            self._resilient_build_sync_components()
            return "reload"
        except _SUMO_TRANSPORT_EXCEPTIONS as exc:
            self._record_sumo_failure("sumo_load", exc)
            self._terminate_sumo(reason="load_failed")
            self._monitor.increment("sumo_restarts")
            self._launch_sumo_process()
            self._connect_traci()
            self._resilient_rebind_sync_components()
            return "sumo_connection_broken"

    def _resilient_build_sync_components(self) -> None:
        if self.cfg.sumo is None:
            raise SumoRuntimeFault(f"{self.cfg.setting_id} is CARLA-only")
        if not self._world or not self._conn:
            raise SumoRuntimeFault("Cannot build synchronizers without CARLA world and TraCI connection")
        net_file = self.cfg.sumo.net_file
        if not os.path.isabs(net_file):
            net_file = os.path.join(_PROJECT_ROOT, net_file)
        net_offset = CarlaSumoCoordinateBridge.net_offset_from_net(net_file)
        self._bridge = CarlaSumoCoordinateBridge(
            net_offset=net_offset,
            lateral_shift=self.cfg.carla.lateral_shift,
        )
        self._ego_proxy = EgoProxySynchronizer(self._bridge, self._conn, self.cfg.sumo.ego_proxy_id)
        # wz = self.cfg.workzone
        # road_ref = ((wz.x_min + wz.x_max) / 2.0, (wz.y_min + wz.y_max) / 2.0)
        # Pedestrian proxy intentionally disabled.  Keep the implementation in
        # sync/pedestrian_proxy.py in case a future SUMO-backed pedestrian
        # scenario needs it again.
        # self._ped_proxy = PedestrianProxySynchronizer(
        #     self._bridge, self._conn,
        #     road_ref=road_ref,
        #     road_heading_deg=self.cfg.carla.road_heading_deg,
        #     road_half_width=self.cfg.carla.road_half_width,
        #     proxy_prefix=self.cfg.sumo.ped_proxy_prefix,
        # )
        self._bg_traffic = BackgroundTrafficSynchronizer(
            self._bridge,
            self._world,
            self._conn,
            worker_id=self._worker_id,
            actor_register=self._track_episode_actor,
            actor_unregister=self._untrack_episode_actor,
            spawn_attempt_callback=self._record_mirror_spawn_attempt,
        )
        self._monitor.record_event("sync_components_built", net_file=net_file, net_offset=net_offset)

    def _resilient_rebind_sync_components(self) -> None:
        if not self._conn:
            raise SumoRuntimeFault("Cannot rebind synchronizers without TraCI")
        if not self._ego_proxy or not self._bg_traffic:
            self._resilient_build_sync_components()
            return
        self._ego_proxy.set_connection(self._conn)
        # self._ped_proxy.set_connection(self._conn)  # pedestrian proxy disabled
        self._bg_traffic.set_connection(self._conn)

    # ------------------------------------------------------------------ #
    #  Timed simulator operations                                         #
    # ------------------------------------------------------------------ #

    def _world_tick(self, phase: str) -> int:
        if not self._world:
            raise CarlaRuntimeFault("world.tick requested without a CARLA world")
        started = time.perf_counter()
        try:
            return self._world.tick()
        except Exception as exc:
            self._monitor.record_exception(
                "world.tick", exc, phase=phase, episode=self._episode_id, environment_step=self._episode_step,
            )
            self._monitor.record_event("world_tick_timeout_or_failure", phase=phase, error=str(exc))
            raise CarlaRuntimeFault(f"world.tick failed during {phase}: {exc}") from exc
        finally:
            self._monitor.record_timing("world.tick", time.perf_counter() - started)

    def _simulation_step(self, phase: str) -> None:
        if not self._conn:
            raise SumoRuntimeFault("simulationStep requested without TraCI connection")
        started = time.perf_counter()
        try:
            self._conn.simulationStep()
        except _SUMO_TRANSPORT_EXCEPTIONS as exc:
            self._record_sumo_failure(f"simulationStep:{phase}", exc)
            raise SumoRuntimeFault(f"SUMO simulationStep failed during {phase}: {exc}") from exc
        finally:
            self._monitor.record_timing("simulationStep_env_phase", time.perf_counter() - started)

# ── helpers ─────────────────────────────────────────────────────────────────

def _loc_str(loc: carla.Location) -> str:
    return f"({loc.x:.1f}, {loc.y:.1f})"


def _canonical_town_name(name: str) -> str:
    """Normalize CARLA's optional ``_Opt`` map suffix for comparisons."""
    normalized = name.strip().lower()
    return normalized[:-4] if normalized.endswith("_opt") else normalized


def _short_repr(value: Any, limit: int = 1000) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _returncode_details(returncode: int | None) -> dict[str, int | str | None]:
    if returncode is None:
        return {"returncode": None, "returncode_hex": None}
    return {
        "returncode": returncode,
        "returncode_hex": f"0x{(returncode & 0xFFFFFFFF):08X}",
    }


def _looks_like_carla_fault(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "world.tick",
        "time-out",
        "timeout",
        "simulator",
        "carla",
        "rpc",
        "connection refused",
        "connection reset",
    )
    return isinstance(exc, (CarlaRuntimeFault, ConnectionError, BrokenPipeError)) or any(marker in text for marker in markers)
