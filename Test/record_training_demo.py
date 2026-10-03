"""Record one complete frozen-policy episode on the canonical training split.

The default is the paper's joint Seed27/P32 checkpoint on ``s4/wz1/b``.
Two RGB sensors (a fixed work-zone overview and an ego-mounted spring-arm
chase camera) are recorded during the *same* simulator rollout.  A completed
episode is retained regardless of whether its terminal reason is success or a
policy failure.  Simulator-infrastructure terminal reasons are discarded and
retried before a manifest is committed.

This entry point never launches CARLA and never touches the Test result queue.
Start a dedicated CARLA server with the required town before running it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import queue
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from baseline.PPO.runtime_v2 import configure_initial_config
from baseline.PPO.validate_allSwithoneExe import OrderedOriginSelector, OriginBinder
from config.scenario_catalog import resolve_setting, scenario_root
from config.scenario_config import ScenarioConfig, load_scenario
from config.scenario_selector import CoverageSelector
from Test.run_policy_test import _OriginTrackingWrapper, _sha256
from validation_debug.runner_v2 import _outcome_flags_v2, _validate_model_spaces_v2


DEFAULT_SETTING = "s4/wz1/b"
DEFAULT_POLICY_NAME = "joint_seed027_p32"
DEFAULT_MODEL = (
    WORKSPACE_ROOT
    / "Aug24_ppo_OldOD_MultiSeed"
    / "runs_center_ego_old_od_seed27_r1"
    / "models"
    / "policy_update_032.zip"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Test" / "demos" / "training"
DEFAULT_SEED = 2704
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_BITRATE_KBPS = 8_000
ENV_MODULE = "env.gym_wrapper_center_v2"


@dataclass(frozen=True)
class CameraPose:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_deg: float
    source: str


# Live-tuneable paper-demo preset for the first requested training setting.
# It frames the pedestrian path, warning sign, and WZ1-B cone line while the
# ego travels in the +x direction.  Other settings use the geometry-derived
# fallback below until a live preset is approved.
OVERVIEW_PRESETS: dict[str, CameraPose] = {
    "s4/wz1/b": CameraPose(
        # WZ1-B sits beneath the Town10HD station canopy.  Keep this fixed
        # camera below the canopy and frame the cone line/crossing locally;
        # the synchronized chase camera retains the full approach and exit.
        eye=(-16.0, 136.5, 4.5),
        target=(11.0, 140.0, 0.8),
        fov_deg=75.0,
        source="training_preset_upstream_under_canopy_v4",
    ),
}

CHASE_LOCATION_M = (-7.5, 0.0, 4.0)
CHASE_ROTATION_DEG = (-1.0, 0.0, 0.0)  # pitch, yaw, roll
CHASE_FOV_DEG = 85.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--setting", default=DEFAULT_SETTING)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--origin-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2080)
    parser.add_argument("--tm-port", type=int, default=8080)
    parser.add_argument("--sumo-port", type=int, default=8893)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--overview-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="override fixed overview camera world location in metres",
    )
    parser.add_argument(
        "--overview-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="override fixed overview look-at point in metres",
    )
    parser.add_argument(
        "--overview-fov",
        type=float,
        help="override fixed overview horizontal field of view in degrees",
    )
    parser.add_argument("--bitrate-kbps", type=int, default=DEFAULT_BITRATE_KBPS)
    parser.add_argument("--max-file-mib", type=float, default=256.0)
    parser.add_argument("--max-infra-retries", type=int, default=10)
    parser.add_argument(
        "--camera-pair-timeout",
        type=float,
        default=30.0,
        help="maximum wait when both GPU camera streams fall behind the rollout",
    )
    parser.add_argument(
        "--camera-flush-timeout",
        type=float,
        default=10.0,
        help="wait for delayed GPU-camera callbacks after the terminal tick",
    )
    parser.add_argument("--ffmpeg", help="ffmpeg executable; auto-detected")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--allow-shared-training-port",
        action="store_true",
        help="explicitly allow a port normally reserved by a training worker",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without CARLA, ffmpeg, or output files",
    )
    return parser


def _canonical_town(value: str) -> str:
    result = str(value).strip().lower()
    return result[:-4] if result.endswith("_opt") else result


def _safe_name(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    result = "".join(character if character in allowed else "_" for character in value)
    result = result.strip("_-")
    if not result:
        raise ValueError("--policy-name must contain a filename-safe character")
    return result


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _overview_pose(cfg: ScenarioConfig) -> CameraPose:
    preset = OVERVIEW_PRESETS.get(cfg.setting_id)
    if preset is not None:
        return preset

    points: list[tuple[float, float]] = [
        (float(x), float(y)) for x, y in cfg.workzone.traffic_cones
    ]
    points.extend((float(x), float(y)) for x, y in cfg.workzone.warning_signs)
    points.extend(
        (
            (float(cfg.workzone.x_min), float(cfg.workzone.y_min)),
            (float(cfg.workzone.x_min), float(cfg.workzone.y_max)),
            (float(cfg.workzone.x_max), float(cfg.workzone.y_min)),
            (float(cfg.workzone.x_max), float(cfg.workzone.y_max)),
        )
    )
    if cfg.jaywalker is not None:
        for xyz in (
            cfg.jaywalker.spawn,
            cfg.jaywalker.disappear,
            cfg.jaywalker.trigger_anchor,
        ):
            points.append((float(xyz[0]), float(xyz[1])))
    center_x = (min(x for x, _ in points) + max(x for x, _ in points)) * 0.5
    center_y = (min(y for _, y in points) + max(y for _, y in points)) * 0.5
    span = max(
        max(x for x, _ in points) - min(x for x, _ in points),
        max(y for _, y in points) - min(y for _, y in points),
        10.0,
    )
    heading = math.radians(float(cfg.carla.road_heading_deg))
    forward = (math.cos(heading), math.sin(heading))
    left = (-math.sin(heading), math.cos(heading))
    forward_offset = max(14.0, span * 0.75)
    lateral_offset = max(12.0, span * 0.65)
    eye = (
        center_x + forward[0] * forward_offset + left[0] * lateral_offset,
        center_y + forward[1] * forward_offset + left[1] * lateral_offset,
        max(12.0, span * 0.65),
    )
    return CameraPose(
        eye=eye,
        target=(center_x, center_y, 0.8),
        fov_deg=90.0,
        source="geometry_fallback_v1",
    )


def _look_at_transform(carla: Any, pose: CameraPose) -> Any:
    dx = pose.target[0] - pose.eye[0]
    dy = pose.target[1] - pose.eye[1]
    dz = pose.target[2] - pose.eye[2]
    return carla.Transform(
        carla.Location(x=pose.eye[0], y=pose.eye[1], z=pose.eye[2]),
        carla.Rotation(
            pitch=math.degrees(math.atan2(dz, math.hypot(dx, dy))),
            yaw=math.degrees(math.atan2(dy, dx)),
        ),
    )


class _SensorFrameSink:
    """Bounded callback queue retaining CARLA frame ids for later pairing."""

    def __init__(self) -> None:
        # CARLA GPU sensors are documented/observed to trail the world tick by
        # a few frames.  Keep a modest buffer instead of requiring the callback
        # for world frame N to arrive before the policy may advance to N+1.
        self.frames: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=8)
        self.callback_drops = 0

    def __call__(self, image: Any) -> None:
        item = (int(image.frame), bytes(image.raw_data))
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            try:
                self.frames.get_nowait()
                self.callback_drops += 1
            except queue.Empty:
                pass
            self.frames.put_nowait(item)

    def drain(self) -> list[tuple[int, bytes]]:
        items: list[tuple[int, bytes]] = []
        while True:
            try:
                items.append(self.frames.get_nowait())
            except queue.Empty:
                return items

    def wait_one(self, timeout_s: float) -> tuple[int, bytes] | None:
        try:
            return self.frames.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            return None


class _FramePairSynchronizer:
    """Pair overview/chase images by sensor frame, never by callback timing.

    A synchronous ``world.tick()`` does not make GPU-camera callbacks
    synchronous with Python.  In particular, waiting immediately for the
    current world frame can deadlock the rollout during sensor warm-up.  This
    coordinator lets the policy advance while callbacks trail, and emits a
    video frame only when *both* cameras supplied the identical CARLA frame id.
    """

    VIEWS = ("overview", "chase")
    MAX_UNPAIRED_PER_VIEW = 16
    MAX_WORLD_FRAME_LAG = 4

    def __init__(self, sinks: dict[str, _SensorFrameSink]) -> None:
        self.sinks = sinks
        self.pending: dict[str, dict[int, bytes]] = {
            view: {} for view in self.VIEWS
        }
        self.paired_frame_ids: list[int] = []

    def _ingest(self) -> None:
        for view in self.VIEWS:
            for frame_id, payload in self.sinks[view].drain():
                if self.paired_frame_ids and frame_id <= self.paired_frame_ids[-1]:
                    continue
                self.pending[view][frame_id] = payload
            if len(self.pending[view]) > self.MAX_UNPAIRED_PER_VIEW:
                oldest = min(self.pending[view])
                newest = max(self.pending[view])
                raise RuntimeError(
                    f"{view} camera stream diverged: {len(self.pending[view])} "
                    f"unpaired frames ({oldest}..{newest})"
                )

    def collect(self) -> list[tuple[int, dict[str, bytes]]]:
        self._ingest()
        common = sorted(set(self.pending["overview"]) & set(self.pending["chase"]))
        pairs: list[tuple[int, dict[str, bytes]]] = []
        for frame_id in common:
            if self.paired_frame_ids and frame_id <= self.paired_frame_ids[-1]:
                for view in self.VIEWS:
                    self.pending[view].pop(frame_id, None)
                continue
            pairs.append(
                (
                    frame_id,
                    {
                        view: self.pending[view].pop(frame_id)
                        for view in self.VIEWS
                    },
                )
            )
            self.paired_frame_ids.append(frame_id)
        return pairs

    def wait_for_pair(
        self,
        *,
        timeout_s: float,
    ) -> list[tuple[int, dict[str, bytes]]]:
        """Wait for any exact overview/chase frame pair, not a chosen frame."""
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            pairs = self.collect()
            if pairs:
                return pairs
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            for view in self.VIEWS:
                item = self.sinks[view].wait_one(min(0.05, remaining))
                if item is not None:
                    frame_id, payload = item
                    if not self.paired_frame_ids or frame_id > self.paired_frame_ids[-1]:
                        self.pending[view][frame_id] = payload
        pairs = self.collect()
        if pairs:
            return pairs
        latest = {
            view: max(frames) if frames else None
            for view, frames in self.pending.items()
        }
        raise RuntimeError(
            "timed out waiting for a common overview/chase sensor frame; "
            f"latest_unpaired={latest}"
        )

    def wait_for_terminal(
        self,
        terminal_world_frame: int,
        *,
        timeout_s: float,
    ) -> list[tuple[int, dict[str, bytes]]]:
        """Flush delayed callbacks after termination without ticking the world."""
        deadline = time.monotonic() + float(timeout_s)
        pairs: list[tuple[int, dict[str, bytes]]] = []
        while time.monotonic() < deadline:
            pairs.extend(self.collect())
            if (
                self.paired_frame_ids
                and self.paired_frame_ids[-1] >= terminal_world_frame
            ):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            # Wait briefly on each callback queue.  No simulator tick occurs,
            # so this cannot alter the frozen-policy episode.
            for view in self.VIEWS:
                item = self.sinks[view].wait_one(min(0.05, remaining))
                if item is not None:
                    frame_id, payload = item
                    if not self.paired_frame_ids or frame_id > self.paired_frame_ids[-1]:
                        self.pending[view][frame_id] = payload
            pairs.extend(self.collect())
        return pairs

    @property
    def diagnostics(self) -> dict[str, Any]:
        ids = self.paired_frame_ids
        gap_count = sum(
            max(0, current - previous - 1)
            for previous, current in zip(ids, ids[1:])
        )
        return {
            "paired_frame_count": len(ids),
            "first_sensor_frame": ids[0] if ids else None,
            "last_sensor_frame": ids[-1] if ids else None,
            "sensor_frame_gaps": gap_count,
            "callback_queue_drops": {
                view: self.sinks[view].callback_drops for view in self.VIEWS
            },
            "unpaired_frames_at_close": {
                view: len(self.pending[view]) for view in self.VIEWS
            },
        }


def _spawn_camera_pair(
    base_env: gym.Env,
    cfg: ScenarioConfig,
    pose: CameraPose,
    *,
    width: int,
    height: int,
) -> tuple[dict[str, Any], dict[str, _SensorFrameSink]]:
    import carla

    world = base_env.engine.world
    ego = base_env.engine.ego
    if world is None or ego is None:
        raise RuntimeError("CARLA world/ego is unavailable after reset")

    library = world.get_blueprint_library()
    blueprint = library.find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("sensor_tick", f"{float(cfg.episode.sim_dt):.9f}")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "qualitative_demo_rgb")

    actors: dict[str, Any] = {}
    sinks: dict[str, _SensorFrameSink] = {}
    try:
        blueprint.set_attribute("fov", str(pose.fov_deg))
        sinks["overview"] = _SensorFrameSink()
        actors["overview"] = world.spawn_actor(
            blueprint,
            _look_at_transform(carla, pose),
        )
        actors["overview"].listen(sinks["overview"])

        blueprint.set_attribute("fov", str(CHASE_FOV_DEG))
        chase_transform = carla.Transform(
            carla.Location(
                x=CHASE_LOCATION_M[0],
                y=CHASE_LOCATION_M[1],
                z=CHASE_LOCATION_M[2],
            ),
            carla.Rotation(
                pitch=CHASE_ROTATION_DEG[0],
                yaw=CHASE_ROTATION_DEG[1],
                roll=CHASE_ROTATION_DEG[2],
            ),
        )
        sinks["chase"] = _SensorFrameSink()
        actors["chase"] = world.spawn_actor(
            blueprint,
            chase_transform,
            attach_to=ego,
            attachment_type=carla.AttachmentType.SpringArmGhost,
        )
        actors["chase"].listen(sinks["chase"])
        world.get_spectator().set_transform(actors["overview"].get_transform())
    except Exception:
        _destroy_actors(actors)
        raise
    return actors, sinks


def _destroy_actors(actors: dict[str, Any]) -> None:
    for actor in reversed(tuple(actors.values())):
        try:
            actor.stop()
        except Exception:
            pass
        try:
            if actor.is_alive:
                actor.destroy()
        except Exception:
            pass


def _require_carla(host: str, port: int, expected_town: str, timeout_s: float) -> None:
    try:
        import carla

        client = carla.Client(str(host), int(port))
        client.set_timeout(float(timeout_s))
        current_town = client.get_world().get_map().name.split("/")[-1]
    except Exception as exc:
        raise RuntimeError(f"CARLA is not ready at {host}:{port}: {exc}") from exc
    if _canonical_town(current_town) != _canonical_town(expected_town):
        raise RuntimeError(
            f"CARLA {host}:{port} has {current_town}; {expected_town} is required. "
            "This recorder does not change or launch the CARLA world."
        )


def _validate_plan(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], ScenarioConfig, Path, str, CameraPose]:
    record, materialized_data = resolve_setting(args.setting)
    cfg = load_scenario(record.setting_id)
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint does not exist: {model_path}")
    policy_name = _safe_name(args.policy_name)
    if not 0 <= args.origin_index < len(cfg.origin.spawn_points):
        raise ValueError(
            f"--origin-index must be in [0, {len(cfg.origin.spawn_points) - 1}]"
        )
    if args.width < 320 or args.height < 180:
        raise ValueError("recording resolution is too small")
    if args.bitrate_kbps < 1_000:
        raise ValueError("--bitrate-kbps must be at least 1000 for the master video")
    if (
        args.max_file_mib <= 0.0
        or args.max_infra_retries < 0
        or args.camera_pair_timeout <= 0.0
        or args.camera_flush_timeout <= 0.0
    ):
        raise ValueError("invalid output/retry limit")
    if not math.isclose(
        float(cfg.episode.sim_dt) * float(cfg.episode.sync_hz),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{cfg.setting_id} has inconsistent sim_dt/sync_hz: "
            f"{cfg.episode.sim_dt}/{cfg.episode.sync_hz}"
        )
    if not float(cfg.episode.sync_hz).is_integer():
        raise ValueError("video writer requires an integer native simulator rate")
    if (
        args.carla_port in {2000, 2020, 2030, 2040}
        and not args.allow_shared_training_port
    ):
        raise ValueError(
            f"CARLA port {args.carla_port} is reserved by a training worker; "
            "use the dedicated default 2080 or explicitly allow sharing"
        )
    pose = _overview_pose(cfg)
    if (
        args.overview_eye is not None
        or args.overview_target is not None
        or args.overview_fov is not None
    ):
        eye = tuple(args.overview_eye) if args.overview_eye is not None else pose.eye
        target = (
            tuple(args.overview_target)
            if args.overview_target is not None
            else pose.target
        )
        fov = float(args.overview_fov) if args.overview_fov is not None else pose.fov_deg
        values = (*eye, *target, fov)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("overview camera values must all be finite")
        if not 5.0 <= fov <= 170.0:
            raise ValueError("--overview-fov must be between 5 and 170 degrees")
        if math.dist(eye, target) < 0.1:
            raise ValueError("overview eye and target must be different")
        pose = CameraPose(
            eye=eye,
            target=target,
            fov_deg=fov,
            source=f"cli_override_from_{pose.source}",
        )
    return record, materialized_data, cfg, model_path, policy_name, pose


def _print_plan(
    args: argparse.Namespace,
    record: Any,
    materialized_data: dict[str, Any],
    cfg: ScenarioConfig,
    model_path: Path,
    policy_name: str,
    pose: CameraPose,
) -> None:
    fps = int(round(cfg.episode.sync_hz))
    print("TRAINING DEMO PLAN", flush=True)
    print(f"  split: training", flush=True)
    print(f"  setting: {cfg.setting_id} ({cfg.carla.town}, {cfg.traffic_backend})", flush=True)
    print(f"  origin_index: {args.origin_index}", flush=True)
    print(f"  policy: {policy_name}", flush=True)
    print(f"  model: {model_path}", flush=True)
    print(f"  model_sha256: {_sha256(model_path)}", flush=True)
    print(f"  config: {record.config_path}", flush=True)
    print(f"  config_sha256: {_sha256(record.config_path)}", flush=True)
    print(f"  materialized_config_sha256: {_json_sha256(materialized_data)}", flush=True)
    print(f"  capture: two simultaneous RGB cameras, {args.width}x{args.height}@{fps}Hz", flush=True)
    print(
        f"  overview[{pose.source}]: eye={pose.eye} target={pose.target} fov={pose.fov_deg}",
        flush=True,
    )
    print(
        f"  chase: location={CHASE_LOCATION_M} rotation={CHASE_ROTATION_DEG} "
        f"fov={CHASE_FOV_DEG} SpringArmGhost",
        flush=True,
    )
    print("  completed success or policy failure: retain", flush=True)
    print("  infrastructure terminal: discard both views and retry", flush=True)


def _record(
    args: argparse.Namespace,
    record: Any,
    materialized_data: dict[str, Any],
    cfg: ScenarioConfig,
    model_path: Path,
    policy_name: str,
    pose: CameraPose,
) -> Path:
    # Imports below this point are intentionally excluded from --dry-run.
    from Test.record_policy_demos import _Mp4Writer, _check_ffmpeg, _resolve_ffmpeg
    from validation_debug.visuals_v2 import ValidationWorkZonePropsV2

    ffmpeg = _resolve_ffmpeg(args.ffmpeg)
    _check_ffmpeg(ffmpeg)
    _require_carla(args.host, args.carla_port, cfg.carla.town, args.connect_timeout)
    configure_initial_config(
        cfg,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        sumo_port=args.sumo_port,
        no_rendering=False,
    )
    cfg.carla.host = str(args.host)

    all_origins = tuple(cfg.origin.spawn_points)
    selected_origin = all_origins[args.origin_index]
    # OrderedOriginSelector labels a one-element tuple as origin 0.  Preserve
    # the canonical index separately in the manifest and filenames.
    selector = OrderedOriginSelector([cfg.setting_id], {cfg.setting_id: 1}, repeats=1)
    binder = OriginBinder(selector, {cfg.setting_id: (selected_origin,)})

    def episode_setup(setting_id: str) -> None:
        binder.bind(setting_id)

    env_type = importlib.import_module(ENV_MODULE).CarlaSumoGymEnv
    base_env = env_type(
        scenario=cfg.setting_id,
        config=cfg,
        mode="eval",
        worker_id=0,
        no_rendering_mode=False,
        scenario_selector=selector,
        episode_setup_callback=episode_setup,
    )
    binder.attach(base_env)
    env = _OriginTrackingWrapper(base_env, selector)

    @dataclass(frozen=True)
    class _VisualCase:
        config: ScenarioConfig
        materialized_data: dict[str, Any]

    props = ValidationWorkZonePropsV2(
        _VisualCase(config=cfg, materialized_data=materialized_data),
        enabled=True,
        cones_only=False,
    )
    model = PPO.load(str(model_path), device=args.device)
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    _validate_model_spaces_v2(model, env)

    started = datetime.now().astimezone()
    timestamp = started.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    setting_slug = cfg.setting_id.replace("/", "_")
    run_dir = Path(args.output_root).expanduser().resolve() / (
        f"{setting_slug}__{policy_name}__{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    fps = int(round(cfg.episode.sync_hz))
    infra_faults: dict[str, int] = {}
    first_reset = True
    accepted: dict[str, Any] | None = None
    partial_paths_created: set[Path] = set()
    try:
        while accepted is None:
            observation, _ = env.reset(seed=args.seed if first_reset else None)
            first_reset = False
            world = base_env.engine.world
            if world is None:
                raise RuntimeError("CARLA world is unavailable after reset")
            props.start(world)
            attempt_number = 1 + sum(infra_faults.values())
            partials = {
                view: run_dir / f"._attempt{attempt_number:02d}_{view}.mp4"
                for view in ("overview", "chase")
            }
            partial_paths_created.update(partials.values())
            for path in partials.values():
                path.unlink(missing_ok=True)
            actors: dict[str, Any] = {}
            writers: dict[str, Any] = {}
            steps = 0
            episode_return = 0.0
            reason = "running"
            synchronizer: _FramePairSynchronizer | None = None
            terminal_world_frame: int | None = None
            try:
                actors, sinks = _spawn_camera_pair(
                    base_env,
                    cfg,
                    pose,
                    width=args.width,
                    height=args.height,
                )
                for view in ("overview", "chase"):
                    writers[view] = _Mp4Writer(
                        ffmpeg,
                        partials[view],
                        width=args.width,
                        height=args.height,
                        fps=fps,
                        bitrate_kbps=args.bitrate_kbps,
                    )
                synchronizer = _FramePairSynchronizer(sinks)
                done = False
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(
                        np.asarray(action, dtype=np.float32).reshape(-1)
                    )
                    steps += 1
                    episode_return += float(reward)
                    reason = str(info.get("reason", "running"))
                    done = bool(terminated or truncated)
                    current_world = base_env.engine.world
                    if current_world is None:
                        raise RuntimeError("CARLA world disappeared during recording")
                    world_frame = int(current_world.get_snapshot().frame)
                    terminal_world_frame = world_frame
                    # GPU camera callbacks can lag behind synchronous world
                    # ticks.  Drain what is ready and pair strictly by the
                    # sensor-provided CARLA frame id; never block this tick on
                    # an exact callback id.
                    paired = synchronizer.collect()
                    last_sensor_frame = (
                        synchronizer.paired_frame_ids[-1]
                        if synchronizer.paired_frame_ids
                        else None
                    )
                    # Permit two initial ticks for sensor registration/warm-up.
                    # Thereafter throttle only when the paired stream is more
                    # than a few world frames behind.  The wait accepts *any*
                    # exact common sensor frame, so it cannot deadlock on a
                    # callback id that a newly spawned camera never emitted.
                    if not paired and steps >= 2 and (
                        last_sensor_frame is None
                        or world_frame - last_sensor_frame
                        > synchronizer.MAX_WORLD_FRAME_LAG
                    ):
                        paired.extend(
                            synchronizer.wait_for_pair(
                                timeout_s=args.camera_pair_timeout
                            )
                        )
                    for _, frames in paired:
                        for view in ("overview", "chase"):
                            writers[view].write(frames[view])

                if terminal_world_frame is None:
                    raise RuntimeError("episode ended without a CARLA world frame")
                for _, frames in synchronizer.wait_for_terminal(
                    terminal_world_frame,
                    timeout_s=args.camera_flush_timeout,
                ):
                    for view in ("overview", "chase"):
                        writers[view].write(frames[view])
                sync_diagnostics = synchronizer.diagnostics
                if sync_diagnostics["paired_frame_count"] < 1:
                    raise RuntimeError(
                        "the two cameras produced no common CARLA sensor frame"
                    )
            finally:
                _destroy_actors(actors)
                close_errors = []
                for view, writer in writers.items():
                    try:
                        writer.close()
                    except Exception as exc:
                        close_errors.append(f"{view}: {exc}")
                if close_errors:
                    raise RuntimeError("; ".join(close_errors))

            if reason in CoverageSelector.INFRASTRUCTURE_REASONS:
                infra_faults[reason] = infra_faults.get(reason, 0) + 1
                for path in partials.values():
                    path.unlink(missing_ok=True)
                print(
                    f"INFRA_RETRY attempt={attempt_number} reason={reason} ",
                    f"count={sum(infra_faults.values())}/{args.max_infra_retries}",
                    flush=True,
                )
                if sum(infra_faults.values()) > args.max_infra_retries:
                    raise RuntimeError("training demo exceeded infrastructure retry limit")
                continue

            filenames = {
                view: (
                    f"training_{setting_slug}_origin{args.origin_index}_"
                    f"{policy_name}_{view}.mp4"
                )
                for view in ("overview", "chase")
            }
            for view in ("overview", "chase"):
                final_path = run_dir / filenames[view]
                partials[view].replace(final_path)
                size_mib = final_path.stat().st_size / (1024 * 1024)
                if size_mib > args.max_file_mib:
                    raise RuntimeError(
                        f"{view} video is {size_mib:.1f} MiB, above the "
                        f"{args.max_file_mib:.1f} MiB safety cap"
                    )
            accepted = {
                "attempt": attempt_number,
                "origin_index": int(args.origin_index),
                "episode_steps": steps,
                "duration_s": round(steps * float(cfg.episode.sim_dt), 3),
                "terminal_world_frame": terminal_world_frame,
                "episode_return": episode_return,
                "terminal_reason": reason,
                "outcome": _outcome_flags_v2(reason),
                "sensor_sync": sync_diagnostics,
                "videos": {
                    view: {
                        "file": filenames[view],
                        "size_bytes": (run_dir / filenames[view]).stat().st_size,
                        "frames": sync_diagnostics["paired_frame_count"],
                        "duration_s": round(
                            sync_diagnostics["paired_frame_count"] / fps,
                            3,
                        ),
                    }
                    for view in ("overview", "chase")
                },
            }
    except BaseException:
        # Only remove files created with this run's explicit partial names.
        # Never recurse and never touch a prior timestamped demo directory.
        for path in partial_paths_created:
            path.unlink(missing_ok=True)
        try:
            run_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        try:
            props.close()
        finally:
            env.close()

    manifest_path = run_dir / "demo_manifest.json"
    scenario_manifest = scenario_root(cfg.scenario_id) / "manifest.json"
    scenario_source = scenario_root(cfg.scenario_id) / "scenario.py"
    controller_source = scenario_root(cfg.scenario_id) / "jaywalker_controller.py"
    manifest = {
        "dataset_split": "training",
        "purpose": "illustrative_demo_not_evaluation",
        "setting_id": cfg.setting_id,
        "scenario_id": cfg.scenario_id,
        "wz_id": cfg.wz_id,
        "layout_id": cfg.layout_id,
        "town": cfg.carla.town,
        "traffic_backend": cfg.traffic_backend,
        "runtime_mode": "eval_with_frozen_policy",
        "policy_name": policy_name,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "rollout_seed": int(args.seed),
        "deterministic_policy": True,
        "inputs": {
            "scenario_manifest": str(scenario_manifest.resolve()),
            "scenario_manifest_sha256": _sha256(scenario_manifest),
            "config": str(record.config_path.resolve()),
            "config_sha256": _sha256(record.config_path),
            "materialized_config_sha256": _json_sha256(materialized_data),
            "scenario_source": str(scenario_source.resolve()),
            "scenario_source_sha256": _sha256(scenario_source),
            "jaywalker_controller": (
                str(controller_source.resolve()) if controller_source.is_file() else None
            ),
            "jaywalker_controller_sha256": (
                _sha256(controller_source) if controller_source.is_file() else None
            ),
            "sumo_network": str(record.network_path.resolve()) if record.network_path else None,
            "sumo_network_sha256": _sha256(record.network_path) if record.network_path else None,
            "sumo_route": str(record.route_path.resolve()) if record.route_path else None,
            "sumo_route_sha256": _sha256(record.route_path) if record.route_path else None,
        },
        "capture": {
            "width": int(args.width),
            "height": int(args.height),
            "fps": fps,
            "native_simulator_rate": True,
            "sim_dt_s": float(cfg.episode.sim_dt),
            "codec": "H.264/yuv420p",
            "bitrate_kbps": int(args.bitrate_kbps),
            "simultaneous_same_rollout": True,
            "overview": {
                "eye": list(pose.eye),
                "target": list(pose.target),
                "fov_deg": pose.fov_deg,
                "source": pose.source,
            },
            "chase": {
                "attachment": "SpringArmGhost",
                "relative_location_m": list(CHASE_LOCATION_M),
                "relative_rotation_deg_pitch_yaw_roll": list(CHASE_ROTATION_DEG),
                "fov_deg": CHASE_FOV_DEG,
            },
        },
        "protocol": {
            "retain_completed_success_or_policy_failure": True,
            "infrastructure_failures_excluded_and_retried": True,
            "infrastructure_faults": dict(sorted(infra_faults.items())),
        },
        "episode": accepted,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"TRAINING DEMO SAVED setting={cfg.setting_id} origin={args.origin_index} "
        f"reason={accepted['terminal_reason']} dir={run_dir}",
        flush=True,
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _validate_plan(args)
    _print_plan(args, *plan)
    if args.dry_run:
        print("Dry-run complete: no CARLA connection and no files written.", flush=True)
        return 0
    manifest = _record(args, *plan)
    print(f"manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
