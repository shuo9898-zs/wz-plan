"""Capture publication stills from the canonical *training* scenarios.

This utility is intentionally separate from validation/test evaluation and
from the demo-video recorders.  It captures one lossless fixed-world RGB PNG
at a named work-zone event, using a conspicuous cyan Tesla Model 3 as the ego.
An optional ego-mounted chase PNG is captured at the exact same CARLA sensor
frame.  Policy mode rolls the frozen joint Seed27/P32 policy to termination so
the manifest can retain the episode terminal reason; staged mode is available
for purely illustrative scene composition and is labelled as such.

The draft paper matrix selects layout B once per work zone: S1/S3/S5 WZ1-3
and S4 WZ1-2.  Representative S2/WZ1 and S6/WZ1 presets are also available
for six-scenario overview figures.  Run one setting at a time against a
dedicated CARLA server already loaded with the required town.  This script
never launches CARLA and never reads or writes the Test result queues.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import queue
import sys
import time
import types
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
from PIL import Image
from stable_baselines3 import PPO

from baseline.PPO.runtime_v2 import configure_initial_config
from baseline.PPO.validate_allSwithoneExe import OrderedOriginSelector, OriginBinder
from config.scenario_catalog import resolve_setting, scenario_root
from config.scenario_config import ScenarioConfig, load_scenario
from config.scenario_selector import CoverageSelector
from Test.record_training_demo import (
    CameraPose,
    _destroy_actors,
    _json_sha256,
    _look_at_transform,
    _require_carla,
    _safe_name,
)
from Test.run_policy_test import _OriginTrackingWrapper, _sha256
from validation_debug.runner_v2 import _outcome_flags_v2, _validate_model_spaces_v2


DEFAULT_POLICY_NAME = "joint_seed027_p32"
DEFAULT_MODEL = (
    WORKSPACE_ROOT
    / "Aug24_ppo_OldOD_MultiSeed"
    / "runs_center_ego_old_od_seed27_r1"
    / "models"
    / "policy_update_032.zip"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Test" / "paper_stills" / "training"
DEFAULT_SEED = 2704
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_EGO_BLUEPRINT = "vehicle.tesla.model3"
DEFAULT_EGO_COLOR_RGB = (0, 170, 255)
ENV_MODULE = "env.gym_wrapper_center_v2"

CHASE_LOCATION_M = (-7.5, 0.0, 4.0)
CHASE_ROTATION_DEG = (-1.0, 0.0, 0.0)  # pitch, yaw, roll
CHASE_FOV_DEG = 85.0


@dataclass(frozen=True)
class StillPreset:
    camera: CameraPose
    event_target_xy: tuple[float, float]
    event_name: str = "closest-target"
    max_workzone_distance_m: float = 12.0


# One selected layout (B) per work zone.  S4/WZ1/B is the approved low,
# upstream station-canopy composition.  S2/WZ1 and S6/WZ1 are representative
# overview-only additions for a complete S1--S6 scenario panel.
PAPER_STILL_PRESETS: dict[str, StillPreset] = {
    "s1/wz1/b": StillPreset(
        CameraPose((188.00, 237.00, 5.00), (193.20, 206.00, 0.80), 75.0, "paper_oblique_qa_v2"),
        (193.70, 223.00),
    ),
    "s1/wz2/b": StillPreset(
        CameraPose((126.00, 108.00, 5.00), (100.00, 105.70, 0.80), 75.0, "paper_oblique_qa_v2"),
        (112.20, 105.70),
    ),
    "s1/wz3/b": StillPreset(
        CameraPose((130.75, 187.80, 4.80), (103.75, 191.00, 0.80), 75.0, "paper_oblique_qa_v2"),
        (116.00, 191.70),
    ),
    "s2/wz1/b": StillPreset(
        CameraPose((-198.00, 112.00, 6.00), (-181.00, 145.00, 0.80), 80.0, "scenario_overview_initial_v1"),
        (-193.00, 126.00),
        max_workzone_distance_m=18.0,
    ),
    "s3/wz1/b": StillPreset(
        CameraPose((-38.00, -61.15, 6.50), (10.00, -57.65, 0.80), 75.0, "paper_oblique_v1"),
        (-12.00, -57.65),
    ),
    "s3/wz2/b": StillPreset(
        CameraPose((78.00, 9.20, 6.50), (30.00, 12.70, 0.80), 75.0, "paper_oblique_mirrored_qa_v2"),
        (52.00, 12.70),
    ),
    "s3/wz3/b": StillPreset(
        CameraPose((-112.00, 8.00, 8.00), (-92.00, -42.00, 0.80), 80.0, "paper_oblique_v1"),
        (-106.65, -15.00),
    ),
    "s4/wz1/b": StillPreset(
        CameraPose((-16.00, 136.50, 4.50), (11.00, 140.00, 0.80), 75.0, "approved_station_upstream_v4"),
        (-7.00, 139.90),
        max_workzone_distance_m=20.0,
    ),
    "s4/wz2/b": StillPreset(
        CameraPose((-45.85, 40.00, 4.50), (-49.35, 67.00, 0.80), 75.0, "paper_oblique_v1"),
        (-49.35, 55.00),
    ),
    "s5/wz1/b": StillPreset(
        CameraPose((32.00, -70.00, 5.50), (25.85, -43.00, 0.80), 78.0, "paper_oblique_qa_v2"),
        (29.00, -58.50),
        max_workzone_distance_m=15.0,
    ),
    "s5/wz2/b": StillPreset(
        CameraPose((31.50, 13.00, 5.00), (25.20, 40.00, 0.80), 78.0, "paper_oblique_qa_v2"),
        (28.10, 25.00),
    ),
    "s5/wz3/b": StillPreset(
        CameraPose((30.70, 109.00, 4.50), (27.20, 136.00, 0.80), 75.0, "paper_oblique_v1"),
        (27.20, 124.00),
    ),
    "s6/wz1/b": StillPreset(
        CameraPose((132.00, 306.50, 5.50), (98.00, 303.20, 0.80), 76.0, "scenario_overview_initial_v1"),
        (119.00, 303.00),
        max_workzone_distance_m=17.0,
    ),
    "s6/wz3/b": StillPreset(
        CameraPose((122.00, 243.50, 5.50), (98.50, 237.65, 0.80), 76.0, "scenario_overview_initial_v1"),
        (112.00, 237.65),
        max_workzone_distance_m=17.0,
    ),
}
PAPER_TRAINING_SETTINGS = tuple(PAPER_STILL_PRESETS)


@dataclass(frozen=True)
class _VisualCase:
    config: ScenarioConfig
    materialized_data: dict[str, Any]


class _SensorFrameSink:
    """Small bounded callback queue retaining raw BGRA and CARLA frame id."""

    def __init__(self) -> None:
        self.frames: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=12)
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
        result: list[tuple[int, bytes]] = []
        while True:
            try:
                result.append(self.frames.get_nowait())
            except queue.Empty:
                return result

    def wait_one(self, timeout_s: float) -> tuple[int, bytes] | None:
        try:
            return self.frames.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            return None


class _ViewSynchronizer:
    """Return only exact common sensor-frame ids for all requested views."""

    MAX_WORLD_FRAME_LAG = 4
    MAX_UNPAIRED_PER_VIEW = 20

    def __init__(self, sinks: dict[str, _SensorFrameSink]) -> None:
        if not sinks:
            raise ValueError("at least one camera sink is required")
        self.sinks = sinks
        self.views = tuple(sinks)
        self.pending: dict[str, dict[int, bytes]] = {view: {} for view in self.views}
        self.emitted_ids: list[int] = []

    def _ingest(self) -> None:
        for view in self.views:
            for frame_id, payload in self.sinks[view].drain():
                if self.emitted_ids and frame_id <= self.emitted_ids[-1]:
                    continue
                self.pending[view][frame_id] = payload
            if len(self.pending[view]) > self.MAX_UNPAIRED_PER_VIEW:
                raise RuntimeError(
                    f"{view} camera has {len(self.pending[view])} unpaired frames"
                )

    def collect(self) -> list[tuple[int, dict[str, bytes]]]:
        self._ingest()
        common = set(self.pending[self.views[0]])
        for view in self.views[1:]:
            common.intersection_update(self.pending[view])
        result: list[tuple[int, dict[str, bytes]]] = []
        for frame_id in sorted(common):
            result.append(
                (
                    frame_id,
                    {view: self.pending[view].pop(frame_id) for view in self.views},
                )
            )
            self.emitted_ids.append(frame_id)
        return result

    def wait_for_pair(self, timeout_s: float) -> list[tuple[int, dict[str, bytes]]]:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            pairs = self.collect()
            if pairs:
                return pairs
            remaining = deadline - time.monotonic()
            for view in self.views:
                item = self.sinks[view].wait_one(min(0.05, max(0.0, remaining)))
                if item is not None:
                    frame_id, payload = item
                    if not self.emitted_ids or frame_id > self.emitted_ids[-1]:
                        self.pending[view][frame_id] = payload
        pairs = self.collect()
        if pairs:
            return pairs
        latest = {
            view: max(frames) if frames else None
            for view, frames in self.pending.items()
        }
        raise RuntimeError(
            "timed out waiting for a common paper-still sensor frame; "
            f"latest_unpaired={latest}"
        )

    def flush_through(
        self,
        world_frame: int,
        timeout_s: float,
    ) -> list[tuple[int, dict[str, bytes]]]:
        deadline = time.monotonic() + float(timeout_s)
        result: list[tuple[int, dict[str, bytes]]] = []
        while time.monotonic() < deadline:
            result.extend(self.collect())
            if self.emitted_ids and self.emitted_ids[-1] >= world_frame:
                break
            remaining = deadline - time.monotonic()
            for view in self.views:
                item = self.sinks[view].wait_one(min(0.05, max(0.0, remaining)))
                if item is not None:
                    frame_id, payload = item
                    if not self.emitted_ids or frame_id > self.emitted_ids[-1]:
                        self.pending[view][frame_id] = payload
        result.extend(self.collect())
        return result

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "views": list(self.views),
            "paired_frame_count": len(self.emitted_ids),
            "first_sensor_frame": self.emitted_ids[0] if self.emitted_ids else None,
            "last_sensor_frame": self.emitted_ids[-1] if self.emitted_ids else None,
            "callback_queue_drops": {
                view: self.sinks[view].callback_drops for view in self.views
            },
            "unpaired_frames_at_close": {
                view: len(self.pending[view]) for view in self.views
            },
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--setting", default="s4/wz1/b", choices=PAPER_TRAINING_SETTINGS)
    parser.add_argument("--list-settings", action="store_true")
    parser.add_argument("--capture-mode", choices=("policy", "staged"), default="staged")
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
    parser.add_argument("--include-chase", action="store_true")
    parser.add_argument(
        "--manual-spectator",
        action="store_true",
        help="do not overwrite the spectator transform when an episode restarts",
    )
    parser.add_argument(
        "--draw-workzone-box",
        action="store_true",
        help="draw a cyan debug outline and label around the configured work-zone bounds",
    )
    parser.add_argument("--ego-blueprint", default=DEFAULT_EGO_BLUEPRINT)
    parser.add_argument(
        "--ego-color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=DEFAULT_EGO_COLOR_RGB,
    )
    parser.add_argument("--camera-eye", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-target", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-fov", type=float)
    parser.add_argument(
        "--capture-event",
        choices=("preset", "closest-target", "workzone-entry", "episode-step"),
        default="preset",
    )
    parser.add_argument("--event-target", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--capture-step", type=int)
    parser.add_argument("--entry-margin-m", type=float, default=2.0)
    parser.add_argument(
        "--pause-at-workzone",
        action="store_true",
        help="freeze the synchronous rollout at the first work-zone entry and wait for Enter",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="pace synchronous policy steps to the scenario simulation clock (approximately 1x)",
    )
    parser.add_argument("--max-workzone-distance-m", type=float)
    parser.add_argument("--camera-pair-timeout", type=float, default=30.0)
    parser.add_argument("--camera-flush-timeout", type=float, default=10.0)
    parser.add_argument("--max-infra-retries", type=int, default=10)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--allow-shared-training-port",
        action="store_true",
        help="explicitly allow a port normally reserved by a training worker",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without CARLA or output files",
    )
    return parser


def _distance_to_workzone(cfg: ScenarioConfig, x: float, y: float) -> float:
    dx = max(float(cfg.workzone.x_min) - x, 0.0, x - float(cfg.workzone.x_max))
    dy = max(float(cfg.workzone.y_min) - y, 0.0, y - float(cfg.workzone.y_max))
    return math.hypot(dx, dy)


def _resolve_event_and_camera(
    args: argparse.Namespace,
    preset: StillPreset,
) -> tuple[str, tuple[float, float], float, CameraPose]:
    event_name = preset.event_name if args.capture_event == "preset" else args.capture_event
    if args.capture_mode == "staged" and event_name != "closest-target":
        raise ValueError(
            "staged capture supports the preset/closest-target event only; "
            "use policy mode for workzone-entry or episode-step selection"
        )
    if event_name == "episode-step" and args.capture_step is None:
        raise ValueError("--capture-event episode-step requires --capture-step")
    if args.capture_step is not None and args.capture_step < 0:
        raise ValueError("--capture-step must be non-negative")
    event_target = (
        tuple(float(value) for value in args.event_target)
        if args.event_target is not None
        else preset.event_target_xy
    )
    max_wz_distance = (
        float(args.max_workzone_distance_m)
        if args.max_workzone_distance_m is not None
        else float(preset.max_workzone_distance_m)
    )
    pose = preset.camera
    if args.camera_eye is not None or args.camera_target is not None or args.camera_fov is not None:
        pose = CameraPose(
            eye=tuple(args.camera_eye) if args.camera_eye is not None else pose.eye,
            target=tuple(args.camera_target) if args.camera_target is not None else pose.target,
            fov_deg=float(args.camera_fov) if args.camera_fov is not None else pose.fov_deg,
            source=f"cli_override_from_{pose.source}",
        )
    numeric = (*event_target, max_wz_distance, *pose.eye, *pose.target, pose.fov_deg)
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("event and camera values must be finite")
    if max_wz_distance < 0.0 or args.entry_margin_m < 0.0:
        raise ValueError("work-zone distance limits must be non-negative")
    if not 5.0 <= float(pose.fov_deg) <= 170.0:
        raise ValueError("--camera-fov must be between 5 and 170 degrees")
    if math.dist(pose.eye, pose.target) < 0.1:
        raise ValueError("camera eye and target must differ")
    return event_name, event_target, max_wz_distance, pose


def _validate_plan(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], ScenarioConfig, Path | None, str, str, tuple[float, float], float, CameraPose]:
    record, materialized_data = resolve_setting(args.setting)
    cfg = load_scenario(record.setting_id)
    preset = PAPER_STILL_PRESETS[cfg.setting_id]
    model_path: Path | None = None
    if args.capture_mode == "policy":
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {model_path}")
    policy_name = _safe_name(args.policy_name)
    if not 0 <= args.origin_index < len(cfg.origin.spawn_points):
        raise ValueError(
            f"--origin-index must be in [0, {len(cfg.origin.spawn_points) - 1}]"
        )
    if args.width < 640 or args.height < 360:
        raise ValueError("paper-still resolution must be at least 640x360")
    if any(value < 0 or value > 255 for value in args.ego_color):
        raise ValueError("--ego-color RGB channels must be in [0, 255]")
    if args.max_infra_retries < 0 or args.camera_pair_timeout <= 0.0 or args.camera_flush_timeout <= 0.0:
        raise ValueError("invalid retry/camera timeout")
    if args.carla_port in {2000, 2020, 2030, 2040} and not args.allow_shared_training_port:
        raise ValueError(
            f"CARLA port {args.carla_port} is reserved by a training worker; "
            "use the dedicated default 2080 or explicitly allow sharing"
        )
    event_name, event_target, max_wz_distance, pose = _resolve_event_and_camera(args, preset)
    return (
        record,
        materialized_data,
        cfg,
        model_path,
        policy_name,
        event_name,
        event_target,
        max_wz_distance,
        pose,
    )


def _print_plan(args: argparse.Namespace, plan: tuple[Any, ...]) -> None:
    (
        record,
        materialized_data,
        cfg,
        model_path,
        policy_name,
        event_name,
        event_target,
        max_wz_distance,
        pose,
    ) = plan
    print("TRAINING PAPER-STILL PLAN", flush=True)
    print(f"  setting: {cfg.setting_id} ({cfg.carla.town}, {cfg.traffic_backend})", flush=True)
    print(f"  split/purpose: training / illustrative paper figure", flush=True)
    print(f"  mode: {args.capture_mode}", flush=True)
    print(f"  origin_index/seed: {args.origin_index}/{args.seed}", flush=True)
    print(f"  ego: {args.ego_blueprint} color={tuple(args.ego_color)}", flush=True)
    if model_path is not None:
        print(f"  policy: {policy_name}", flush=True)
        print(f"  model: {model_path}", flush=True)
        print(f"  model_sha256: {_sha256(model_path)}", flush=True)
    print(f"  config: {record.config_path}", flush=True)
    print(f"  config_sha256: {_sha256(record.config_path)}", flush=True)
    print(f"  materialized_config_sha256: {_json_sha256(materialized_data)}", flush=True)
    print(
        f"  event: {event_name} target_xy={event_target} "
        f"max_wz_distance={max_wz_distance}m",
        flush=True,
    )
    print(
        f"  fixed spectator: eye={pose.eye} target={pose.target} "
        f"fov={pose.fov_deg} source={pose.source}",
        flush=True,
    )
    views = "overview+chase (exact frame paired)" if args.include_chase else "overview"
    print(f"  output: {args.width}x{args.height} lossless RGB PNG, views={views}", flush=True)
    print("  Test queues/results: untouched", flush=True)


def _install_paper_ego_spawner(engine: Any, blueprint_id: str, color_rgb: tuple[int, int, int]) -> None:
    """Override only this environment instance's pre-reset ego spawn method."""
    color_text = ",".join(str(int(value)) for value in color_rgb)
    # Import only after ENV_MODULE has loaded the V2 SUMO runtime and pinned
    # the matching TraCI tools.  Importing this legacy engine at module import
    # time would make even --dry-run depend on a machine-wide SUMO install.
    legacy_env = importlib.import_module("env.carla_sumo_env")

    def _spawn_paper_ego(self: Any, mode: str, origin_t: Any) -> tuple[Any, Any]:
        if not self._world:
            raise legacy_env.CarlaRuntimeFault(
                "Cannot spawn paper-still ego without a CARLA world"
            )
        library = self._world.get_blueprint_library()
        blueprint = library.find(blueprint_id)
        if blueprint is None:
            raise legacy_env.CarlaRuntimeFault(
                f"Missing requested ego blueprint: {blueprint_id}"
            )
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", self.cfg.ego_role or "ego")
        if not blueprint.has_attribute("color"):
            raise legacy_env.CarlaRuntimeFault(
                f"Requested ego blueprint has no color attribute: {blueprint_id}"
            )
        blueprint.set_attribute("color", color_text)
        for attempt in range(1, 11):
            actor = self._world.try_spawn_actor(blueprint, origin_t)
            self._monitor.increment("spawn_attempts")
            if actor is not None:
                self._track_episode_actor(actor, "ego")
                self._monitor.record_event(
                    "spawn_attempt", actor_kind="ego", attempt=attempt, success=True
                )
                return actor, origin_t
            self._monitor.increment("spawn_collisions")
            self._monitor.record_event(
                "spawn_attempt",
                actor_kind="ego",
                attempt=attempt,
                success=False,
                reason="collision",
            )
            origin_t = self._od_sampler.sample_origin(mode)
        raise legacy_env.SpawnExhaustedError(
            "Paper-still ego spawn was occupied for all 10 attempts"
        )

    engine._spawn_ego = types.MethodType(_spawn_paper_ego, engine)


def _spawn_cameras(
    base_env: gym.Env,
    cfg: ScenarioConfig,
    pose: CameraPose,
    *,
    width: int,
    height: int,
    include_chase: bool,
    manual_spectator: bool,
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
        blueprint.set_attribute("role_name", "paper_still_rgb")

    actors: dict[str, Any] = {}
    sinks: dict[str, _SensorFrameSink] = {}
    try:
        blueprint.set_attribute("fov", str(pose.fov_deg))
        sinks["overview"] = _SensorFrameSink()
        actors["overview"] = world.spawn_actor(blueprint, _look_at_transform(carla, pose))
        actors["overview"].listen(sinks["overview"])
        if not manual_spectator:
            world.get_spectator().set_transform(actors["overview"].get_transform())

        if include_chase:
            blueprint.set_attribute("fov", str(CHASE_FOV_DEG))
            relative = carla.Transform(
                carla.Location(*CHASE_LOCATION_M),
                carla.Rotation(
                    pitch=CHASE_ROTATION_DEG[0],
                    yaw=CHASE_ROTATION_DEG[1],
                    roll=CHASE_ROTATION_DEG[2],
                ),
            )
            sinks["chase"] = _SensorFrameSink()
            actors["chase"] = world.spawn_actor(
                blueprint,
                relative,
                attach_to=ego,
                attachment_type=carla.AttachmentType.SpringArmGhost,
            )
            actors["chase"].listen(sinks["chase"])
    except Exception:
        _destroy_actors(actors)
        raise
    return actors, sinks


def _draw_workzone_box(world: Any, cfg: ScenarioConfig) -> None:
    """Draw a visualization-only outline around the active work-zone AABB."""
    import carla

    x_min = float(cfg.workzone.x_min)
    x_max = float(cfg.workzone.x_max)
    y_min = float(cfg.workzone.y_min)
    y_max = float(cfg.workzone.y_max)
    center = carla.Location(
        x=(x_min + x_max) / 2.0,
        y=(y_min + y_max) / 2.0,
        z=0.35,
    )
    extent = carla.Vector3D(
        x=max((x_max - x_min) / 2.0, 0.05),
        y=max((y_max - y_min) / 2.0, 0.05),
        z=0.25,
    )
    color = carla.Color(r=0, g=255, b=255)
    world.debug.draw_box(
        carla.BoundingBox(center, extent),
        carla.Rotation(),
        thickness=0.12,
        color=color,
        life_time=0.0,
        persistent_lines=True,
    )
    world.debug.draw_string(
        carla.Location(x=center.x, y=center.y, z=1.2),
        "WORK ZONE",
        draw_shadow=True,
        color=color,
        life_time=0.0,
        persistent_lines=True,
    )


def _ego_frame_metadata(
    base_env: gym.Env,
    cfg: ScenarioConfig,
    *,
    step: int,
    reason: str,
    event_target: tuple[float, float],
) -> dict[str, Any]:
    ego = base_env.engine.ego
    if ego is None:
        raise RuntimeError("ego disappeared during paper-still capture")
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    x = float(transform.location.x)
    y = float(transform.location.y)
    return {
        "episode_step": int(step),
        "reason_at_frame": str(reason),
        "ego_location_m": [x, y, float(transform.location.z)],
        "ego_rotation_deg_pitch_yaw_roll": [
            float(transform.rotation.pitch),
            float(transform.rotation.yaw),
            float(transform.rotation.roll),
        ],
        "ego_speed_mps": math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2),
        "distance_to_event_target_m": math.hypot(x - event_target[0], y - event_target[1]),
        "distance_to_workzone_m": _distance_to_workzone(cfg, x, y),
        "inside_workzone_bbox": (
            cfg.workzone.x_min <= x <= cfg.workzone.x_max
            and cfg.workzone.y_min <= y <= cfg.workzone.y_max
        ),
    }


def _candidate_score(
    event_name: str,
    metadata: dict[str, Any],
    *,
    capture_step: int | None,
    entry_margin_m: float,
) -> tuple[float, int] | None:
    if event_name == "closest-target":
        # A staged ego is stationary, so all warm-up frames have the same
        # geometric score.  Prefer the latest equal-distance frame after TAA
        # and motion blur have settled.  Policy captures are unaffected unless
        # two frames have exactly the same target distance.
        return (float(metadata["distance_to_event_target_m"]), -int(metadata["episode_step"]))
    if event_name == "workzone-entry":
        if float(metadata["distance_to_workzone_m"]) > float(entry_margin_m):
            return None
        return (float(metadata["episode_step"]), int(metadata["episode_step"]))
    if event_name == "episode-step":
        assert capture_step is not None
        return (abs(int(metadata["episode_step"]) - capture_step), int(metadata["episode_step"]))
    raise ValueError(f"unsupported capture event: {event_name}")


def _update_candidate(
    candidate: dict[str, Any] | None,
    pairs: list[tuple[int, dict[str, bytes]]],
    frame_metadata: dict[int, dict[str, Any]],
    *,
    event_name: str,
    capture_step: int | None,
    entry_margin_m: float,
) -> dict[str, Any] | None:
    for frame_id, frames in pairs:
        metadata = frame_metadata.get(frame_id)
        if metadata is None:
            continue
        score = _candidate_score(
            event_name,
            metadata,
            capture_step=capture_step,
            entry_margin_m=entry_margin_m,
        )
        if score is None:
            continue
        if candidate is None or score < candidate["score"]:
            candidate = {
                "score": score,
                "sensor_frame": int(frame_id),
                "frames": frames,
                "metadata": metadata,
            }
    return candidate


def _stage_ego_at_event(
    base_env: gym.Env,
    cfg: ScenarioConfig,
    target_xy: tuple[float, float],
    *,
    authored_heading_deg: float | None = None,
) -> None:
    import carla

    world = base_env.engine.world
    ego = base_env.engine.ego
    if world is None or ego is None:
        raise RuntimeError("cannot stage paper ego before reset")
    query = carla.Location(x=target_xy[0], y=target_xy[1], z=2.0)
    waypoint = world.get_map().get_waypoint(query, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is not None:
        road_location = waypoint.transform.location
        # CARLA vehicle actor origins sit about 0.2 m above the road surface
        # once the suspension is settled.  Because this staged actor is frozen
        # immediately, place it directly at that resting height.
        z = float(road_location.z) + 0.20
        waypoint_yaw = float(waypoint.transform.rotation.yaw)
        configured_yaw = float(
            cfg.carla.road_heading_deg if authored_heading_deg is None else authored_heading_deg
        )
        # Some scenarios deliberately drive opposite CARLA's lane direction
        # (notably S1/WZ3).  Follow the authored experiment heading in that
        # case, while retaining the road tangent for genuinely curved routes.
        heading_error = abs((waypoint_yaw - configured_yaw + 180.0) % 360.0 - 180.0)
        if heading_error <= 45.0:
            stage_x = float(road_location.x)
            stage_y = float(road_location.y)
            yaw = waypoint_yaw
        else:
            # At junctions get_waypoint() can select a closer crossing road.
            # Keep the authored point and heading instead of moving the ego
            # onto that semantically unrelated lane.
            stage_x = float(target_xy[0])
            stage_y = float(target_xy[1])
            yaw = configured_yaw
    else:
        stage_x = float(target_xy[0])
        stage_y = float(target_xy[1])
        z = 0.5
        yaw = float(cfg.carla.road_heading_deg)
    # Freeze before teleporting.  In the high-traffic scenario a passing SUMO
    # actor can otherwise hit the illustrative ego during a multi-tick
    # suspension-settle window and rotate/push it out of the intended lane.
    ego.set_simulate_physics(False)
    ego.set_transform(
        carla.Transform(
            carla.Location(x=stage_x, y=stage_y, z=z),
            carla.Rotation(yaw=yaw),
        )
    )
    ego.set_target_velocity(carla.Vector3D())
    ego.set_target_angular_velocity(carla.Vector3D())
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))


def _stage_jaywalker_for_figure(base_env: gym.Env, cfg: ScenarioConfig) -> Any | None:
    """Place one frozen pedestrian in S4's authored crossing path."""
    import carla

    if cfg.jaywalker is None:
        return None
    world = base_env.engine.world
    if world is None:
        raise RuntimeError("cannot stage paper jaywalker without a CARLA world")
    blueprint = world.get_blueprint_library().find(cfg.jaywalker.blueprint)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "paper_still_jaywalker")
    # The collision target is the semantically meaningful crossing location.
    x, y, z = cfg.jaywalker.collision_target
    transform = carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(z) + 0.15),
        carla.Rotation(yaw=float(cfg.jaywalker.yaw_deg)),
    )
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        transform.location.z += 0.35
        actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        raise RuntimeError("could not stage the S4 paper-still jaywalker")
    actor.set_simulate_physics(False)
    return actor


def _save_rgb_png(path: Path, bgra: bytes, width: int, height: int) -> None:
    image = Image.frombytes("RGBA", (width, height), bgra, "raw", "BGRA").convert("RGB")
    image.save(path, format="PNG", compress_level=6)


def _record(
    args: argparse.Namespace,
    record: Any,
    materialized_data: dict[str, Any],
    cfg: ScenarioConfig,
    model_path: Path | None,
    policy_name: str,
    event_name: str,
    event_target: tuple[float, float],
    max_wz_distance: float,
    pose: CameraPose,
) -> Path:
    from validation_debug.visuals_v2 import ValidationWorkZonePropsV2

    _require_carla(args.host, args.carla_port, cfg.carla.town, args.connect_timeout)
    configure_initial_config(
        cfg,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        sumo_port=args.sumo_port,
        no_rendering=False,
    )
    cfg.carla.host = str(args.host)

    selected_origin = tuple(cfg.origin.spawn_points)[args.origin_index]
    # Capture this before reset: scenario setup may materialize/mutate runtime
    # configuration, but the selected authored origin remains the ground truth
    # for the intended direction of travel.
    authored_heading_deg = float(selected_origin.yaw_deg)
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
    _install_paper_ego_spawner(base_env.engine, args.ego_blueprint, tuple(args.ego_color))
    binder.attach(base_env)
    env = _OriginTrackingWrapper(base_env, selector)
    props = ValidationWorkZonePropsV2(
        _VisualCase(config=cfg, materialized_data=materialized_data),
        enabled=True,
        cones_only=False,
    )

    model: PPO | None = None
    if args.capture_mode == "policy":
        assert model_path is not None
        model = PPO.load(str(model_path), device=args.device)
        model.policy.set_training_mode(False)
        for parameter in model.policy.parameters():
            parameter.requires_grad_(False)
        _validate_model_spaces_v2(model, env)

    started = datetime.now().astimezone()
    setting_slug = cfg.setting_id.replace("/", "_")
    timestamp = started.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = Path(args.output_root).expanduser().resolve() / f"{setting_slug}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    infra_faults: dict[str, int] = {}
    accepted: dict[str, Any] | None = None
    output_files: dict[str, str] = {}
    ego_identity: dict[str, Any] | None = None
    first_reset = True
    try:
        while accepted is None:
            observation, _ = env.reset(seed=args.seed if first_reset else None)
            first_reset = False
            world = base_env.engine.world
            ego = base_env.engine.ego
            if world is None or ego is None:
                raise RuntimeError("CARLA world/ego unavailable after reset")
            actual_color = str(ego.attributes.get("color", ""))
            actual_color_rgb = tuple(
                int(component.strip()) for component in actual_color.split(",")
            ) if actual_color else ()
            if str(ego.type_id) != str(args.ego_blueprint):
                raise RuntimeError(
                    f"paper ego type mismatch: requested={args.ego_blueprint} "
                    f"actual={ego.type_id}"
                )
            if actual_color_rgb != tuple(args.ego_color):
                raise RuntimeError(
                    f"paper ego color mismatch: requested={tuple(args.ego_color)} "
                    f"actual={actual_color!r}"
                )
            ego_identity = {
                "type_id": str(ego.type_id),
                "actor_id": int(ego.id),
                "actual_color_attribute": actual_color,
                "actual_color_rgb": list(actual_color_rgb),
            }
            props.start(world)
            if args.draw_workzone_box:
                _draw_workzone_box(world, cfg)
            actors: dict[str, Any] = {}
            terminal_world_frame: int | None = None
            reason = "running"
            steps = 0
            episode_return = 0.0
            candidate: dict[str, Any] | None = None
            frame_metadata: dict[int, dict[str, Any]] = {}
            synchronizer: _ViewSynchronizer | None = None
            try:
                if args.capture_mode == "staged":
                    _stage_ego_at_event(
                        base_env,
                        cfg,
                        event_target,
                        authored_heading_deg=authored_heading_deg,
                    )
                    staged_walker = _stage_jaywalker_for_figure(base_env, cfg)
                    if staged_walker is not None:
                        actors["staged_jaywalker"] = staged_walker
                camera_actors, sinks = _spawn_cameras(
                    base_env,
                    cfg,
                    pose,
                    width=args.width,
                    height=args.height,
                    include_chase=args.include_chase,
                    manual_spectator=args.manual_spectator,
                )
                actors.update(camera_actors)
                synchronizer = _ViewSynchronizer(sinks)
                if args.capture_mode == "staged":
                    # Advance CARLA only long enough for GPU camera warm-up.
                    for staged_step in range(1, 13):
                        world_frame = int(world.tick())
                        frame_metadata[world_frame] = _ego_frame_metadata(
                            base_env,
                            cfg,
                            step=staged_step,
                            reason="staged",
                            event_target=event_target,
                        )
                        pairs = synchronizer.collect()
                        if not pairs and staged_step >= 2:
                            pairs = synchronizer.wait_for_pair(args.camera_pair_timeout)
                        candidate = _update_candidate(
                            candidate,
                            pairs,
                            frame_metadata,
                            event_name="closest-target",
                            capture_step=None,
                            entry_margin_m=args.entry_margin_m,
                        )
                    terminal_world_frame = int(world.get_snapshot().frame)
                    reason = "not_applicable_staged_capture"
                    steps = 0
                else:
                    assert model is not None
                    done = False
                    paused_at_workzone = False
                    next_wall_tick = time.perf_counter()
                    while not done:
                        action, _ = model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, info = env.step(
                            np.asarray(action, dtype=np.float32).reshape(-1)
                        )
                        steps += 1
                        episode_return += float(reward)
                        reason = str(info.get("reason", "running"))
                        done = bool(terminated or truncated)
                        world = base_env.engine.world
                        if world is None:
                            raise RuntimeError("CARLA world disappeared during still capture")
                        world_frame = int(world.get_snapshot().frame)
                        terminal_world_frame = world_frame
                        frame_metadata[world_frame] = _ego_frame_metadata(
                            base_env,
                            cfg,
                            step=steps,
                            reason=reason,
                            event_target=event_target,
                        )
                        current_metadata = frame_metadata[world_frame]
                        pairs = synchronizer.collect()
                        last_sensor_frame = (
                            synchronizer.emitted_ids[-1]
                            if synchronizer.emitted_ids
                            else None
                        )
                        if not pairs and steps >= 2 and (
                            last_sensor_frame is None
                            or world_frame - last_sensor_frame > synchronizer.MAX_WORLD_FRAME_LAG
                        ):
                            pairs = synchronizer.wait_for_pair(args.camera_pair_timeout)
                        candidate = _update_candidate(
                            candidate,
                            pairs,
                            frame_metadata,
                            event_name=event_name,
                            capture_step=args.capture_step,
                            entry_margin_m=args.entry_margin_m,
                        )
                        if (
                            args.pause_at_workzone
                            and not paused_at_workzone
                            and float(current_metadata["distance_to_workzone_m"])
                            <= float(args.entry_margin_m)
                        ):
                            paused_at_workzone = True
                            print(
                                "SCREENSHOT_READY "
                                f"setting={cfg.setting_id} step={steps} "
                                f"distance_to_workzone_m="
                                f"{current_metadata['distance_to_workzone_m']:.2f}; "
                                "CARLA is frozen. Press Enter to resume.",
                                flush=True,
                            )
                            input()
                            next_wall_tick = time.perf_counter()
                        if args.real_time:
                            next_wall_tick += float(cfg.episode.sim_dt)
                            remaining_s = next_wall_tick - time.perf_counter()
                            if remaining_s > 0.0:
                                time.sleep(remaining_s)

                if terminal_world_frame is None:
                    raise RuntimeError("capture ended without a CARLA world frame")
                pairs = synchronizer.flush_through(
                    terminal_world_frame,
                    args.camera_flush_timeout,
                )
                candidate = _update_candidate(
                    candidate,
                    pairs,
                    frame_metadata,
                    event_name=("closest-target" if args.capture_mode == "staged" else event_name),
                    capture_step=args.capture_step,
                    entry_margin_m=args.entry_margin_m,
                )
                sync_diagnostics = synchronizer.diagnostics
            finally:
                _destroy_actors(actors)

            if args.capture_mode == "policy" and reason in CoverageSelector.INFRASTRUCTURE_REASONS:
                infra_faults[reason] = infra_faults.get(reason, 0) + 1
                print(
                    f"INFRA_RETRY reason={reason} "
                    f"count={sum(infra_faults.values())}/{args.max_infra_retries}",
                    flush=True,
                )
                if sum(infra_faults.values()) > args.max_infra_retries:
                    raise RuntimeError("paper-still capture exceeded infrastructure retry limit")
                continue
            if candidate is None:
                raise RuntimeError(
                    f"no camera frame satisfied capture event {event_name!r}; "
                    "try staged mode or override the event target"
                )
            event_meta = candidate["metadata"]
            if float(event_meta["distance_to_workzone_m"]) > max_wz_distance:
                raise RuntimeError(
                    "best event frame was not in/near the work zone: "
                    f"distance={event_meta['distance_to_workzone_m']:.2f}m > "
                    f"limit={max_wz_distance:.2f}m"
                )

            for view, bgra in candidate["frames"].items():
                filename = f"training_{setting_slug}_origin{args.origin_index}_{view}.png"
                output = run_dir / filename
                _save_rgb_png(output, bgra, args.width, args.height)
                output_files[view] = filename
            accepted = {
                "origin_index": int(args.origin_index),
                "episode_steps": int(steps),
                "episode_return": float(episode_return),
                "terminal_world_frame": terminal_world_frame,
                "terminal_reason": reason,
                "outcome": (
                    _outcome_flags_v2(reason)
                    if args.capture_mode == "policy"
                    else None
                ),
                "event": {
                    "selection": event_name,
                    "target_xy_m": list(event_target),
                    "sensor_frame": int(candidate["sensor_frame"]),
                    **event_meta,
                },
                "sensor_sync": sync_diagnostics,
            }
    except BaseException:
        for filename in output_files.values():
            (run_dir / filename).unlink(missing_ok=True)
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

    if ego_identity is None:
        raise RuntimeError("paper ego identity was not recorded")
    ego_actor = {
        "blueprint": str(args.ego_blueprint),
        "requested_color_rgb": list(args.ego_color),
        "requested_color_attribute": ",".join(str(value) for value in args.ego_color),
        **ego_identity,
    }
    scenario_manifest = scenario_root(cfg.scenario_id) / "manifest.json"
    scenario_source = scenario_root(cfg.scenario_id) / "scenario.py"
    controller_source = scenario_root(cfg.scenario_id) / "jaywalker_controller.py"
    outputs = {
        view: {
            "file": filename,
            "size_bytes": (run_dir / filename).stat().st_size,
            "sha256": _sha256(run_dir / filename),
            "encoding": "PNG/RGB/lossless",
        }
        for view, filename in output_files.items()
    }
    manifest = {
        "dataset_split": "training",
        "purpose": "illustrative_paper_still_not_evaluation",
        "setting_id": cfg.setting_id,
        "scenario_id": cfg.scenario_id,
        "wz_id": cfg.wz_id,
        "layout_id": cfg.layout_id,
        "town": cfg.carla.town,
        "traffic_backend": cfg.traffic_backend,
        "capture_mode": args.capture_mode,
        "policy_name": policy_name if model_path is not None else None,
        "model": str(model_path) if model_path is not None else None,
        "model_sha256": _sha256(model_path) if model_path is not None else None,
        "rollout_seed": int(args.seed),
        "deterministic_policy": args.capture_mode == "policy",
        "ego_actor": ego_actor,
        "inputs": {
            "scenario_manifest": str(scenario_manifest.resolve()),
            "scenario_manifest_sha256": _sha256(scenario_manifest),
            "config": str(record.config_path.resolve()),
            "config_sha256": _sha256(record.config_path),
            "materialized_config_sha256": _json_sha256(materialized_data),
            "scenario_source": str(scenario_source.resolve()),
            "scenario_source_sha256": _sha256(scenario_source),
            "jaywalker_controller": str(controller_source.resolve()) if controller_source.is_file() else None,
            "jaywalker_controller_sha256": _sha256(controller_source) if controller_source.is_file() else None,
            "sumo_network": str(record.network_path.resolve()) if record.network_path else None,
            "sumo_network_sha256": _sha256(record.network_path) if record.network_path else None,
            "sumo_route": str(record.route_path.resolve()) if record.route_path else None,
            "sumo_route_sha256": _sha256(record.route_path) if record.route_path else None,
        },
        "capture": {
            "width": int(args.width),
            "height": int(args.height),
            "format": "PNG/RGB/lossless",
            "sim_dt_s": float(cfg.episode.sim_dt),
            "simultaneous_same_frame": bool(args.include_chase),
            "overview": {
                "kind": "fixed_world_camera_spectator_aligned",
                "eye": list(pose.eye),
                "target": list(pose.target),
                "fov_deg": float(pose.fov_deg),
                "source": pose.source,
            },
            "chase": (
                {
                    "attachment": "SpringArmGhost",
                    "relative_location_m": list(CHASE_LOCATION_M),
                    "relative_rotation_deg_pitch_yaw_roll": list(CHASE_ROTATION_DEG),
                    "fov_deg": CHASE_FOV_DEG,
                }
                if args.include_chase
                else None
            ),
        },
        "protocol": {
            "test_queues_or_results_touched": False,
            "capture_must_be_near_workzone": True,
            "max_workzone_distance_m": float(max_wz_distance),
            "infrastructure_failures_excluded_and_retried": args.capture_mode == "policy",
            "infrastructure_faults": dict(sorted(infra_faults.items())),
            "staged_capture_not_a_policy_result": args.capture_mode == "staged",
        },
        "episode": accepted,
        "outputs": outputs,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    manifest_path = run_dir / "paper_still_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"TRAINING PAPER STILL SAVED setting={cfg.setting_id} "
        f"event_frame={accepted['event']['sensor_frame']} dir={run_dir}",
        flush=True,
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_settings:
        for setting_id in PAPER_TRAINING_SETTINGS:
            cfg = load_scenario(setting_id)
            print(f"{setting_id}\t{cfg.carla.town}\t{cfg.traffic_backend}")
        return 0
    plan = _validate_plan(args)
    _print_plan(args, plan)
    if args.dry_run:
        print("Dry-run complete: no CARLA connection and no files written.", flush=True)
        return 0
    manifest = _record(args, *plan)
    print(f"manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
