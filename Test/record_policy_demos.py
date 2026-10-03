"""Record two compact spectator-view success demos for each held-out test case.

The first clip uses a fixed elevated work-zone overview.  The second uses an
elevated chase camera attached to the ego vehicle.  Failed attempts are encoded
to a temporary file and deleted; only successful episodes are retained.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from baseline.PPO.runtime_v2 import configure_initial_config
from baseline.PPO.validate_allSwithoneExe import OrderedOriginSelector, OriginBinder
from config.scenario_selector import CoverageSelector
from Test.run_policy_test import (
    DEFAULT_TEST_SEED,
    TEST_ENV_MODULE,
    _OriginTrackingWrapper,
    _sha256,
)
from Test.test_case import PROJECT_ROOT, TEST_SPECS, TestCase, load_test_case
from validation_debug.runner_v2 import _outcome_flags_v2, _validate_model_spaces_v2
from validation_debug.visuals_v2 import ValidationWorkZonePropsV2


@dataclass(frozen=True)
class DemoPolicy:
    label: str
    model: Path
    validation_success: str


MULTISEED_ROOT = PROJECT_ROOT.parent / "Aug24_ppo_OldOD_MultiSeed"
QUEUE_ROOT = MULTISEED_ROOT / "Aug29_S6_S1to5_Queue" / "runs"
DEMO_POLICIES: dict[str, DemoPolicy] = {
    "s1": DemoPolicy(
        "s1to5_seed27_p17",
        QUEUE_ROOT / "s1to5_seed027_r1/models/policy_update_017.zip",
        "21/30 (70.0%)",
    ),
    "s2": DemoPolicy(
        "s2_specialist_seed7_p31",
        QUEUE_ROOT / "s2_specialist_seed007_r1/models/policy_update_031.zip",
        "10/30 (33.3%)",
    ),
    "s3": DemoPolicy(
        "s1to5_seed7_p13",
        QUEUE_ROOT / "s1to5_seed007_r1/models/policy_update_013.zip",
        "21/30 (70.0%)",
    ),
    "s4": DemoPolicy(
        "s1to5_seed27_p17",
        QUEUE_ROOT / "s1to5_seed027_r1/models/policy_update_017.zip",
        "20/30 (66.7%)",
    ),
    "s5": DemoPolicy(
        "s1to5_seed27_p17",
        QUEUE_ROOT / "s1to5_seed027_r1/models/policy_update_017.zip",
        "17/30 (56.7%)",
    ),
    "s6": DemoPolicy(
        "s6_specialist_seed7_p32",
        QUEUE_ROOT / "s6_specialist_seed007_r1/models/policy_update_032.zip",
        "30/30 (100.0%)",
    ),
}

# World-space camera and look-at points, selected to keep the complete hazard
# region visible.  The second clip for every scenario uses the generic chase.
OVERVIEW_CAMERA: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "s1": ((-70.0, 34.0, 15.0), (-54.5, 51.0, 1.0)),
    "s2": ((68.0, 64.0, 27.0), (86.0, 78.0, 1.0)),
    "s3": ((63.0, -78.0, 16.0), (20.0, -64.5, 1.0)),
    "s4": ((123.0, 29.0, 11.0), (108.0, 15.0, 1.0)),
    "s5": ((96.0, 176.0, 16.0), (80.0, 191.0, 1.0)),
    "s6": ((104.0, 88.0, 18.0), (101.0, 107.0, 1.0)),
}
VIEW_FOR_SLOT = ("overview", "chase")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", default="all", choices=("all", *tuple(TEST_SPECS))
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "Test" / "demos"))
    parser.add_argument("--seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2080)
    parser.add_argument("--tm-port", type=int, default=8080)
    parser.add_argument("--sumo-port", type=int, default=8893)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--bitrate-kbps", type=int, default=1200)
    parser.add_argument("--max-file-mib", type=float, default=12.0)
    parser.add_argument("--max-valid-attempts", type=int, default=20)
    parser.add_argument("--max-infra-retries", type=int, default=10)
    parser.add_argument("--ffmpeg", help="ffmpeg executable; auto-detected by default")
    parser.add_argument(
        "--allow-shared-training-port",
        action="store_true",
        help="Explicitly permit 2000/2020/2030/2040 (unsafe during training)",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Allow the current test manifests while their live check is pending",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cases/checkpoints/settings without CARLA or video encoding",
    )
    return parser


def _resolve_ffmpeg(explicit: str | None) -> str:
    if explicit:
        found = shutil.which(explicit) or (explicit if Path(explicit).is_file() else None)
        if found:
            return str(found)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required for compact MP4 output. Install the small "
            "'imageio-ffmpeg' package in ppo_carla, or pass --ffmpeg PATH."
        ) from exc


def _check_ffmpeg(executable: str) -> None:
    result = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode or "libx264" not in result.stdout:
        raise RuntimeError("ffmpeg must provide the libx264 encoder")


def _require_carla_server(host: str, port: int, timeout_s: float) -> None:
    try:
        import carla

        client = carla.Client(host, int(port))
        client.set_timeout(float(timeout_s))
        client.get_world()
    except Exception as exc:
        raise RuntimeError(f"CARLA is not ready at {host}:{port}: {exc}") from exc


def _look_at_transform(carla: Any, eye: tuple[float, float, float], target: tuple[float, float, float]):
    dx, dy, dz = (target[index] - eye[index] for index in range(3))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return carla.Transform(
        carla.Location(x=eye[0], y=eye[1], z=eye[2]),
        carla.Rotation(pitch=pitch, yaw=yaw),
    )


class _FrameSink:
    def __init__(self) -> None:
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=3)
        self.last_frame: bytes | None = None

    def __call__(self, image: Any) -> None:
        frame = bytes(image.raw_data)
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(frame)

    def next(self, timeout_s: float = 5.0) -> bytes | None:
        try:
            frame = self.frames.get(timeout=timeout_s)
        except queue.Empty:
            return self.last_frame
        while True:
            try:
                frame = self.frames.get_nowait()
            except queue.Empty:
                self.last_frame = frame
                return frame


class _Mp4Writer:
    def __init__(
        self,
        ffmpeg: str,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate_kbps: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgra",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{int(bitrate_kbps * 1.25)}k",
            "-bufsize", f"{bitrate_kbps * 2}k", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]
        self.path = path
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, bgra: bytes) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(bgra)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        returncode = self.process.wait(timeout=30)
        if returncode:
            raise RuntimeError(
                f"ffmpeg failed ({returncode}): {stderr.decode(errors='replace').strip()}"
            )


def _spawn_camera(
    case: TestCase,
    base_env: gym.Env,
    *,
    view: str,
    width: int,
    height: int,
    fps: int,
) -> tuple[Any, _FrameSink]:
    import carla

    world = base_env.engine.world
    ego = base_env.engine.ego
    if world is None or ego is None:
        raise RuntimeError("CARLA world/ego is unavailable after reset")
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", "90")
    # CARLA advances at 10 Hz. Capture each tick, then downsample to 5 Hz in
    # Python if requested so the synchronous simulator never waits for a frame.
    blueprint.set_attribute("sensor_tick", "0.100000")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "episode_sensor")
    sink = _FrameSink()
    if view == "overview":
        transform = _look_at_transform(carla, *OVERVIEW_CAMERA[case.spec.scenario_id])
        camera = world.spawn_actor(blueprint, transform)
    else:
        transform = carla.Transform(
            carla.Location(x=-10.0, z=6.5),
            carla.Rotation(pitch=-1.0),
        )
        camera = world.spawn_actor(
            blueprint,
            transform,
            attach_to=ego,
            attachment_type=carla.AttachmentType.SpringArmGhost,
        )
    camera.listen(sink)
    world.get_spectator().set_transform(camera.get_transform())
    return camera, sink


def _destroy_camera(camera: Any | None) -> None:
    if camera is None:
        return
    try:
        camera.stop()
    except Exception:
        pass
    try:
        if camera.is_alive:
            camera.destroy()
    except Exception:
        pass


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _record_scenario(args: argparse.Namespace, scenario_id: str, ffmpeg: str) -> None:
    case = load_test_case(scenario_id, allow_blocked=args.allow_blocked)
    policy = DEMO_POLICIES[scenario_id]
    cfg = case.config
    configure_initial_config(
        cfg,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        sumo_port=args.sumo_port,
        no_rendering=False,
    )
    cfg.carla.host = str(args.host)
    _require_carla_server(args.host, args.carla_port, args.connect_timeout)

    import importlib

    env_type = importlib.import_module(TEST_ENV_MODULE).CarlaSumoGymEnv
    original_jaywalker_controller: Any = None
    legacy_env_module: Any = None
    if case.spec.runtime_scenario_id == "s4":
        controller_module = importlib.import_module(
            "Test.Scenarios.test.S4_Town10HD_Jaywalker.jaywalker_controller"
        )
        legacy_env_module = importlib.import_module("env.carla_sumo_env")
        original_jaywalker_controller = legacy_env_module.JaywalkerController
        legacy_env_module.JaywalkerController = controller_module.JaywalkerController

    origins = tuple(cfg.origin.spawn_points)
    selector = OrderedOriginSelector(
        [cfg.setting_id], {cfg.setting_id: len(origins)}, repeats=1
    )
    binder = OriginBinder(selector, {cfg.setting_id: origins})

    def episode_setup(setting_id: str) -> None:
        binder.bind(setting_id)

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
    props = ValidationWorkZonePropsV2(case, enabled=True, cones_only=False)
    model = PPO.load(str(policy.model), device=args.device)
    model.policy.set_training_mode(False)
    _validate_model_spaces_v2(model, env)

    scenario_dir = Path(args.output_root) / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = scenario_dir / "demo_manifest.json"
    manifest: dict[str, Any] = {
        "scenario": scenario_id,
        "runtime_scenario": case.spec.runtime_scenario_id,
        "display_name": case.spec.display_name,
        "town": case.spec.town,
        "test_manifest_status": case.status,
        "policy": policy.label,
        "model": str(policy.model.resolve()),
        "model_sha256": _sha256(policy.model),
        "validation_success": policy.validation_success,
        "video": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "bitrate_kbps": args.bitrate_kbps,
            "codec": "H.264/yuv420p",
        },
        "clips": [],
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("model_sha256") == manifest["model_sha256"]:
            manifest["clips"] = [
                clip for clip in previous.get("clips", [])
                if (scenario_dir / str(clip.get("file", ""))).is_file()
            ]

    existing_slots = {int(clip["slot"]) for clip in manifest["clips"]}
    valid_attempts = 0
    infra_retries = 0
    first_reset = True
    try:
        for slot, view in enumerate(VIEW_FOR_SLOT, start=1):
            if slot in existing_slots:
                print(f"[DEMO {scenario_id}] slot={slot} already exists; skip", flush=True)
                continue
            saved = False
            while valid_attempts < args.max_valid_attempts:
                observation, _ = env.reset(seed=args.seed if first_reset else None)
                first_reset = False
                origin_index = int(env.active_origin_index)
                attempt_id = valid_attempts + infra_retries + 1
                partial = scenario_dir / f"._attempt_{attempt_id:03d}_{view}.mp4"
                partial.unlink(missing_ok=True)
                camera = None
                writer = None
                episode_steps = 0
                episode_return = 0.0
                reason = "running"
                try:
                    world = base_env.engine.world
                    if world is None:
                        raise RuntimeError("CARLA world is unavailable after reset")
                    props.start(world)
                    camera, sink = _spawn_camera(
                        case, base_env, view=view,
                        width=args.width, height=args.height, fps=args.fps,
                    )
                    writer = _Mp4Writer(
                        ffmpeg, partial,
                        width=args.width, height=args.height, fps=args.fps,
                        bitrate_kbps=args.bitrate_kbps,
                    )
                    done = False
                    while not done:
                        action, _ = model.predict(
                            observation, deterministic=not args.stochastic
                        )
                        observation, reward, terminated, truncated, info = env.step(
                            np.asarray(action, dtype=np.float32).reshape(-1)
                        )
                        episode_steps += 1
                        episode_return += float(reward)
                        done = bool(terminated or truncated)
                        reason = str(info.get("reason", "running"))
                        frame = sink.next()
                        if frame is not None and episode_steps % (10 // args.fps) == 0:
                            writer.write(frame)
                        world = base_env.engine.world
                        if world is not None and camera.is_alive:
                            world.get_spectator().set_transform(camera.get_transform())
                finally:
                    _destroy_camera(camera)
                    if writer is not None:
                        writer.close()

                if reason in CoverageSelector.INFRASTRUCTURE_REASONS:
                    infra_retries += 1
                    partial.unlink(missing_ok=True)
                    if infra_retries > args.max_infra_retries:
                        raise RuntimeError(
                            f"{scenario_id} exceeded {args.max_infra_retries} infrastructure retries"
                        )
                    continue

                valid_attempts += 1
                success = bool(_outcome_flags_v2(reason)["success"])
                print(
                    f"[DEMO {scenario_id}] attempt={valid_attempts}/"
                    f"{args.max_valid_attempts} origin={origin_index} view={view} "
                    f"steps={episode_steps} reason={reason}",
                    flush=True,
                )
                if not success:
                    partial.unlink(missing_ok=True)
                    continue

                final_name = (
                    f"{scenario_id}_demo{slot:02d}_{view}_origin{origin_index}_"
                    f"{policy.label}.mp4"
                )
                final_path = scenario_dir / final_name
                final_path.unlink(missing_ok=True)
                partial.replace(final_path)
                max_bytes = int(args.max_file_mib * 1024 * 1024)
                if final_path.stat().st_size > max_bytes:
                    actual_mib = final_path.stat().st_size / (1024 * 1024)
                    final_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"{scenario_id} clip is {actual_mib:.1f} MiB, above "
                        f"the {args.max_file_mib:.1f} MiB cap"
                    )
                manifest["clips"].append({
                    "slot": slot,
                    "view": view,
                    "file": final_name,
                    "origin_index": origin_index,
                    "episode_steps": episode_steps,
                    "duration_s": round(episode_steps * 0.1, 3),
                    "episode_return": episode_return,
                    "terminal_reason": reason,
                    "size_bytes": final_path.stat().st_size,
                    "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                })
                _write_manifest(manifest_path, manifest)
                saved = True
                print(f"[DEMO {scenario_id}] saved {final_path}", flush=True)
                break
            if not saved:
                raise RuntimeError(
                    f"{scenario_id} did not produce two successes within "
                    f"{args.max_valid_attempts} valid attempts"
                )
    finally:
        try:
            props.close()
        finally:
            try:
                env.close()
            finally:
                if legacy_env_module is not None:
                    legacy_env_module.JaywalkerController = original_jaywalker_controller


def _validate_plan(args: argparse.Namespace, scenarios: tuple[str, ...]) -> None:
    if args.width < 320 or args.height < 180:
        raise ValueError("resolution is too small")
    if args.fps not in (5, 10):
        raise ValueError("--fps must be 5 or 10 for the 10 Hz simulator")
    if args.bitrate_kbps < 400:
        raise ValueError("--bitrate-kbps must be at least 400")
    if args.max_file_mib < 2.0:
        raise ValueError("--max-file-mib must be at least 2")
    if args.max_valid_attempts < 2 or args.max_infra_retries < 0:
        raise ValueError("invalid attempt limits")
    if (
        args.carla_port in {2000, 2020, 2030, 2040}
        and not args.allow_shared_training_port
    ):
        raise ValueError(
            f"CARLA port {args.carla_port} is reserved for training; use the "
            "dedicated demo default 2080"
        )
    for scenario_id in scenarios:
        case = load_test_case(scenario_id, allow_blocked=args.allow_blocked)
        policy = DEMO_POLICIES[scenario_id]
        if not policy.model.is_file():
            raise FileNotFoundError(policy.model)
        print(
            f"{scenario_id}: {case.spec.display_name} | {case.spec.town} | "
            f"{policy.label} | val {policy.validation_success}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scenarios = tuple(TEST_SPECS) if args.scenario == "all" else (args.scenario,)
    _validate_plan(args, scenarios)
    if args.dry_run:
        print("Dry-run complete: no CARLA connection and no files written.", flush=True)
        return 0
    ffmpeg = _resolve_ffmpeg(args.ffmpeg)
    _check_ffmpeg(ffmpeg)
    for scenario_id in scenarios:
        _record_scenario(args, scenario_id, ffmpeg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
