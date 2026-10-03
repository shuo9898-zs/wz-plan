"""Record qualitative frozen-policy demos on the held-out Test split.

This recorder is deliberately separate from ``run_policy_test.py`` and the
paper evaluation queues.  It records one completed rollout for each requested
paper-facing scenario and origin, retaining success and every policy failure.
Only simulator-infrastructure terminal reasons are discarded and retried.

The fixed overview and rear SpringArmGhost RGB cameras run simultaneously in
the same rollout.  Their callbacks are paired by the exact CARLA sensor frame
id before encoding, at the scenario's native 10 Hz simulation rate.

The script never launches CARLA.  Do not run it on a CARLA/SUMO port currently
owned by an evaluation queue.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO

from baseline.PPO.runtime_v2 import configure_initial_config
from baseline.PPO.validate_allSwithoneExe import OrderedOriginSelector, OriginBinder
from config.scenario_selector import CoverageSelector
from Test.record_training_demo import (
    CHASE_FOV_DEG,
    CHASE_LOCATION_M,
    CHASE_ROTATION_DEG,
    CameraPose,
    _FramePairSynchronizer,
    _destroy_actors,
    _json_sha256,
    _require_carla,
    _safe_name,
    _spawn_camera_pair,
)
from Test.run_policy_test import (
    _OriginTrackingWrapper,
    _sha256,
    _suppress_windows_crash_dialogs,
)
from Test.test_case import TEST_SPECS, TestCase, load_test_case
from validation_debug.runner_v2 import _outcome_flags_v2, _validate_model_spaces_v2


SCENARIOS = ("s1", "s3", "s4", "s5")
DEFAULT_POLICY_NAME = "joint_seed027_p32"
DEFAULT_MODEL = (
    WORKSPACE_ROOT
    / "Aug24_ppo_OldOD_MultiSeed"
    / "runs_center_ego_old_od_seed27_r1"
    / "models"
    / "policy_update_032.zip"
)
DEFAULT_MODEL_SHA256 = (
    "4a333ff7b8160daabab981f1bbf9595af720a63588d0e0f7e5c32cfcf0083591"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Test" / "demos" / "test"
DEFAULT_SEED = 1007
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_BITRATE_KBPS = 8_000
ENV_MODULE = "env.gym_wrapper_center_v2"


# Keys are paper-facing Test identities.  S1 and S5 intentionally retain the
# established legacy runtime mapping documented in EXPECTED_RUNTIME_ID.
OVERVIEW_PRESETS: dict[str, CameraPose] = {
    "s1": CameraPose(
        eye=(-60.0, 30.0, 20.0),
        target=(-54.5, 51.0, 1.0),
        fov_deg=90.0,
        source="audited_test_preset_v2_clear_corridor",
    ),
    "s3": CameraPose(
        eye=(63.0, -78.0, 16.0),
        target=(20.0, -64.5, 1.0),
        fov_deg=90.0,
        source="audited_test_preset_v1",
    ),
    "s4": CameraPose(
        eye=(123.0, 29.0, 11.0),
        target=(108.0, 15.0, 1.0),
        fov_deg=90.0,
        source="audited_test_preset_v1",
    ),
    "s5": CameraPose(
        eye=(96.0, 176.0, 16.0),
        target=(80.0, 191.0, 1.0),
        fov_deg=90.0,
        source="audited_test_preset_v1",
    ),
}
EXPECTED_RUNTIME_ID = {
    "s1": "s5",
    "s3": "s3",
    "s4": "s4",
    "s5": "s1",
}
EXPECTED_TOWN = {
    "s1": "Town05",
    "s3": "Town10HD",
    "s4": "Town10HD",
    "s5": "Town02",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="s4",
        help="paper-facing held-out Test scenario",
    )
    parser.add_argument("--origin-index", type=int, default=0)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--carla-port",
        type=int,
        help="single-scenario override; otherwise use the audited Test port",
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        help="single-scenario override; otherwise use the audited Test port",
    )
    parser.add_argument(
        "--sumo-port",
        type=int,
        help="single-scenario override; otherwise use the audited Test port",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--bitrate-kbps", type=int, default=DEFAULT_BITRATE_KBPS)
    parser.add_argument("--max-file-mib", type=float, default=256.0)
    parser.add_argument("--max-infra-retries", type=int, default=10)
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="discard policy-failure rollouts and save only goal_reached",
    )
    parser.add_argument(
        "--max-policy-attempts",
        type=int,
        default=50,
        help="maximum completed rollouts when --require-success is enabled",
    )
    parser.add_argument(
        "--camera-pair-timeout",
        type=float,
        default=30.0,
        help="maximum wait when both GPU camera streams trail the rollout",
    )
    parser.add_argument(
        "--camera-flush-timeout",
        type=float,
        default=10.0,
        help="wait for delayed GPU callbacks after the terminal tick",
    )
    parser.add_argument(
        "--overview-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="override overview world location in metres (single scenario only)",
    )
    parser.add_argument(
        "--overview-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="override overview look-at point in metres (single scenario only)",
    )
    parser.add_argument(
        "--overview-fov",
        type=float,
        help="override overview horizontal FOV (single scenario only)",
    )
    parser.add_argument("--ffmpeg", help="ffmpeg executable; auto-detected")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate/print the plan without CARLA, SUMO, ffmpeg, or output",
    )
    return parser


def _scenario_ids(requested: str) -> tuple[str, ...]:
    return SCENARIOS if requested == "all" else (requested,)


def _camera_pose(args: argparse.Namespace, scenario_id: str) -> CameraPose:
    pose = OVERVIEW_PRESETS[scenario_id]
    if (
        args.overview_eye is None
        and args.overview_target is None
        and args.overview_fov is None
    ):
        return pose
    eye = tuple(args.overview_eye) if args.overview_eye is not None else pose.eye
    target = (
        tuple(args.overview_target)
        if args.overview_target is not None
        else pose.target
    )
    fov = float(args.overview_fov) if args.overview_fov is not None else pose.fov_deg
    if not all(math.isfinite(float(value)) for value in (*eye, *target, fov)):
        raise ValueError("overview camera values must all be finite")
    if not 5.0 <= fov <= 170.0:
        raise ValueError("--overview-fov must be between 5 and 170 degrees")
    if math.dist(eye, target) < 0.1:
        raise ValueError("overview eye and target must be different")
    return CameraPose(
        eye=eye,
        target=target,
        fov_deg=fov,
        source=f"cli_override_from_{pose.source}",
    )


def _validate_common(args: argparse.Namespace, scenario_ids: tuple[str, ...]) -> tuple[Path, str]:
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint does not exist: {model_path}")
    if (
        model_path == DEFAULT_MODEL.resolve()
        and _sha256(model_path) != DEFAULT_MODEL_SHA256
    ):
        raise RuntimeError(
            "default joint_seed027_p32 checkpoint hash no longer matches the "
            "frozen model used by the audited Test results"
        )
    policy_name = _safe_name(args.policy_name)
    if not 0 <= args.origin_index <= 2:
        raise ValueError("--origin-index must be 0, 1, or 2")
    if args.width < 320 or args.height < 180:
        raise ValueError("recording resolution is too small")
    if args.bitrate_kbps < 1_000:
        raise ValueError("--bitrate-kbps must be at least 1000")
    if (
        args.max_file_mib <= 0.0
        or args.max_infra_retries < 0
        or args.camera_pair_timeout <= 0.0
        or args.camera_flush_timeout <= 0.0
    ):
        raise ValueError("invalid output/retry limit")
    has_camera_override = any(
        value is not None
        for value in (args.overview_eye, args.overview_target, args.overview_fov)
    )
    has_port_override = any(
        value is not None for value in (args.carla_port, args.tm_port, args.sumo_port)
    )
    if len(scenario_ids) > 1 and has_camera_override:
        raise ValueError("overview overrides require one explicit --scenario")
    if len(scenario_ids) > 1 and has_port_override:
        raise ValueError("port overrides require one explicit --scenario")
    return model_path, policy_name


def _load_and_audit_case(scenario_id: str) -> TestCase:
    case = load_test_case(scenario_id)
    expected_runtime = EXPECTED_RUNTIME_ID[scenario_id]
    if case.spec.runtime_scenario_id != expected_runtime:
        raise RuntimeError(
            f"legacy Test mapping changed for {scenario_id}: "
            f"{case.spec.runtime_scenario_id} != {expected_runtime}"
        )
    if case.config.scenario_id != expected_runtime:
        raise RuntimeError(
            f"materialized runtime ID mismatch for {scenario_id}: "
            f"{case.config.scenario_id} != {expected_runtime}"
        )
    if case.spec.town != EXPECTED_TOWN[scenario_id]:
        raise RuntimeError(
            f"Test town mapping changed for {scenario_id}: {case.spec.town}"
        )
    if case.config.setting_id != f"test/{scenario_id}/wz1/a":
        raise RuntimeError(f"unexpected Test setting: {case.config.setting_id}")
    if len(case.config.origin.spawn_points) != 3:
        raise RuntimeError(f"{scenario_id} no longer exposes exactly three origins")
    if not math.isclose(
        float(case.config.episode.sim_dt) * float(case.config.episode.sync_hz),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(f"{scenario_id} sim_dt/sync_hz is inconsistent")
    if not float(case.config.episode.sync_hz).is_integer():
        raise RuntimeError(f"{scenario_id} native simulator rate is not integral")
    return case


def _ports(args: argparse.Namespace, case: TestCase) -> tuple[int, int, int]:
    return (
        int(args.carla_port or case.spec.carla_port),
        int(args.tm_port or case.spec.tm_port),
        int(args.sumo_port or case.spec.sumo_port),
    )


def _print_plan(
    args: argparse.Namespace,
    cases: tuple[TestCase, ...],
    model_path: Path,
    policy_name: str,
) -> None:
    print("HELD-OUT TEST QUALITATIVE DEMO PLAN", flush=True)
    print("  purpose: qualitative_demo_not_evaluation", flush=True)
    print(f"  policy: {policy_name} (deterministic)", flush=True)
    print(f"  model: {model_path}", flush=True)
    print(f"  model_sha256: {_sha256(model_path)}", flush=True)
    print(f"  canonical_origin_index: {args.origin_index}", flush=True)
    for case in cases:
        scenario_id = case.spec.scenario_id
        carla_port, tm_port, sumo_port = _ports(args, case)
        pose = _camera_pose(args, scenario_id)
        fps = int(round(case.config.episode.sync_hz))
        mapping = (
            "legacy mapping"
            if scenario_id != case.spec.runtime_scenario_id
            else "identity mapping"
        )
        print(
            f"  {scenario_id}: paper={scenario_id} -> "
            f"runtime={case.spec.runtime_scenario_id} ({mapping}) | "
            f"{case.spec.display_name} | {case.spec.town} | "
            f"setting={case.config.setting_id}",
            flush=True,
        )
        print(
            f"    ports carla/tm/sumo={carla_port}/{tm_port}/{sumo_port} | "
            f"capture={args.width}x{args.height}@{fps}Hz | "
            f"overview eye={pose.eye} target={pose.target} fov={pose.fov_deg}",
            flush=True,
        )
        print(
            f"    config_sha256={_sha256(case.config_path)} | "
            f"materialized_sha256={_json_sha256(case.materialized_data)}",
            flush=True,
        )
    outcome_rule = (
        "discard policy failures and save only goal_reached"
        if args.require_success
        else "retain every completed success/policy failure"
    )
    print(
        f"  outcome rule: {outcome_rule}; discard+retry infrastructure terminals",
        flush=True,
    )
    print("  evaluation result files and queue state: untouched", flush=True)


def _record_case(
    args: argparse.Namespace,
    case: TestCase,
    model_path: Path,
    policy_name: str,
    ffmpeg: str,
) -> Path:
    from Test.record_policy_demos import _Mp4Writer
    from validation_debug.visuals_v2 import ValidationWorkZonePropsV2

    scenario_id = case.spec.scenario_id
    cfg = case.config
    pose = _camera_pose(args, scenario_id)
    carla_port, tm_port, sumo_port = _ports(args, case)
    _require_carla(args.host, carla_port, case.spec.town, args.connect_timeout)
    configure_initial_config(
        cfg,
        carla_port=carla_port,
        tm_port=tm_port,
        sumo_port=sumo_port,
        no_rendering=False,
    )
    cfg.carla.host = str(args.host)

    all_origins = tuple(cfg.origin.spawn_points)
    selected_origin = all_origins[args.origin_index]
    selector = OrderedOriginSelector([cfg.setting_id], {cfg.setting_id: 1}, repeats=1)
    binder = OriginBinder(selector, {cfg.setting_id: (selected_origin,)})

    def episode_setup(setting_id: str) -> None:
        binder.bind(setting_id)

    # Import the centre-V2 environment first.  Its import chain pins the
    # repository's supported TraCI/SUMO path before any direct legacy-module
    # access.  S4 then patches the already-loaded legacy controller symbol.
    env_type = importlib.import_module(ENV_MODULE).CarlaSumoGymEnv
    original_jaywalker_controller: Any = None
    legacy_env_module: Any = None
    controller_source: Path | None = None
    if case.spec.runtime_scenario_id == "s4":
        controller_module = importlib.import_module(
            "Test.Scenarios.test.S4_Town10HD_Jaywalker.jaywalker_controller"
        )
        controller_source = Path(controller_module.__file__).resolve()
        legacy_env_module = importlib.import_module("env.carla_sumo_env")
        original_jaywalker_controller = legacy_env_module.JaywalkerController
        legacy_env_module.JaywalkerController = controller_module.JaywalkerController

    base_env: Any = None
    env: Any = None
    try:
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

        model = PPO.load(str(model_path), device=args.device)
        model.policy.set_training_mode(False)
        for parameter in model.policy.parameters():
            parameter.requires_grad_(False)
        _validate_model_spaces_v2(model, env)
    except BaseException:
        if env is not None:
            env.close()
        elif base_env is not None:
            base_env.close()
        if legacy_env_module is not None:
            legacy_env_module.JaywalkerController = original_jaywalker_controller
        raise

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = Path(args.output_root).expanduser().resolve() / (
        f"{scenario_id}_origin{args.origin_index}__{policy_name}__{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    fps = int(round(cfg.episode.sync_hz))
    infra_faults: dict[str, int] = {}
    policy_failures = 0
    partial_paths_created: set[Path] = set()
    first_reset = True
    accepted: dict[str, Any] | None = None
    try:
        try:
            while accepted is None:
                observation, _ = env.reset(seed=args.seed if first_reset else None)
                first_reset = False
                world = base_env.engine.world
                if world is None:
                    raise RuntimeError("CARLA world is unavailable after reset")
                props.start(world)
                attempt_number = 1 + sum(infra_faults.values()) + policy_failures
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
                        paired = synchronizer.collect()
                        last_sensor_frame = (
                            synchronizer.paired_frame_ids[-1]
                            if synchronizer.paired_frame_ids
                            else None
                        )
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
                        f"INFRA_RETRY scenario={scenario_id} attempt={attempt_number} "
                        f"reason={reason} count={sum(infra_faults.values())}/"
                        f"{args.max_infra_retries}",
                        flush=True,
                    )
                    if sum(infra_faults.values()) > args.max_infra_retries:
                        raise RuntimeError(
                            f"{scenario_id} qualitative demo exceeded infra retry limit"
                        )
                    continue

                if args.require_success and reason != "goal_reached":
                    policy_failures += 1
                    for path in partials.values():
                        path.unlink(missing_ok=True)
                    print(
                        f"POLICY_RETRY scenario={scenario_id} attempt={attempt_number} "
                        f"reason={reason} count={policy_failures}/"
                        f"{args.max_policy_attempts}",
                        flush=True,
                    )
                    if policy_failures >= args.max_policy_attempts:
                        raise RuntimeError(
                            f"{scenario_id} qualitative demo did not reach the goal "
                            f"within {args.max_policy_attempts} completed rollouts"
                        )
                    continue

                filenames = {
                    view: (
                        f"test_{scenario_id}_origin{args.origin_index}_"
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
                    "origin_spawn": {
                        "x_m": float(selected_origin.x),
                        "y_m": float(selected_origin.y),
                        "z_m": float(selected_origin.z),
                        "pitch_deg": float(selected_origin.pitch_deg),
                        "roll_deg": float(selected_origin.roll_deg),
                        "yaw_deg": float(selected_origin.yaw_deg),
                    },
                    "episode_steps": steps,
                    "episode_duration_s": round(
                        steps * float(cfg.episode.sim_dt), 3
                    ),
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
                                sync_diagnostics["paired_frame_count"] / fps, 3
                            ),
                        }
                        for view in ("overview", "chase")
                    },
                }
        except BaseException:
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
                try:
                    env.close()
                finally:
                    if legacy_env_module is not None:
                        legacy_env_module.JaywalkerController = (
                            original_jaywalker_controller
                        )
    except BaseException:
        # The environment has already been closed/restored above.  This outer
        # guard exists so no controller patch survives construction failures.
        if legacy_env_module is not None:
            legacy_env_module.JaywalkerController = original_jaywalker_controller
        raise

    manifest_path = run_dir / "demo_manifest.json"
    scenario_source = case.scenario_root / "scenario.py"
    manifest = {
        "dataset_split": "test",
        "purpose": "qualitative_demo_not_evaluation",
        "paper_scenario_id": scenario_id,
        "runtime_scenario_id": case.spec.runtime_scenario_id,
        "legacy_scenario_mapping_applied": (
            scenario_id != case.spec.runtime_scenario_id
        ),
        "display_name": case.spec.display_name,
        "setting_id": cfg.setting_id,
        "town": case.spec.town,
        "traffic_backend": cfg.traffic_backend,
        "manifest_status": case.status,
        "runtime_mode": "eval_with_frozen_policy",
        "policy_name": policy_name,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "rollout_seed": int(args.seed),
        "deterministic_policy": True,
        "ports": {
            "carla": carla_port,
            "traffic_manager": tm_port,
            "sumo": sumo_port if cfg.uses_sumo else None,
        },
        "inputs": {
            "test_manifest": str(case.manifest_path.resolve()),
            "test_manifest_sha256": _sha256(case.manifest_path),
            "test_config": str(case.config_path.resolve()),
            "test_config_sha256": _sha256(case.config_path),
            "materialized_config_sha256": _json_sha256(case.materialized_data),
            "scenario_source": str(scenario_source.resolve()),
            "scenario_source_sha256": _sha256(scenario_source),
            "jaywalker_controller": (
                str(controller_source) if controller_source is not None else None
            ),
            "jaywalker_controller_sha256": (
                _sha256(controller_source) if controller_source is not None else None
            ),
            "sumo_network": (
                str(case.network_path.resolve()) if case.network_path else None
            ),
            "sumo_network_sha256": (
                _sha256(case.network_path) if case.network_path else None
            ),
            "sumo_route": (
                str(case.route_path.resolve()) if case.route_path else None
            ),
            "sumo_route_sha256": (
                _sha256(case.route_path) if case.route_path else None
            ),
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
            "frame_pairing": "exact_carla_sensor_frame_id",
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
            "completed_rollouts_requested": 1,
            "require_success": bool(args.require_success),
            "retain_completed_success_or_policy_failure": not args.require_success,
            "policy_failures_excluded": int(policy_failures),
            "infrastructure_failures_excluded_and_retried": True,
            "infrastructure_faults": dict(sorted(infra_faults.items())),
            "writes_evaluation_results": False,
            "mutates_evaluation_queue": False,
        },
        "episode": accepted,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"TEST QUALITATIVE DEMO SAVED paper={scenario_id} "
        f"runtime={case.spec.runtime_scenario_id} origin={args.origin_index} "
        f"reason={accepted['terminal_reason']} dir={run_dir}",
        flush=True,
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    _suppress_windows_crash_dialogs()
    args = _parser().parse_args(argv)
    scenario_ids = _scenario_ids(args.scenario)
    model_path, policy_name = _validate_common(args, scenario_ids)
    cases = tuple(_load_and_audit_case(scenario_id) for scenario_id in scenario_ids)
    _print_plan(args, cases, model_path, policy_name)
    if args.dry_run:
        print(
            "Dry-run complete: no CARLA/SUMO connection and no files written.",
            flush=True,
        )
        return 0

    from Test.record_policy_demos import _check_ffmpeg, _resolve_ffmpeg

    ffmpeg = _resolve_ffmpeg(args.ffmpeg)
    _check_ffmpeg(ffmpeg)
    manifests = [
        _record_case(args, case, model_path, policy_name, ffmpeg)
        for case in cases
    ]
    for manifest in manifests:
        print(f"manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
