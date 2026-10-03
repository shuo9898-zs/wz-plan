"""Read-only migration preflight for the PPO V2 training stack.

The default check does not connect to a simulator.  It verifies the exact
software versions used by the current run, the pinned SUMO executable and
TraCI client, CUDA availability when requested, and the complete SB3 policy
contract (observation/action shapes, encoder outputs and parameter count).

Use ``--check-servers`` after starting CARLA to additionally verify that the
three RPC ports expose Town02, Town05 and Town10HD respectively.  No world
settings, actors or maps are changed by this tool.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from baseline.PPO.training_config_v2 import DEFAULT_THREE_SERVER_TRAINING_V2


EXPECTED_PYTHON = (3, 8)
EXPECTED_PACKAGES = {
    "numpy": "1.24.3",
    "gymnasium": "0.29.1",
    "stable-baselines3": "2.3.2",
    "shapely": "2.0.5",
    "torch": "2.4.1",
    "carla": "0.9.15",
    "eclipse-sumo": "1.27.1",
}
EXPECTED_OBSERVATION_DIM = 1_311
EXPECTED_ACTION_DIM = 2
EXPECTED_ENCODER_PARAMETERS = 419_584
EXPECTED_POLICY_PARAMETERS = 1_103_109
EXPECTED_SETTINGS = 54
_TRAINING_DEFAULTS = DEFAULT_THREE_SERVER_TRAINING_V2
EXPECTED_TOWNS = tuple(worker.town for worker in _TRAINING_DEFAULTS.town_workers)
EXPECTED_CARLA_PORTS = tuple(
    worker.carla_port for worker in _TRAINING_DEFAULTS.town_workers
)
EXPECTED_TRAINING_DEFAULTS = {
    "total_updates": _TRAINING_DEFAULTS.total_updates,
    "town02_steps": _TRAINING_DEFAULTS.town_step_quotas["Town02"],
    "town05_steps": _TRAINING_DEFAULTS.town_step_quotas["Town05"],
    "town10hd_steps": _TRAINING_DEFAULTS.town_step_quotas["Town10HD"],
    "batch_size": _TRAINING_DEFAULTS.ppo.batch_size,
    "ppo_epochs": _TRAINING_DEFAULTS.ppo.max_epochs_per_update,
    "target_kl": _TRAINING_DEFAULTS.ppo.target_kl,
}
EXPECTED_SB3_DEFAULTS = {
    "learning_rate": _TRAINING_DEFAULTS.ppo.learning_rate,
    "clip_range": _TRAINING_DEFAULTS.ppo.clip_range,
    "ent_coef": _TRAINING_DEFAULTS.ppo.entropy_coefficient,
    "vf_coef": _TRAINING_DEFAULTS.ppo.value_function_coefficient,
    "max_grad_norm": _TRAINING_DEFAULTS.ppo.max_gradient_norm,
    "normalize_advantage": _TRAINING_DEFAULTS.ppo.normalize_advantage,
}


@dataclass(frozen=True)
class CheckResultV2:
    name: str
    passed: bool
    detail: str


def _result(name: str, condition: bool, detail: str) -> CheckResultV2:
    return CheckResultV2(name=name, passed=bool(condition), detail=str(detail))


def _base_version(version: str) -> str:
    return str(version).split("+", 1)[0]


def check_runtime_versions_v2(*, require_cuda: bool) -> list[CheckResultV2]:
    results = [
        _result(
            "python",
            sys.version_info[:2] == EXPECTED_PYTHON,
            f"found {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; "
            f"expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}.x",
        )
    ]
    for distribution, expected in EXPECTED_PACKAGES.items():
        try:
            found = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            results.append(_result(f"package:{distribution}", False, "not installed"))
            continue
        comparable = _base_version(found) if distribution == "torch" else found
        results.append(
            _result(
                f"package:{distribution}",
                comparable == expected,
                f"found {found}; expected {expected}",
            )
        )

    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device = torch.cuda.get_device_name(0)
            cuda_detail = f"available: {device}; torch CUDA {torch.version.cuda}"
        else:
            cuda_detail = "not available"
        results.append(_result("cuda", cuda_available or not require_cuda, cuda_detail))
    except Exception as exc:
        results.append(_result("cuda", False, f"torch CUDA check failed: {exc}"))
    return results


def check_sumo_runtime_v2() -> list[CheckResultV2]:
    try:
        runtime_module = importlib.import_module("env.sumo_runtime_v2")
        runtime_module.prefer_sumo_python_tools_v2()
        runtime = runtime_module.SUMO_RUNTIME_V2
        completed = subprocess.run(
            [str(runtime.sumo_binary), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15.0,
            check=False,
        )
        first_line = completed.stdout.splitlines()[0] if completed.stdout else "no output"
        binary_ok = completed.returncode == 0 and "1.27.1" in completed.stdout

        traci = importlib.import_module("traci")
        traci_path = Path(traci.__file__).resolve()
        tools_path = runtime.tools_dir.resolve()
        try:
            traci_path.relative_to(tools_path)
            traci_is_pinned = True
        except ValueError:
            traci_is_pinned = False
        return [
            _result("sumo:binary", binary_ok, f"{first_line} at {runtime.sumo_binary}"),
            _result(
                "sumo:traci",
                traci_is_pinned,
                f"{traci_path}; expected beneath {tools_path}",
            ),
        ]
    except Exception as exc:
        return [_result("sumo:runtime", False, str(exc))]


def check_policy_contract_v2(*, device: str) -> list[CheckResultV2]:
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv

    from baseline.PPO.encoder_v2 import (
        ENCODER_FEATURES_DIM_V2,
        CrossAttentionEncoderV2,
    )
    from baseline.PPO.rollout_plan import RolloutPlan
    from baseline.PPO.runtime_v2 import (
        V2_DEFAULT_PPO_EPOCHS,
        V2_POLICY_TRAINABLE_PARAMETERS,
        build_or_load_ppo,
    )
    from baseline.PPO.train_three_servers_v2 import (
        DEFAULT_TOWN_STEPS,
        _SpaceOnlyEnv,
        build_parser,
    )
    from env.observation_encoder_v2 import (
        DEFAULT_OBSERVATION_DIM_V2,
        DEFAULT_OBSERVATION_SPEC_V2,
    )

    results: list[CheckResultV2] = []
    observation_space = gym.spaces.Box(
        -1.0,
        1.0,
        shape=(DEFAULT_OBSERVATION_DIM_V2,),
        dtype=np.float32,
    )
    extractor = CrossAttentionEncoderV2(observation_space).eval()
    with torch.no_grad():
        encoded = extractor(torch.zeros(2, DEFAULT_OBSERVATION_DIM_V2))
    encoder_parameters = sum(parameter.numel() for parameter in extractor.parameters())
    results.extend(
        (
            _result(
                "contract:observation",
                DEFAULT_OBSERVATION_DIM_V2 == EXPECTED_OBSERVATION_DIM,
                f"{DEFAULT_OBSERVATION_DIM_V2}D",
            ),
            _result(
                "contract:observation-shape",
                DEFAULT_OBSERVATION_SPEC_V2.observation_dim == EXPECTED_OBSERVATION_DIM,
                f"spec={DEFAULT_OBSERVATION_SPEC_V2.observation_dim}D",
            ),
            _result(
                "contract:encoder-forward",
                tuple(encoded.shape) == (2, ENCODER_FEATURES_DIM_V2)
                and bool(torch.isfinite(encoded).all()),
                f"shape={tuple(encoded.shape)}, finite={bool(torch.isfinite(encoded).all())}",
            ),
            _result(
                "contract:encoder-parameters",
                encoder_parameters == EXPECTED_ENCODER_PARAMETERS,
                str(encoder_parameters),
            ),
        )
    )

    env = DummyVecEnv([_SpaceOnlyEnv])
    plan = RolloutPlan(
        setting_count=1,
        repeats=1,
        n_envs=1,
        group_step_budgets=(8,),
        n_steps_per_env=8,
        buffer_size_steps=8,
    )
    try:
        model = build_or_load_ppo(
            env,
            plan,
            model_path=None,
            ppo_epochs=V2_DEFAULT_PPO_EPOCHS,
            seed=7,
            device=device,
            batch_size=4,
            target_kl=None,
        )
        policy_parameters = sum(
            parameter.numel() for parameter in model.policy.parameters()
        )
        policy_device = str(next(model.policy.parameters()).device)
        sb3_defaults = {
            "learning_rate": float(model.learning_rate),
            "clip_range": float(model.clip_range(1.0)),
            "ent_coef": float(model.ent_coef),
            "vf_coef": float(model.vf_coef),
            "max_grad_norm": float(model.max_grad_norm),
            "normalize_advantage": bool(model.normalize_advantage),
        }
        results.extend(
            (
                _result(
                    "contract:action",
                    tuple(env.action_space.shape) == (EXPECTED_ACTION_DIM,),
                    f"shape={env.action_space.shape}",
                ),
                _result(
                    "contract:policy-parameters",
                    policy_parameters == EXPECTED_POLICY_PARAMETERS
                    == V2_POLICY_TRAINABLE_PARAMETERS,
                    str(policy_parameters),
                ),
                _result(
                    "contract:independent-encoders",
                    not model.policy.share_features_extractor
                    and model.policy.pi_features_extractor
                    is not model.policy.vf_features_extractor,
                    "actor and critic feature extractors are separate",
                ),
                _result("contract:policy-device", True, policy_device),
                _result(
                    "contract:sb3-defaults",
                    sb3_defaults == EXPECTED_SB3_DEFAULTS,
                    json.dumps(sb3_defaults, sort_keys=True),
                ),
            )
        )
    finally:
        env.close()

    defaults = build_parser().parse_args([])
    found_defaults = {
        "total_updates": defaults.total_updates,
        "town02_steps": defaults.town02_steps,
        "town05_steps": defaults.town05_steps,
        "town10hd_steps": defaults.town10hd_steps,
        "batch_size": defaults.batch_size,
        "ppo_epochs": defaults.ppo_epochs,
        "target_kl": defaults.target_kl,
    }
    results.append(
        _result(
            "contract:training-defaults",
            found_defaults == EXPECTED_TRAINING_DEFAULTS
            and DEFAULT_TOWN_STEPS["Town05"] == 43_000,
            json.dumps(found_defaults, sort_keys=True),
        )
    )
    return results


def check_scenario_catalog_v2() -> list[CheckResultV2]:
    try:
        from config.scenario_catalog import list_setting_ids

        settings: list[str] = []
        for scenario in ("s1", "s2", "s3", "s4", "s5", "s6"):
            settings.extend(list_setting_ids(scenario, runnable_only=True))
        unique = len(set(settings))
        return [
            _result(
                "contract:scenario-catalog",
                len(settings) == unique == EXPECTED_SETTINGS,
                f"{len(settings)} runnable, {unique} unique",
            )
        ]
    except Exception as exc:
        return [_result("contract:scenario-catalog", False, str(exc))]


def check_carla_servers_v2(
    *,
    host: str,
    ports: Sequence[int],
    timeout_s: float,
) -> list[CheckResultV2]:
    try:
        carla = importlib.import_module("carla")
    except Exception as exc:
        return [_result("carla:python-api", False, str(exc))]

    results: list[CheckResultV2] = []
    for port, expected_town in zip(ports, EXPECTED_TOWNS):
        try:
            client = carla.Client(host, int(port))
            client.set_timeout(float(timeout_s))
            server_version = client.get_server_version()
            map_name = str(client.get_world().get_map().name).split("/")[-1]
            results.append(
                _result(
                    f"carla:{port}",
                    map_name == expected_town,
                    f"server={server_version}, map={map_name}, expected={expected_town}",
                )
            )
        except Exception as exc:
            results.append(_result(f"carla:{port}", False, str(exc)))
    return results


def run_preflight_v2(
    *,
    require_cuda: bool,
    policy_device: str,
    check_servers: bool,
    host: str,
    ports: Sequence[int],
    timeout_s: float,
) -> list[CheckResultV2]:
    results: list[CheckResultV2] = []
    results.extend(check_runtime_versions_v2(require_cuda=require_cuda))
    results.extend(check_sumo_runtime_v2())
    results.extend(check_scenario_catalog_v2())
    try:
        results.extend(check_policy_contract_v2(device=policy_device))
    except Exception as exc:
        results.append(_result("contract:policy", False, str(exc)))
    if check_servers:
        results.extend(
            check_carla_servers_v2(host=host, ports=ports, timeout_s=timeout_s)
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--policy-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check-servers", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--carla-ports",
        type=int,
        nargs=3,
        default=list(EXPECTED_CARLA_PORTS),
        metavar=("TOWN02", "TOWN05", "TOWN10HD"),
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _print_results(results: Iterable[CheckResultV2], *, as_json: bool) -> None:
    items = list(results)
    if as_json:
        print(json.dumps([asdict(item) for item in items], indent=2, sort_keys=True))
        return
    for item in items:
        marker = "PASS" if item.passed else "FAIL"
        print(f"[{marker}] {item.name}: {item.detail}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise SystemExit("--timeout must be finite and positive")
    results = run_preflight_v2(
        require_cuda=bool(args.require_cuda),
        policy_device=str(args.policy_device),
        check_servers=bool(args.check_servers),
        host=str(args.host),
        ports=tuple(int(port) for port in args.carla_ports),
        timeout_s=float(args.timeout),
    )
    _print_results(results, as_json=bool(args.as_json))
    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
