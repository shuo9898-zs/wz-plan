"""Run the existing three-server PPO pipeline with the plain Transformer."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import baseline.PPO.train_three_servers_v2 as _ppo_v2
from baseline.PPO.global_rollout_checkpoint import file_digest
from baseline.PPO.training_logging import TrainingTelemetry
from baseline.Transformer_PPO.encoder import (
    DEFAULT_TRANSFORMER_NETWORK_SPEC,
    TRANSFORMER_ENCODER_CONTRACT,
    TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS,
    TRANSFORMER_EXPECTED_POLICY_PARAMETERS,
)
from baseline.Transformer_PPO.runtime import build_or_load_ppo


SCHEMA_VERSION = 1
THREE_SERVER_CONTRACT = "plain_transformer_ppo_three_servers_no_kl_v1"
DEFAULT_RUN_ROOT = Path("runs_transformer_ppo")

_CONFIGURED = False
_ORIGINAL_CHECKPOINT_PLAN = _ppo_v2._checkpoint_plan_payload
_ORIGINAL_TRAINING_PLAN = _ppo_v2._training_plan
_ORIGINAL_BANNER = TrainingTelemetry.banner


def build_parser():
    parser = _ppo_v2.build_parser()
    parser.description = __doc__
    parser.set_defaults(run_root=DEFAULT_RUN_ROOT, target_kl=None)
    return parser


def _worker_command_transformer(spec_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "baseline.Transformer_PPO.train_three_servers",
        "--worker-spec",
        str(spec_path),
    ]


def _checkpoint_plan_payload_transformer(*args, **kwargs) -> Dict[str, object]:
    payload = dict(_ORIGINAL_CHECKPOINT_PLAN(*args, **kwargs))
    for stale_key in (
        "encoder_contract",
        "encoder_token_dim",
        "encoder_branch_hidden_dim",
        "encoder_features_dim",
        "encoder_attention_heads",
    ):
        payload.pop(stale_key, None)
    payload.update(
        encoder=DEFAULT_TRANSFORMER_NETWORK_SPEC.fingerprint_payload(),
        encoder_contract=TRANSFORMER_ENCODER_CONTRACT,
        encoder_features_dim=DEFAULT_TRANSFORMER_NETWORK_SPEC.d_model,
        encoder_trainable_parameters_per_extractor=(
            TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS
        ),
        policy_trainable_parameters=TRANSFORMER_EXPECTED_POLICY_PARAMETERS,
        actor_critic_share_encoder=False,
    )
    return payload


def _training_plan_transformer(*args, **kwargs) -> Dict[str, Any]:
    plan = dict(_ORIGINAL_TRAINING_PLAN(*args, **kwargs))
    plan["contract"] = THREE_SERVER_CONTRACT
    workspace_root = Path(__file__).resolve().parents[2]
    hashes = dict(plan.get("frozen_source_sha256", {}))
    hashes.pop(str(Path("baseline") / "PPO" / "encoder_v2.py"), None)
    for path in (
        Path(__file__).resolve(),
        Path(__file__).with_name("encoder.py").resolve(),
        Path(__file__).with_name("runtime.py").resolve(),
    ):
        hashes[str(path.relative_to(workspace_root))] = file_digest(path)
    plan["frozen_source_sha256"] = hashes
    return plan


def _transformer_banner(title: str, detail: str = "") -> None:
    suffix = "encoder=plain-MLP-global-Transformer"
    _ORIGINAL_BANNER(
        title.replace("PPO V2", "Plain Transformer + PPO"),
        f"{detail} | {suffix}" if detail else suffix,
    )


def _configure_transformer_runtime() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _ppo_v2.SCHEMA_VERSION = SCHEMA_VERSION
    _ppo_v2.THREE_SERVER_CONTRACT = THREE_SERVER_CONTRACT
    _ppo_v2.build_or_load_ppo = build_or_load_ppo
    _ppo_v2._worker_command = _worker_command_transformer
    _ppo_v2._checkpoint_plan_payload = _checkpoint_plan_payload_transformer
    _ppo_v2._training_plan = _training_plan_transformer
    TrainingTelemetry.banner = staticmethod(_transformer_banner)
    _CONFIGURED = True


def main(argv: Optional[List[str]] = None) -> int:
    _configure_transformer_runtime()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--worker-spec"]:
        if len(arguments) != 2:
            print("ERROR --worker-spec requires exactly one path", flush=True)
            return 2
        return _ppo_v2._worker_main(Path(arguments[1]))
    parser = build_parser()
    args = parser.parse_args(arguments)
    _ppo_v2._validate_args(parser, args)
    return _ppo_v2._run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
