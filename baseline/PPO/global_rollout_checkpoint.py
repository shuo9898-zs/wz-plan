"""Town-boundary checkpointing for one frozen-policy global PPO rollout."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer


SCHEMA_VERSION = 1
BUFFER_ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "returns",
    "episode_starts",
    "values",
    "log_probs",
    "advantages",
)


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_digest(model: Any) -> str:
    """Hash policy parameters only; rollout counters must not affect validity."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_replace_bytes(target: Path, writer) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(target: Path, payload: Mapping[str, Any]) -> None:
    def write(stream) -> None:
        stream.write(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        )

    _atomic_replace_bytes(target, write)


def save_rollout_buffer(target: Path, buffer: RolloutBuffer) -> Dict[str, Any]:
    if not buffer.full or buffer.pos != buffer.buffer_size:
        raise ValueError("Cannot checkpoint an incomplete rollout buffer")

    def write(stream) -> None:
        np.savez_compressed(
            stream,
            **{name: getattr(buffer, name) for name in BUFFER_ARRAYS},
        )

    _atomic_replace_bytes(target, write)
    return {
        "file": target.name,
        "sha256": file_digest(target),
        "size_bytes": target.stat().st_size,
        "n_steps": buffer.buffer_size,
        "arrays": {
            name: {
                "shape": list(getattr(buffer, name).shape),
                "dtype": str(getattr(buffer, name).dtype),
            }
            for name in BUFFER_ARRAYS
        },
    }


def load_rollout_buffer(
    source: Path,
    descriptor: Mapping[str, Any],
    model: Any,
) -> RolloutBuffer:
    if not source.is_file():
        raise ValueError(f"Missing rollout checkpoint: {source}")
    if file_digest(source) != descriptor.get("sha256"):
        raise ValueError(f"Rollout checkpoint hash mismatch: {source}")
    n_steps = int(descriptor["n_steps"])
    buffer = RolloutBuffer(
        n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1,
    )
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != set(BUFFER_ARRAYS):
            raise ValueError(f"Unexpected rollout arrays in {source}")
        for name in BUFFER_ARRAYS:
            array = archive[name]
            expected = getattr(buffer, name)
            if array.shape != expected.shape or array.dtype != expected.dtype:
                raise ValueError(
                    f"Incompatible {name} in {source}: "
                    f"{array.shape}/{array.dtype} != "
                    f"{expected.shape}/{expected.dtype}"
                )
            if name in {"returns", "values", "log_probs", "advantages"} and not np.isfinite(array).all():
                raise ValueError(f"Non-finite {name} in {source}")
            setattr(buffer, name, array.copy())
    buffer.pos = n_steps
    buffer.full = True
    buffer.generator_ready = False
    return buffer


def save_model_atomic(model: Any, target: Path) -> Dict[str, Any]:
    target = target.with_suffix(".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.{uuid.uuid4().hex}.zip")
    try:
        model.save(str(temporary))
        if not temporary.is_file():
            raise RuntimeError(f"SB3 did not create checkpoint {temporary}")
        # Verify the zip is readable before publishing it.
        import zipfile

        with zipfile.ZipFile(temporary, "r") as archive:
            bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt SB3 checkpoint member: {bad_member}")
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "file": target.name,
        "sha256": file_digest(target),
        "size_bytes": target.stat().st_size,
    }


class GlobalRolloutCheckpoint:
    """Commit and validate complete Town fragments for one rollout iteration."""

    def __init__(self, root: Path, plan: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.plan = dict(plan)
        self.plan_fingerprint = canonical_digest(self.plan)
        self.state_path = self.root / "state.json"

    def load(self) -> Dict[str, Any]:
        if not self.state_path.is_file():
            raise ValueError(f"Resume state not found: {self.state_path}")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported global rollout checkpoint schema")
        if state.get("plan_fingerprint") != self.plan_fingerprint:
            raise ValueError(
                "Checkpoint plan/config fingerprint differs from this command"
            )
        model_info = state.get("model")
        if model_info:
            model_path = self.root / model_info["file"]
            if not model_path.is_file() or file_digest(model_path) != model_info["sha256"]:
                raise ValueError("Frozen-policy checkpoint is missing or corrupt")
        for fragment in state.get("fragments", []):
            path = self.root / fragment["file"]
            if not path.is_file() or file_digest(path) != fragment["sha256"]:
                raise ValueError(f"Rollout fragment is missing or corrupt: {path}")
        return state

    def commit_town(
        self,
        *,
        model: Any,
        town: str,
        phase_index: int,
        phase_count: int,
        buffer: RolloutBuffer,
        previous_state: Mapping[str, Any] | None,
        coverage: Mapping[str, int],
        outcomes: Sequence[Mapping[str, Any]],
        episodes: Sequence[Mapping[str, Any]] = (),
        context_summary: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        frozen_digest = policy_digest(model)
        if previous_state and previous_state.get("frozen_policy_sha256") not in (
            None,
            frozen_digest,
        ):
            raise ValueError("Policy changed while the global rollout was being collected")

        buffer_path = self.root / f"fragment_{phase_index:02d}_{town}.npz"
        fragment = save_rollout_buffer(buffer_path, buffer)
        fragment.update(
            {
                "town": town,
                "phase_index": phase_index,
                "coverage": dict(coverage),
                "outcomes": list(outcomes),
                "episodes": list(episodes),
                "context_summary": list(context_summary),
                "valid_steps": int(buffer.buffer_size),
                "physical_env_steps": int(
                    getattr(buffer, "physical_env_steps", buffer.buffer_size)
                ),
                "infrastructure_faults": int(
                    getattr(buffer, "infrastructure_faults", 0)
                ),
                "policy_sha256": frozen_digest,
            }
        )
        fragments = list((previous_state or {}).get("fragments", []))
        fragments = [item for item in fragments if item.get("town") != town]
        fragments.append(fragment)
        fragments.sort(key=lambda item: int(item["phase_index"]))

        model_info = save_model_atomic(model, self.root / "frozen_policy.zip")
        state = {
            "schema_version": SCHEMA_VERSION,
            "state": "ready_to_update" if phase_index + 1 == phase_count else "collecting",
            "next_phase_index": phase_index + 1,
            "plan": self.plan,
            "plan_fingerprint": self.plan_fingerprint,
            "frozen_policy_sha256": frozen_digest,
            "model": model_info,
            "model_num_timesteps": int(model.num_timesteps),
            "model_n_updates": int(model._n_updates),
            "fragments": fragments,
        }
        atomic_write_json(self.state_path, state)
        return state

    def mark_complete(self, state: Mapping[str, Any], final_model: Path) -> Dict[str, Any]:
        completed = dict(state)
        final_path = final_model.with_suffix(".zip")
        completed["state"] = "iteration_complete"
        completed["final_model"] = {
            "file": str(final_path.resolve()),
            "sha256": file_digest(final_path),
            "size_bytes": final_path.stat().st_size,
        }
        atomic_write_json(self.state_path, completed)
        return completed


__all__ = [
    "BUFFER_ARRAYS",
    "GlobalRolloutCheckpoint",
    "canonical_digest",
    "file_digest",
    "load_rollout_buffer",
    "policy_digest",
    "save_model_atomic",
    "save_rollout_buffer",
]
