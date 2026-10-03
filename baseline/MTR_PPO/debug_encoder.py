"""Offline checks for the MTR-PPO feature extractor (no CARLA/SUMO)."""
from __future__ import annotations

import numpy as np
import sys
import torch
from gymnasium import spaces
from pathlib import Path


_BUNDLE_ROOT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_ROOT))

from baseline.MTR_PPO.encoder import InputMatchedMTREncoder
from env.observation_encoder_v2 import DEFAULT_OBSERVATION_SPEC_V2, ObservationEncoderV2


def _representative_observation() -> torch.Tensor:
    spec = DEFAULT_OBSERVATION_SPEC_V2
    layout = ObservationEncoderV2(spec).layout
    observation = torch.zeros(spec.observation_dim, dtype=torch.float32)

    observation[layout.current_ego] = torch.tensor((0.0, 0.0, 0.0, 1.0, 0.25))
    ego_history = observation[layout.ego_history].view(len(spec.history_lags), 6)
    ego_history[:, 0] = 1.0
    ego_history[:, 1] = torch.tensor((-0.01, -0.02, -0.04, -0.07, -0.10))
    ego_history[:, 4] = 1.0
    ego_history[:, 5] = 0.25

    agents = observation[layout.other_agents].view(
        spec.max_other_agents, spec.dynamic_entity_dim
    )
    agents[0, 0:3] = torch.tensor((1.0, 1.0, 0.0))
    agents[0, 3 + spec.local_id_bits : 5 + spec.local_id_bits] = torch.tensor(
        (0.10, 0.02)
    )
    agent_history = agents[0, 5 + spec.local_id_bits :].view(
        len(spec.history_lags), 3
    )
    agent_history[:, 0] = 1.0
    agent_history[:, 1] = torch.tensor((0.095, 0.090, 0.080, 0.065, 0.050))
    agent_history[:, 2] = 0.02

    lanes = observation[layout.lane_segments].view(
        spec.max_lane_segments, spec.lane_segment_dim
    )
    lanes[0] = torch.tensor((1.0, 0.05, 0.0, 0.20, 0.35, 1.0, 0.0))

    workzone = observation[layout.workzone_elements].view(
        spec.max_workzone_elements, spec.workzone_entity_dim
    )
    workzone[0, 0:3] = torch.tensor((1.0, 1.0, 0.0))
    workzone[0, 3 + spec.local_id_bits : 5 + spec.local_id_bits] = torch.tensor(
        (0.08, -0.03)
    )
    return observation


def main() -> int:
    torch.manual_seed(0)
    spec = DEFAULT_OBSERVATION_SPEC_V2
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(spec.observation_dim,),
        dtype=np.float32,
    )
    encoder = InputMatchedMTREncoder(observation_space).cpu()
    representative = _representative_observation()
    batch = torch.stack((representative, representative.clone()), dim=0)

    encoder.eval()
    with torch.no_grad():
        output = encoder(batch)
        empty_output = encoder(torch.zeros_like(batch))
        repeated_output = encoder(batch)
    assert output.shape == (2, encoder.features_dim)
    assert torch.isfinite(output).all()
    assert torch.isfinite(empty_output).all()
    assert torch.equal(output, repeated_output)
    duplicate_max_error = float((output[0] - output[1]).abs().max())
    assert torch.allclose(output[0], output[1], rtol=1.0e-5, atol=1.0e-6), (
        f"identical observations diverged: max_abs_error={duplicate_max_error}"
    )

    encoder.train()
    train_output = encoder(batch)
    loss = train_output.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in encoder.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    print(
        "MTR_ENCODER_DEBUG_OK",
        f"observation_dim={spec.observation_dim}",
        f"features_dim={encoder.features_dim}",
        f"parameters={sum(parameter.numel() for parameter in encoder.parameters())}",
        f"batch_shape={tuple(output.shape)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
