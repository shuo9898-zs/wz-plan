"""Plain MLP + global-Transformer encoder for the PPO V2 observation.

The environment observation and its upstream track association stay unchanged.
Episode-local identity bits are discarded, matching the current encoder and the
MTR baseline.  Each entity is projected once, all valid tokens enter one global
Transformer layer, and the contextualized ego token is returned to PPO.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_SPEC_V2,
    ObservationEncoderV2,
    ObservationSpecV2,
)


TRANSFORMER_ENCODER_CONTRACT = "plain_mlp_global_transformer_encoder_v1"
TRANSFORMER_FEATURES_DIM = 256
TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS = 670_848
TRANSFORMER_EXPECTED_POLICY_PARAMETERS = 1_605_637


@dataclass(frozen=True)
class TransformerNetworkSpec:
    """Small, frozen architecture for the plain encoder baseline."""

    d_model: int = TRANSFORMER_FEATURES_DIM
    token_hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 1
    ffn_dim: int = 512
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.d_model,
            self.token_hidden_dim,
            self.num_heads,
            self.num_layers,
            self.ffn_dim,
        )
        if any(isinstance(value, bool) or int(value) < 1 for value in integer_fields):
            raise ValueError("Transformer dimensions must be positive integers")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def fingerprint_payload(self) -> Dict[str, object]:
        return {
            "contract": TRANSFORMER_ENCODER_CONTRACT,
            **asdict(self),
            "features_dim": self.d_model,
            "parameters_per_extractor": (
                TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS
            ),
            "actor_critic_share_encoder": False,
            "spatial_encoding": False,
            "workzone_specific_attention": False,
        }


DEFAULT_TRANSFORMER_NETWORK_SPEC = TransformerNetworkSpec()


class _TokenMLP(nn.Module):
    """The same simple projection shape for every entity group."""

    def __init__(self, input_dim: int, network: TransformerNetworkSpec) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), network.token_hidden_dim),
            nn.ReLU(),
            nn.Linear(network.token_hidden_dim, network.d_model),
            nn.LayerNorm(network.d_model),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class PlainTransformerEncoder(BaseFeaturesExtractor):
    """Convert the input-matched entity set into one 256-D scene feature."""

    def __init__(
        self,
        observation_space: gym.Space,
        observation_spec: ObservationSpecV2 = DEFAULT_OBSERVATION_SPEC_V2,
        network_spec: TransformerNetworkSpec = DEFAULT_TRANSFORMER_NETWORK_SPEC,
    ) -> None:
        expected_shape = (observation_spec.observation_dim,)
        if tuple(observation_space.shape or ()) != expected_shape:
            raise ValueError(
                "PlainTransformerEncoder requires observation shape "
                f"{expected_shape}, got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim=network_spec.d_model)
        self.observation_spec = observation_spec
        self.network_spec = network_spec
        self.layout = ObservationEncoderV2(observation_spec).layout

        ego_input_dim = self.layout.ego_history.stop - self.layout.current_ego.start
        agent_input_dim = (
            observation_spec.dynamic_entity_dim - observation_spec.local_id_bits
        )
        workzone_input_dim = (
            observation_spec.workzone_entity_dim - observation_spec.local_id_bits
        )
        lane_input_dim = observation_spec.lane_segment_dim

        self.ego_mlp = _TokenMLP(ego_input_dim, network_spec)
        self.agent_mlp = _TokenMLP(agent_input_dim, network_spec)
        self.workzone_mlp = _TokenMLP(workzone_input_dim, network_spec)
        self.lane_mlp = _TokenMLP(lane_input_dim, network_spec)

        layer = nn.TransformerEncoderLayer(
            d_model=network_spec.d_model,
            nhead=network_spec.num_heads,
            dim_feedforward=network_spec.ffn_dim,
            dropout=network_spec.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=network_spec.num_layers,
            norm=nn.LayerNorm(network_spec.d_model),
        )

        actual_parameters = sum(parameter.numel() for parameter in self.parameters())
        if actual_parameters != TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS:
            raise RuntimeError(
                "Plain Transformer parameter contract changed: "
                f"{actual_parameters} != "
                f"{TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS}"
            )

    def _without_local_ids(self, rows: torch.Tensor) -> torch.Tensor:
        bits = self.observation_spec.local_id_bits
        return torch.cat((rows[..., :3], rows[..., 3 + bits :]), dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        spec = self.observation_spec
        batch_size = observations.shape[0]

        ego_values = observations[
            :, self.layout.current_ego.start : self.layout.ego_history.stop
        ]
        ego_token = self.ego_mlp(ego_values).unsqueeze(1)
        ego_valid = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=observations.device
        )

        agent_rows = observations[:, self.layout.other_agents].reshape(
            batch_size,
            spec.max_other_agents,
            spec.dynamic_entity_dim,
        )
        agent_values = self._without_local_ids(agent_rows)
        agent_history_presence = agent_values[..., 5::3]
        agent_valid = (agent_values[..., 0] > 0.5) | torch.any(
            agent_history_presence > 0.5,
            dim=-1,
        )
        agent_tokens = self.agent_mlp(agent_values)

        workzone_rows = observations[:, self.layout.workzone_elements].reshape(
            batch_size,
            spec.max_workzone_elements,
            spec.workzone_entity_dim,
        )
        workzone_values = self._without_local_ids(workzone_rows)
        workzone_valid = workzone_values[..., 0] > 0.5
        workzone_tokens = self.workzone_mlp(workzone_values)

        lane_rows = observations[:, self.layout.lane_segments].reshape(
            batch_size,
            spec.max_lane_segments,
            spec.lane_segment_dim,
        )
        lane_valid = lane_rows[..., 0] > 0.5
        lane_tokens = self.lane_mlp(lane_rows)

        tokens = torch.cat(
            (ego_token, agent_tokens, workzone_tokens, lane_tokens),
            dim=1,
        )
        valid = torch.cat(
            (ego_valid, agent_valid, workzone_valid, lane_valid),
            dim=1,
        )
        contextualized = self.transformer(
            tokens,
            src_key_padding_mask=~valid,
        )
        return contextualized[:, 0, :]


__all__ = [
    "DEFAULT_TRANSFORMER_NETWORK_SPEC",
    "PlainTransformerEncoder",
    "TRANSFORMER_ENCODER_CONTRACT",
    "TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS",
    "TRANSFORMER_EXPECTED_POLICY_PARAMETERS",
    "TRANSFORMER_FEATURES_DIM",
    "TransformerNetworkSpec",
]
