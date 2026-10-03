"""Algorithm-independent spatial-temporal scene encoder for RL V2.

This module knows the observation schema but knows nothing about PPO, SAC,
TD3 or Stable-Baselines3.  The default architecture intentionally preserves
the module names and tensor operations of the running PPO V2 encoder so a
future compatibility switch can load the same state dictionary exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn

from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_SPEC_V2,
    ObservationEncoderV2,
    ObservationSpecV2,
)


SCENE_ENCODER_CONTRACT_VERSION_V2 = "typed_spatiotemporal_cross_attention_v2"


@dataclass(frozen=True)
class SceneEncoderSpecV2:
    """Network dimensions independent from the observation and RL algorithm."""

    token_dim: int = 128
    branch_hidden_dim: int = 256
    attention_heads: int = 8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.token_dim % self.attention_heads != 0:
            raise ValueError("token_dim must be divisible by attention_heads")

    @property
    def features_dim(self) -> int:
        return 2 * self.token_dim

    def fingerprint_payload(self) -> dict:
        return {
            "contract": SCENE_ENCODER_CONTRACT_VERSION_V2,
            **asdict(self),
            "features_dim": self.features_dim,
        }


DEFAULT_SCENE_ENCODER_SPEC_V2 = SceneEncoderSpecV2()
ENCODER_TOKEN_DIM_V2 = DEFAULT_SCENE_ENCODER_SPEC_V2.token_dim
ENCODER_BRANCH_HIDDEN_DIM_V2 = DEFAULT_SCENE_ENCODER_SPEC_V2.branch_hidden_dim
ENCODER_ATTENTION_HEADS_V2 = DEFAULT_SCENE_ENCODER_SPEC_V2.attention_heads
ENCODER_FEATURES_DIM_V2 = DEFAULT_SCENE_ENCODER_SPEC_V2.features_dim


class _TokenEncoderV2(nn.Module):
    def __init__(self, input_dim: int, network: SceneEncoderSpecV2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), network.branch_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(network.branch_hidden_dim),
            nn.Linear(network.branch_hidden_dim, network.token_dim),
            nn.LayerNorm(network.token_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _EgoConditionedPoolV2(nn.Module):
    def __init__(self, network: SceneEncoderSpecV2) -> None:
        super().__init__()
        self.query_bias = nn.Parameter(torch.zeros(1, 1, network.token_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, network.token_dim))
        self.attention = nn.MultiheadAttention(
            network.token_dim,
            network.attention_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(network.token_dim)

    def forward(
        self,
        ego_token: torch.Tensor,
        entity_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = ego_token.shape[0]
        query = ego_token.unsqueeze(1) + self.query_bias
        null_token = self.null_token.expand(batch_size, -1, -1)
        key_value = torch.cat((entity_tokens, null_token), dim=1)
        null_valid = torch.ones(
            (batch_size, 1),
            dtype=torch.bool,
            device=valid_mask.device,
        )
        key_padding_mask = ~torch.cat((valid_mask, null_valid), dim=1)
        pooled, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.output_norm(query + pooled).squeeze(1)


class SceneEncoderMixinV2:
    """Shared implementation used by pure Torch and framework adapters."""

    def _initialize_scene_encoder_v2(
        self,
        observation: ObservationSpecV2,
        network: SceneEncoderSpecV2,
    ) -> None:
        self.observation_spec = observation
        # Preserve the old public attribute for checkpoint/tool compatibility.
        self.spec = observation
        self.network_spec = network
        self.layout = ObservationEncoderV2(observation).layout

        ego_input_dim = self.layout.ego_history.stop - self.layout.current_ego.start
        workzone_input_dim = observation.workzone_entity_dim - observation.local_id_bits
        agent_input_dim = observation.dynamic_entity_dim - observation.local_id_bits
        lane_input_dim = observation.lane_segment_dim

        self.ego_encoder = _TokenEncoderV2(ego_input_dim, network)
        self.agent_encoder = _TokenEncoderV2(agent_input_dim, network)
        self.lane_encoder = _TokenEncoderV2(lane_input_dim, network)
        self.workzone_encoder = _TokenEncoderV2(workzone_input_dim, network)

        self.agent_pool = _EgoConditionedPoolV2(network)
        self.lane_pool = _EgoConditionedPoolV2(network)
        self.workzone_pool = _EgoConditionedPoolV2(network)
        self.fusion_pool = _EgoConditionedPoolV2(network)
        self.output_norm = nn.LayerNorm(network.features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        spec = self.observation_spec
        ego_values = observations[
            :, self.layout.current_ego.start : self.layout.ego_history.stop
        ]
        ego_token = self.ego_encoder(ego_values)

        workzone_rows = observations[:, self.layout.workzone_elements].reshape(
            -1, spec.max_workzone_elements, spec.workzone_entity_dim
        )
        workzone_values = torch.cat(
            (
                workzone_rows[..., :3],
                workzone_rows[..., 3 + spec.local_id_bits :],
            ),
            dim=-1,
        )
        workzone_valid = workzone_rows[..., 0] > 0.5
        workzone_summary = self.workzone_pool(
            ego_token,
            self.workzone_encoder(workzone_values),
            workzone_valid,
        )

        agent_rows = observations[:, self.layout.other_agents].reshape(
            -1, spec.max_other_agents, spec.dynamic_entity_dim
        )
        agent_values = torch.cat(
            (
                agent_rows[..., :3],
                agent_rows[..., 3 + spec.local_id_bits :],
            ),
            dim=-1,
        )
        history_presence = agent_values[..., 5::3]
        agent_valid = (agent_values[..., 0] > 0.5) | torch.any(
            history_presence > 0.5,
            dim=-1,
        )
        agent_summary = self.agent_pool(
            ego_token,
            self.agent_encoder(agent_values),
            agent_valid,
        )

        lane_rows = observations[:, self.layout.lane_segments].reshape(
            -1, spec.max_lane_segments, spec.lane_segment_dim
        )
        lane_valid = lane_rows[..., 0] > 0.5
        lane_summary = self.lane_pool(
            ego_token,
            self.lane_encoder(lane_rows),
            lane_valid,
        )

        modality_summaries = torch.stack(
            (agent_summary, lane_summary, workzone_summary), dim=1
        )
        modality_valid = torch.ones(
            modality_summaries.shape[:2],
            dtype=torch.bool,
            device=observations.device,
        )
        fused = self.fusion_pool(ego_token, modality_summaries, modality_valid)
        return self.output_norm(torch.cat((ego_token, fused), dim=-1))

    def branch_parameter_counts(self) -> Dict[str, int]:
        return {
            "agents": _parameter_count(self.agent_encoder)
            + _parameter_count(self.agent_pool),
            "lanes": _parameter_count(self.lane_encoder)
            + _parameter_count(self.lane_pool),
            "workzone": _parameter_count(self.workzone_encoder)
            + _parameter_count(self.workzone_pool),
        }


class SceneEncoderV2(SceneEncoderMixinV2, nn.Module):
    """Pure PyTorch encoder reusable outside Stable-Baselines3."""

    def __init__(
        self,
        observation: ObservationSpecV2 = DEFAULT_OBSERVATION_SPEC_V2,
        network: SceneEncoderSpecV2 = DEFAULT_SCENE_ENCODER_SPEC_V2,
    ) -> None:
        nn.Module.__init__(self)
        self._initialize_scene_encoder_v2(observation, network)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "DEFAULT_SCENE_ENCODER_SPEC_V2",
    "ENCODER_ATTENTION_HEADS_V2",
    "ENCODER_BRANCH_HIDDEN_DIM_V2",
    "ENCODER_FEATURES_DIM_V2",
    "ENCODER_TOKEN_DIM_V2",
    "SCENE_ENCODER_CONTRACT_VERSION_V2",
    "SceneEncoderMixinV2",
    "SceneEncoderSpecV2",
    "SceneEncoderV2",
]

