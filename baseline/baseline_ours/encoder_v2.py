"""Typed spatial-temporal cross-attention feature extractor for PPO V2.

The environment keeps one flat, finite Gymnasium ``Box`` for rollout and
checkpoint compatibility.  This module restores structure inside the policy:
other-agent tracks, undirected lane segments, and work-zone elements are
encoded by parameter-balanced branches, pooled with ego-conditioned cross
attention, and fused into one scene representation.

SB3 instantiates this extractor twice with ``share_features_extractor=False``:
the actor and critic therefore learn independent representations even though
they consume the same observation contract.
"""
from __future__ import annotations

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


ENCODER_CONTRACT_VERSION_V2 = "typed_spatiotemporal_cross_attention_v2"
ENCODER_TOKEN_DIM_V2 = 128
ENCODER_BRANCH_HIDDEN_DIM_V2 = 256
ENCODER_ATTENTION_HEADS_V2 = 8
ENCODER_FEATURES_DIM_V2 = 256


class _TokenEncoderV2(nn.Module):
    """Map one typed entity row to the shared 128-D token space."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), ENCODER_BRANCH_HIDDEN_DIM_V2),
            nn.SiLU(),
            nn.LayerNorm(ENCODER_BRANCH_HIDDEN_DIM_V2),
            nn.Linear(ENCODER_BRANCH_HIDDEN_DIM_V2, ENCODER_TOKEN_DIM_V2),
            nn.LayerNorm(ENCODER_TOKEN_DIM_V2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _EgoConditionedPoolV2(nn.Module):
    """Cross-attend one ego query to a masked variable-size entity set."""

    def __init__(self) -> None:
        super().__init__()
        self.query_bias = nn.Parameter(torch.zeros(1, 1, ENCODER_TOKEN_DIM_V2))
        self.null_token = nn.Parameter(torch.zeros(1, 1, ENCODER_TOKEN_DIM_V2))
        self.attention = nn.MultiheadAttention(
            ENCODER_TOKEN_DIM_V2,
            ENCODER_ATTENTION_HEADS_V2,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(ENCODER_TOKEN_DIM_V2)

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


class CrossAttentionEncoderV2(BaseFeaturesExtractor):
    """SB3 feature extractor for the flat V2 observation contract."""

    def __init__(
        self,
        observation_space: gym.Space,
        spec: ObservationSpecV2 = DEFAULT_OBSERVATION_SPEC_V2,
    ) -> None:
        expected_shape = (spec.observation_dim,)
        if observation_space.shape != expected_shape:
            raise ValueError(
                "PPO V2 encoder expected observation shape "
                f"{expected_shape}, got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim=ENCODER_FEATURES_DIM_V2)
        self.spec = spec
        self.layout = ObservationEncoderV2(spec).layout

        ego_input_dim = (
            self.layout.ego_history.stop - self.layout.current_ego.start
        )
        workzone_input_dim = spec.workzone_entity_dim - spec.local_id_bits
        agent_input_dim = spec.dynamic_entity_dim - spec.local_id_bits
        lane_input_dim = spec.lane_segment_dim

        self.ego_encoder = _TokenEncoderV2(ego_input_dim)
        self.agent_encoder = _TokenEncoderV2(agent_input_dim)
        self.lane_encoder = _TokenEncoderV2(lane_input_dim)
        self.workzone_encoder = _TokenEncoderV2(workzone_input_dim)

        self.agent_pool = _EgoConditionedPoolV2()
        self.lane_pool = _EgoConditionedPoolV2()
        self.workzone_pool = _EgoConditionedPoolV2()
        self.fusion_pool = _EgoConditionedPoolV2()
        self.output_norm = nn.LayerNorm(ENCODER_FEATURES_DIM_V2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ego_values = observations[
            :, self.layout.current_ego.start : self.layout.ego_history.stop
        ]
        ego_token = self.ego_encoder(ego_values)

        workzone_rows = observations[:, self.layout.workzone_elements].reshape(
            -1,
            self.spec.max_workzone_elements,
            self.spec.workzone_entity_dim,
        )
        workzone_values = torch.cat(
            (
                workzone_rows[..., :3],
                workzone_rows[..., 3 + self.spec.local_id_bits :],
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
            -1,
            self.spec.max_other_agents,
            self.spec.dynamic_entity_dim,
        )
        agent_values = torch.cat(
            (
                agent_rows[..., :3],
                agent_rows[..., 3 + self.spec.local_id_bits :],
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
            -1,
            self.spec.max_lane_segments,
            self.spec.lane_segment_dim,
        )
        lane_valid = lane_rows[..., 0] > 0.5
        lane_summary = self.lane_pool(
            ego_token,
            self.lane_encoder(lane_rows),
            lane_valid,
        )

        modality_summaries = torch.stack(
            (agent_summary, lane_summary, workzone_summary),
            dim=1,
        )
        modality_valid = torch.ones(
            modality_summaries.shape[:2],
            dtype=torch.bool,
            device=observations.device,
        )
        fused = self.fusion_pool(
            ego_token,
            modality_summaries,
            modality_valid,
        )
        return self.output_norm(torch.cat((ego_token, fused), dim=-1))

    def branch_parameter_counts(self) -> Dict[str, int]:
        """Return comparable per-modality encoder+pool parameter budgets."""
        return {
            "agents": _parameter_count(self.agent_encoder)
            + _parameter_count(self.agent_pool),
            "lanes": _parameter_count(self.lane_encoder)
            + _parameter_count(self.lane_pool),
            "workzone": _parameter_count(self.workzone_encoder)
            + _parameter_count(self.workzone_pool),
        }


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "CrossAttentionEncoderV2",
    "ENCODER_ATTENTION_HEADS_V2",
    "ENCODER_BRANCH_HIDDEN_DIM_V2",
    "ENCODER_CONTRACT_VERSION_V2",
    "ENCODER_FEATURES_DIM_V2",
    "ENCODER_TOKEN_DIM_V2",
]
