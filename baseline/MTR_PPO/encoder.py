"""Input-matched adaptation of MTR's context encoder for PPO.

The environment-facing observation contract is intentionally unchanged.  The
flat 1,311-D PPO V2 observation is restored into the following polylines:

* one ego and up to eight other-agent temporal polylines, each containing the
  current point and the fixed lags 0.1, 0.2, 0.4, 0.7 and 1.0 seconds;
* up to 128 two-point, direction-invariant lane segments;
* up to 12 one-point traffic-cone or warning-sign polylines.

Episode-local identity bits are discarded exactly as in the running PPO V2
encoder.  No Town, scenario, route, traffic-density configuration, raw SUMO
speed, vehicle dimensions, or future state is introduced.  Other-agent
velocity and motion heading are deterministic finite differences of the same
fixed-lag positions already present in the observation.

The MTR core keeps the official full-size dimensions: PointNet-style polyline
encoders, D=256, six local-attention layers, eight heads, FFN=1024, and spatial
kNN with k=16.  Only the contextualized ego token is returned to PPO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from env.observation_encoder_v2 import (
    DEFAULT_OBSERVATION_SPEC_V2,
    ObservationEncoderV2,
    ObservationSpecV2,
)


MTR_ENCODER_CONTRACT = "input_matched_mtr_context_encoder_v1"
MTR_FEATURES_DIM = 256
MTR_EXPECTED_EXTRACTOR_PARAMETERS = 5_116_160
MTR_EXPECTED_POLICY_PARAMETERS = 10_496_261


@dataclass(frozen=True)
class MTRNetworkSpec:
    """Frozen full-size MTR context-encoder dimensions."""

    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1024
    num_neighbors: int = 16
    dropout: float = 0.1
    agent_point_dim: int = 20
    agent_input_dim: int = 21  # semantic point features plus validity
    map_point_dim: int = 9
    agent_hidden_dim: int = 256
    map_hidden_dim: int = 64

    def __post_init__(self) -> None:
        integer_fields = (
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.ffn_dim,
            self.num_neighbors,
            self.agent_point_dim,
            self.agent_input_dim,
            self.map_point_dim,
            self.agent_hidden_dim,
            self.map_hidden_dim,
        )
        if any(isinstance(value, bool) or int(value) < 1 for value in integer_fields):
            raise ValueError("MTR dimensions must be positive integers")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.agent_input_dim != self.agent_point_dim + 1:
            raise ValueError("agent_input_dim must include one validity channel")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def fingerprint_payload(self) -> Dict[str, object]:
        return {
            "contract": MTR_ENCODER_CONTRACT,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "num_neighbors": self.num_neighbors,
            "dropout": self.dropout,
            "agent_point_dim": self.agent_point_dim,
            "agent_input_dim": self.agent_input_dim,
            "map_point_dim": self.map_point_dim,
            "agent_hidden_dim": self.agent_hidden_dim,
            "map_hidden_dim": self.map_hidden_dim,
            "features_dim": self.d_model,
            "parameters_per_extractor": MTR_EXPECTED_EXTRACTOR_PARAMETERS,
            "actor_critic_share_encoder": False,
        }


DEFAULT_MTR_NETWORK_SPEC = MTRNetworkSpec()


def _linear_norm_relu(c_in: int, c_out: int) -> Tuple[nn.Module, ...]:
    return (
        nn.Linear(int(c_in), int(c_out), bias=False),
        nn.BatchNorm1d(int(c_out)),
        nn.ReLU(),
    )


class _PointNetPolylineEncoder(nn.Module):
    """MTR's masked PointNet-style polyline aggregation."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_pre_layers: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        if not 0 < num_pre_layers < num_layers:
            raise ValueError("num_pre_layers must be between zero and num_layers")
        self.hidden_dim = int(hidden_dim)
        self.out_channels = int(out_channels)

        pre_layers = []
        current = int(in_channels)
        for _ in range(int(num_pre_layers)):
            pre_layers.extend(_linear_norm_relu(current, self.hidden_dim))
            current = self.hidden_dim
        self.pre_mlps = nn.Sequential(*pre_layers)

        post_layers = []
        current = 2 * self.hidden_dim
        for _ in range(int(num_layers) - int(num_pre_layers)):
            post_layers.extend(_linear_norm_relu(current, self.hidden_dim))
            current = self.hidden_dim
        self.mlps = nn.Sequential(*post_layers)

        self.out_mlps = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_channels, bias=True),
        )

    @staticmethod
    def _masked_max(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        minimum = torch.finfo(values.dtype).min
        masked = values.masked_fill(~valid_mask.unsqueeze(-1), minimum)
        pooled = masked.max(dim=dim).values
        any_valid = valid_mask.any(dim=dim)
        return torch.where(any_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))

    def forward(
        self,
        polylines: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_polylines, num_points, _ = polylines.shape
        point_mask = point_mask.to(dtype=torch.bool)

        point_features = polylines.new_zeros(
            batch_size,
            num_polylines,
            num_points,
            self.hidden_dim,
        )
        if torch.any(point_mask):
            point_features[point_mask] = self.pre_mlps(polylines[point_mask])

        pooled = self._masked_max(point_features, point_mask, dim=2)
        pooled_repeated = pooled.unsqueeze(2).expand(-1, -1, num_points, -1)
        combined = torch.cat((point_features, pooled_repeated), dim=-1)

        refined = torch.zeros_like(point_features)
        if torch.any(point_mask):
            refined[point_mask] = self.mlps(combined[point_mask])
        polyline_features = self._masked_max(refined, point_mask, dim=2)

        polyline_mask = point_mask.any(dim=2)
        output = polylines.new_zeros(
            batch_size,
            num_polylines,
            self.out_channels,
        )
        if torch.any(polyline_mask):
            output[polyline_mask] = self.out_mlps(polyline_features[polyline_mask])
        return output


def _sine_position_embedding(
    positions_xy: torch.Tensor,
    d_model: int,
    temperature: float = 10_000.0,
) -> torch.Tensor:
    """Two-dimensional sinusoidal encoding matching MTR's spatial use."""
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by four for XY sine encoding")
    features_per_axis = d_model // 2
    dimension = torch.arange(
        features_per_axis,
        dtype=positions_xy.dtype,
        device=positions_xy.device,
    )
    scale = temperature ** (
        2.0 * torch.div(dimension, 2, rounding_mode="floor") / features_per_axis
    )

    def encode_axis(axis: torch.Tensor) -> torch.Tensor:
        phase = axis.unsqueeze(-1) / scale
        encoded = torch.stack(
            (phase[..., 0::2].sin(), phase[..., 1::2].cos()),
            dim=-1,
        )
        return encoded.flatten(start_dim=-2)

    return torch.cat(
        (encode_axis(positions_xy[..., 1]), encode_axis(positions_xy[..., 0])),
        dim=-1,
    )


def _spatial_knn_attention_mask(
    positions_xy: torch.Tensor,
    token_mask: torch.Tensor,
    num_neighbors: int,
) -> torch.Tensor:
    """Return [B,N,N] allowed-attention mask for exact spatial kNN."""
    token_mask = token_mask.to(dtype=torch.bool)
    batch_size, num_tokens, _ = positions_xy.shape
    k = min(int(num_neighbors), int(num_tokens))
    distances = torch.cdist(positions_xy, positions_xy, p=2)
    distances = distances.masked_fill(~token_mask.unsqueeze(1), float("inf"))
    distances = distances.masked_fill(~token_mask.unsqueeze(2), float("inf"))
    neighbor_indices = distances.topk(k=k, dim=-1, largest=False).indices

    expanded_valid = token_mask.unsqueeze(1).expand(-1, num_tokens, -1)
    neighbor_valid = torch.gather(expanded_valid, 2, neighbor_indices)
    neighbor_valid = neighbor_valid & token_mask.unsqueeze(-1)
    allowed = torch.zeros(
        batch_size,
        num_tokens,
        num_tokens,
        dtype=torch.bool,
        device=positions_xy.device,
    )
    allowed.scatter_(2, neighbor_indices, neighbor_valid)

    # Invalid queries are zeroed after every layer.  Giving each one a single
    # self edge avoids an all-masked SDPA row without exposing it to valid keys.
    diagonal = torch.eye(
        num_tokens,
        dtype=torch.bool,
        device=positions_xy.device,
    ).unsqueeze(0)
    allowed = allowed | (diagonal & (~token_mask).unsqueeze(-1))
    return allowed


class _MTRLocalTransformerLayer(nn.Module):
    """Post-norm local self-attention layer with MTR parameterization."""

    def __init__(self, network: MTRNetworkSpec) -> None:
        super().__init__()
        self.d_model = network.d_model
        self.num_heads = network.num_heads
        self.head_dim = network.d_model // network.num_heads
        self.dropout_probability = float(network.dropout)

        self.q_proj = nn.Linear(network.d_model, network.d_model, bias=True)
        self.k_proj = nn.Linear(network.d_model, network.d_model, bias=True)
        self.v_proj = nn.Linear(network.d_model, network.d_model, bias=True)
        self.out_proj = nn.Linear(network.d_model, network.d_model, bias=True)
        self.linear1 = nn.Linear(network.d_model, network.ffn_dim, bias=True)
        self.linear2 = nn.Linear(network.ffn_dim, network.d_model, bias=True)
        self.norm1 = nn.LayerNorm(network.d_model)
        self.norm2 = nn.LayerNorm(network.d_model)
        self.dropout1 = nn.Dropout(network.dropout)
        self.dropout2 = nn.Dropout(network.dropout)
        self.ffn_dropout = nn.Dropout(network.dropout)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = values.shape
        return values.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        values: torch.Tensor,
        position_embedding: torch.Tensor,
        allowed_attention: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_key_input = values + position_embedding
        query = self._heads(self.q_proj(query_key_input))
        key = self._heads(self.k_proj(query_key_input))
        value = self._heads(self.v_proj(values))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_attention.unsqueeze(1),
            dropout_p=self.dropout_probability if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view_as(values)
        values = self.norm1(values + self.dropout1(self.out_proj(attended)))
        feedforward = self.linear2(
            self.ffn_dropout(F.relu(self.linear1(values)))
        )
        values = self.norm2(values + self.dropout2(feedforward))
        return values * token_mask.unsqueeze(-1).to(dtype=values.dtype)


class InputMatchedMTREncoder(BaseFeaturesExtractor):
    """Stable-Baselines3 adapter for the input-matched MTR context encoder."""

    def __init__(
        self,
        observation_space: gym.Space,
        observation: ObservationSpecV2 = DEFAULT_OBSERVATION_SPEC_V2,
        network: MTRNetworkSpec = DEFAULT_MTR_NETWORK_SPEC,
    ) -> None:
        expected_shape = (observation.observation_dim,)
        if observation_space.shape != expected_shape:
            raise ValueError(
                "MTR PPO expected observation shape "
                f"{expected_shape}, got {observation_space.shape}"
            )
        if network.d_model != MTR_FEATURES_DIM:
            raise ValueError("The formal MTR baseline requires a 256-D output")
        super().__init__(observation_space, features_dim=network.d_model)
        self.observation_spec = observation
        self.network_spec = network
        self.layout = ObservationEncoderV2(observation).layout

        self.agent_polyline_encoder = _PointNetPolylineEncoder(
            in_channels=network.agent_input_dim,
            hidden_dim=network.agent_hidden_dim,
            num_layers=3,
            num_pre_layers=1,
            out_channels=network.d_model,
        )
        self.map_polyline_encoder = _PointNetPolylineEncoder(
            in_channels=network.map_point_dim,
            hidden_dim=network.map_hidden_dim,
            num_layers=5,
            num_pre_layers=3,
            out_channels=network.d_model,
        )
        self.self_attention_layers = nn.ModuleList(
            [_MTRLocalTransformerLayer(network) for _ in range(network.num_layers)]
        )

        actual_parameters = sum(parameter.numel() for parameter in self.parameters())
        if actual_parameters != MTR_EXPECTED_EXTRACTOR_PARAMETERS:
            raise RuntimeError(
                "MTR extractor parameter contract changed: "
                f"{actual_parameters} != {MTR_EXPECTED_EXTRACTOR_PARAMETERS}"
            )

    def _without_local_ids(
        self,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        bits = self.observation_spec.local_id_bits
        return torch.cat((rows[..., :3], rows[..., 3 + bits :]), dim=-1)

    @staticmethod
    def _finite_difference_velocity(
        positions: torch.Tensor,
        point_mask: torch.Tensor,
        ages_s: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate forward-time velocity at each point from valid positions."""
        num_times = positions.shape[-2]
        velocity = torch.zeros_like(positions)
        assigned = torch.zeros_like(point_mask, dtype=torch.bool)

        for current in range(num_times):
            for older in range(current + 1, num_times):
                usable = (
                    point_mask[..., current]
                    & point_mask[..., older]
                    & ~assigned[..., current]
                )
                delta_t = ages_s[older] - ages_s[current]
                estimate = (
                    positions[..., current, :] - positions[..., older, :]
                ) / delta_t
                velocity[..., current, :] = torch.where(
                    usable.unsqueeze(-1),
                    estimate,
                    velocity[..., current, :],
                )
                assigned[..., current] = assigned[..., current] | usable

            for newer in range(current - 1, -1, -1):
                usable = (
                    point_mask[..., current]
                    & point_mask[..., newer]
                    & ~assigned[..., current]
                )
                delta_t = ages_s[current] - ages_s[newer]
                estimate = (
                    positions[..., newer, :] - positions[..., current, :]
                ) / delta_t
                velocity[..., current, :] = torch.where(
                    usable.unsqueeze(-1),
                    estimate,
                    velocity[..., current, :],
                )
                assigned[..., current] = assigned[..., current] | usable
        return velocity

    @staticmethod
    def _motion_heading(velocity: torch.Tensor) -> torch.Tensor:
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        moving = speed > 1.0e-4
        safe_speed = speed.clamp_min(1.0e-4)
        sine = torch.where(moving, velocity[..., 1] / safe_speed, 0.0)
        cosine = torch.where(moving, velocity[..., 0] / safe_speed, 0.0)
        return torch.stack((sine, cosine), dim=-1)

    def _dynamic_polylines(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return points, point masks, and spatial anchors for ego+agents."""
        spec = self.observation_spec
        batch_size = observations.shape[0]
        num_times = 1 + len(spec.history_lags)
        ages_s = observations.new_tensor((0.0, *spec.history_times_s))
        timestamps = -ages_s
        time_one_hot = torch.eye(
            num_times,
            dtype=observations.dtype,
            device=observations.device,
        )

        current_ego = observations[:, self.layout.current_ego]
        ego_history = observations[:, self.layout.ego_history].reshape(
            batch_size,
            len(spec.history_lags),
            6,
        )
        ego_point_mask = torch.cat(
            (
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=observations.device,
                ),
                ego_history[..., 0] > 0.5,
            ),
            dim=1,
        )
        ego_positions = torch.cat(
            (
                observations.new_zeros(batch_size, 1, 2),
                ego_history[..., 1:3] * spec.history_distance_scale_m,
            ),
            dim=1,
        )
        ego_heading = torch.cat(
            (
                observations.new_tensor((0.0, 1.0)).view(1, 1, 2).expand(
                    batch_size, -1, -1
                ),
                ego_history[..., 3:5],
            ),
            dim=1,
        )
        ego_speed = torch.cat(
            (
                current_ego[:, 4:5] * spec.max_ego_speed_mps,
                ego_history[..., 5] * spec.max_ego_speed_mps,
            ),
            dim=1,
        )
        ego_velocity = torch.stack(
            (
                ego_speed * ego_heading[..., 1],
                ego_speed * ego_heading[..., 0],
            ),
            dim=-1,
        )
        ego_type = observations.new_zeros(batch_size, num_times, 3)
        ego_type[..., 0] = 1.0
        ego_context = observations.new_zeros(batch_size, num_times, 4)
        ego_context[:, 0, :] = current_ego[:, :4]
        ego_semantic = torch.cat(
            (
                ego_type,
                ego_positions,
                ego_heading,
                ego_velocity,
                time_one_hot.unsqueeze(0).expand(batch_size, -1, -1),
                timestamps.view(1, num_times, 1).expand(batch_size, -1, -1),
                ego_context,
            ),
            dim=-1,
        )

        raw_agents = observations[:, self.layout.other_agents].reshape(
            batch_size,
            spec.max_other_agents,
            spec.dynamic_entity_dim,
        )
        agents = self._without_local_ids(raw_agents)
        current_agent_mask = agents[..., 0] > 0.5
        current_agent_position = (
            agents[..., 3:5] * spec.perception_radius_m
        )
        history = agents[..., 5:].reshape(
            batch_size,
            spec.max_other_agents,
            len(spec.history_lags),
            3,
        )
        agent_point_mask = torch.cat(
            (current_agent_mask.unsqueeze(-1), history[..., 0] > 0.5),
            dim=-1,
        )
        agent_positions = torch.cat(
            (
                current_agent_position.unsqueeze(-2),
                history[..., 1:3] * spec.perception_radius_m,
            ),
            dim=-2,
        )
        agent_velocity = self._finite_difference_velocity(
            agent_positions,
            agent_point_mask,
            ages_s,
        )
        agent_heading = self._motion_heading(agent_velocity)
        agent_type = observations.new_zeros(
            batch_size,
            spec.max_other_agents,
            num_times,
            3,
        )
        agent_type[..., 1:] = agents[..., 1:3].unsqueeze(-2).expand(
            -1, -1, num_times, -1
        )
        agent_time_one_hot = time_one_hot.view(
            1, 1, num_times, num_times
        ).expand(batch_size, spec.max_other_agents, -1, -1)
        agent_timestamps = timestamps.view(1, 1, num_times, 1).expand(
            batch_size,
            spec.max_other_agents,
            -1,
            -1,
        )
        agent_context = observations.new_zeros(
            batch_size,
            spec.max_other_agents,
            num_times,
            4,
        )
        agent_semantic = torch.cat(
            (
                agent_type,
                agent_positions,
                agent_heading,
                agent_velocity,
                agent_time_one_hot,
                agent_timestamps,
                agent_context,
            ),
            dim=-1,
        )

        semantic = torch.cat(
            (ego_semantic.unsqueeze(1), agent_semantic),
            dim=1,
        )
        point_mask = torch.cat(
            (ego_point_mask.unsqueeze(1), agent_point_mask),
            dim=1,
        )
        point_values = torch.cat(
            (semantic, point_mask.unsqueeze(-1).to(dtype=semantic.dtype)),
            dim=-1,
        )

        # Points are ordered newest-to-oldest, so the first valid position is
        # the same last-valid anchor used by the original MTR context encoder.
        first_valid = point_mask.to(dtype=torch.int64).argmax(dim=-1)
        anchors = torch.gather(
            torch.cat(
                (ego_positions.unsqueeze(1), agent_positions),
                dim=1,
            ),
            2,
            first_valid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2),
        ).squeeze(2)
        return point_values, point_mask, anchors

    def _static_polylines(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return two-point lane and one-point cone/sign polylines."""
        spec = self.observation_spec
        batch_size = observations.shape[0]

        lanes = observations[:, self.layout.lane_segments].reshape(
            batch_size,
            spec.max_lane_segments,
            spec.lane_segment_dim,
        )
        lane_mask = lanes[..., 0] > 0.5
        midpoint = lanes[..., 1:3] * spec.perception_radius_m
        length = lanes[..., 3:4] * spec.lane_length_scale_m
        width = lanes[..., 4:5] * spec.lane_width_scale_m
        axis = lanes[..., 5:7]
        theta = 0.5 * torch.atan2(axis[..., 1], axis[..., 0])
        direction = torch.stack((theta.cos(), theta.sin()), dim=-1)
        half_vector = 0.5 * length * direction
        endpoints = torch.stack(
            (midpoint - half_vector, midpoint + half_vector),
            dim=-2,
        )
        repeated_geometry = torch.cat((length, width, axis), dim=-1)
        repeated_geometry = repeated_geometry.unsqueeze(-2).expand(-1, -1, 2, -1)
        lane_type = observations.new_zeros(
            batch_size,
            spec.max_lane_segments,
            2,
            3,
        )
        lane_type[..., 0] = 1.0
        lane_points = torch.cat(
            (endpoints, repeated_geometry, lane_type),
            dim=-1,
        )
        lane_point_mask = lane_mask.unsqueeze(-1).expand(-1, -1, 2)

        raw_workzone = observations[:, self.layout.workzone_elements].reshape(
            batch_size,
            spec.max_workzone_elements,
            spec.workzone_entity_dim,
        )
        workzone = self._without_local_ids(raw_workzone)
        workzone_mask = workzone[..., 0] > 0.5
        workzone_position = workzone[..., 3:5] * spec.perception_radius_m
        workzone_points = observations.new_zeros(
            batch_size,
            spec.max_workzone_elements,
            2,
            self.network_spec.map_point_dim,
        )
        workzone_points[..., 0, 0:2] = workzone_position
        workzone_points[..., 0, 7:9] = workzone[..., 1:3]
        workzone_point_mask = torch.zeros(
            batch_size,
            spec.max_workzone_elements,
            2,
            dtype=torch.bool,
            device=observations.device,
        )
        workzone_point_mask[..., 0] = workzone_mask

        points = torch.cat((lane_points, workzone_points), dim=1)
        point_mask = torch.cat((lane_point_mask, workzone_point_mask), dim=1)
        anchors = torch.cat((midpoint, workzone_position), dim=1)
        return points, point_mask, anchors

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        dynamic_points, dynamic_point_mask, dynamic_anchors = (
            self._dynamic_polylines(observations)
        )
        static_points, static_point_mask, static_anchors = (
            self._static_polylines(observations)
        )
        dynamic_tokens = self.agent_polyline_encoder(
            dynamic_points,
            dynamic_point_mask,
        )
        static_tokens = self.map_polyline_encoder(
            static_points,
            static_point_mask,
        )
        token_values = torch.cat((dynamic_tokens, static_tokens), dim=1)
        token_mask = torch.cat(
            (dynamic_point_mask.any(dim=-1), static_point_mask.any(dim=-1)),
            dim=1,
        )
        token_positions = torch.cat((dynamic_anchors, static_anchors), dim=1)
        token_values = token_values * token_mask.unsqueeze(-1).to(
            dtype=token_values.dtype
        )
        position_embedding = _sine_position_embedding(
            token_positions,
            self.network_spec.d_model,
        )
        position_embedding = position_embedding * token_mask.unsqueeze(-1).to(
            dtype=position_embedding.dtype
        )
        allowed_attention = _spatial_knn_attention_mask(
            token_positions,
            token_mask,
            self.network_spec.num_neighbors,
        )
        for layer in self.self_attention_layers:
            token_values = layer(
                token_values,
                position_embedding,
                allowed_attention,
                token_mask,
            )
        return token_values[:, 0, :]


__all__ = [
    "DEFAULT_MTR_NETWORK_SPEC",
    "InputMatchedMTREncoder",
    "MTR_ENCODER_CONTRACT",
    "MTR_EXPECTED_EXTRACTOR_PARAMETERS",
    "MTR_EXPECTED_POLICY_PARAMETERS",
    "MTR_FEATURES_DIM",
    "MTRNetworkSpec",
]
