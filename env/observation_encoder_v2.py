"""Generalization-oriented, fixed-lag PPO observation encoder (V2).

The encoder deliberately has no scenario, Town, work-zone, layout, route, or
origin input.  ``ObjectSampleV2.object_id`` is only an episode-local identity
key used to keep a cone, sign, SUMO vehicle, or pedestrian in the same slot
across frames. Raw ID text is returned as diagnostics but never copied into
the numeric vector. The flat contract retains a shuffled episode-local
pseudonym, while ``baseline.PPO.encoder_v2`` deliberately removes those bits
before learned feature extraction; identity is represented by each track.

At 10 Hz the default history lags are 0.1, 0.2, 0.4, 0.7, and 1.0 seconds.
Time is represented by fixed vector positions; timestamps are intentionally
not policy features.  Work-zone elements are current-frame only.  Dynamic
actor history is matched by ``(kind, object_id)`` and expressed in the current
ego frame, so a reused slot can never inherit another object's trajectory.
Driving-lane geometry is represented by direction-invariant local segments:
the segment axis is encoded with ``sin(2 theta)``/``cos(2 theta)``, so reversing
the source waypoint direction produces the same policy input.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np


OBSERVATION_CONTRACT_VERSION_V2 = "local_entities_lane_segments_v2"

ObjectKindV2 = Literal[
    "traffic_cone",
    "warning_sign",
    "sumo_vehicle",
    "pedestrian",
]
ObjectKeyV2 = Tuple[ObjectKindV2, str]

WORKZONE_KIND_ORDER_V2: Tuple[ObjectKindV2, ...] = (
    "traffic_cone",
    "warning_sign",
)
OTHER_AGENT_KIND_ORDER_V2: Tuple[ObjectKindV2, ...] = (
    "sumo_vehicle",
    "pedestrian",
)
WORKZONE_KINDS_V2 = frozenset(WORKZONE_KIND_ORDER_V2)
OTHER_AGENT_KINDS_V2 = frozenset(OTHER_AGENT_KIND_ORDER_V2)
KIND_ORDER_V2: Tuple[ObjectKindV2, ...] = (
    *WORKZONE_KIND_ORDER_V2,
    *OTHER_AGENT_KIND_ORDER_V2,
)

CURRENT_EGO_DIM_V2 = 5
EGO_HISTORY_FRAME_DIM_V2 = 6
DYNAMIC_HISTORY_FRAME_DIM_V2 = 3
GROUP_KIND_DIM_V2 = 2
LANE_SEGMENT_DIM_V2 = 7


@dataclass(frozen=True)
class EgoSampleV2:
    """One world-frame ego sample."""

    x: float
    y: float
    heading_deg: float
    speed_mps: float


@dataclass(frozen=True)
class ObjectSampleV2:
    """One perceived object with an identity stable inside one episode.

    ``object_id`` identifies the physical object, not its scenario.  Callers
    may use simulator IDs internally; the encoder exposes them only as slot
    metadata and maps them to a freshly salted binary pseudonym in the flat
    transport contract. The PPO V2 feature extractor discards those bits.
    """

    object_id: str
    kind: ObjectKindV2
    x: float
    y: float


@dataclass(frozen=True)
class LaneSegmentSampleV2:
    """One undirected driving-lane center segment in world coordinates.

    ``segment_id`` is diagnostics-only. ``axis_heading_deg`` may come from a
    map waypoint, but the numeric encoder removes its 180-degree direction.
    """

    segment_id: str
    x: float
    y: float
    axis_heading_deg: float
    length_m: float
    width_m: float


@dataclass(frozen=True)
class ObservationSpecV2:
    """Scenario-independent capacities, lags, and physical scales."""

    history_lags: Tuple[int, ...] = (1, 2, 4, 7, 10)
    max_workzone_elements: int = 12
    max_other_agents: int = 8
    max_lane_segments: int = 128
    perception_radius_m: float = 50.0
    ego_position_scale_m: float = 120.0
    history_distance_scale_m: float = 20.0
    max_ego_speed_mps: float = 13.89
    lane_length_scale_m: float = 10.0
    lane_width_scale_m: float = 5.0
    local_id_bits: int = 8
    missing_actor_ttl_steps: int = 10
    control_dt_s: float = 0.1

    def __post_init__(self) -> None:
        lags = tuple(self.history_lags)
        object.__setattr__(self, "history_lags", lags)
        if not lags:
            raise ValueError("history_lags must contain at least one lag")
        if any(
            isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1
            for lag in lags
        ):
            raise ValueError("history_lags must contain positive integer frame offsets")
        if tuple(sorted(set(int(lag) for lag in lags))) != lags:
            raise ValueError("history_lags must be unique and strictly increasing")

        integer_fields = (
            ("max_workzone_elements", self.max_workzone_elements),
            ("max_other_agents", self.max_other_agents),
            ("max_lane_segments", self.max_lane_segments),
            ("local_id_bits", self.local_id_bits),
        )
        for name, value in integer_fields:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.local_id_bits > 16:
            raise ValueError("local_id_bits must not exceed 16")
        token_capacity = 2 ** self.local_id_bits - 1
        if token_capacity < max(
            self.max_workzone_elements,
            self.max_other_agents,
        ):
            raise ValueError(
                "local_id_bits cannot uniquely identify every active slot"
            )
        if (
            isinstance(self.missing_actor_ttl_steps, bool)
            or not isinstance(self.missing_actor_ttl_steps, (int, np.integer))
            or self.missing_actor_ttl_steps < 0
        ):
            raise ValueError("missing_actor_ttl_steps must be a non-negative integer")
        for name, value in (
            ("perception_radius_m", self.perception_radius_m),
            ("ego_position_scale_m", self.ego_position_scale_m),
            ("history_distance_scale_m", self.history_distance_scale_m),
            ("max_ego_speed_mps", self.max_ego_speed_mps),
            ("lane_length_scale_m", self.lane_length_scale_m),
            ("lane_width_scale_m", self.lane_width_scale_m),
            ("control_dt_s", self.control_dt_s),
        ):
            if not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def history_frames(self) -> int:
        """Number of selected lag frames (read-only compatibility helper)."""
        return len(self.history_lags)

    @property
    def max_history_lag(self) -> int:
        return self.history_lags[-1]

    @property
    def history_times_s(self) -> Tuple[float, ...]:
        """Physical age of every fixed lag in the synchronous control loop."""
        return tuple(
            round(float(lag) * self.control_dt_s, 12)
            for lag in self.history_lags
        )

    @property
    def workzone_entity_dim(self) -> int:
        # presence + cone/sign + one episode-local token + relative x/y
        return 1 + GROUP_KIND_DIM_V2 + self.local_id_bits + 2

    @property
    def dynamic_entity_dim(self) -> int:
        # Current entity header plus presence/x/y for each fixed lag.
        return (
            1
            + GROUP_KIND_DIM_V2
            + self.local_id_bits
            + 2
            + len(self.history_lags) * DYNAMIC_HISTORY_FRAME_DIM_V2
        )

    @property
    def lane_segment_dim(self) -> int:
        # presence + ego-frame midpoint x/y + length/width + undirected axis
        return LANE_SEGMENT_DIM_V2

    @property
    def observation_dim(self) -> int:
        return (
            CURRENT_EGO_DIM_V2
            + len(self.history_lags) * EGO_HISTORY_FRAME_DIM_V2
            + self.max_workzone_elements * self.workzone_entity_dim
            + self.max_other_agents * self.dynamic_entity_dim
            + self.max_lane_segments * self.lane_segment_dim
        )


@dataclass(frozen=True)
class ObservationLayoutV2:
    """Named slices in the flat PPO vector."""

    current_ego: slice
    ego_history: slice
    workzone_elements: slice
    other_agents: slice
    lane_segments: slice


@dataclass(frozen=True)
class EncodedObservationV2:
    """Flat observation transport plus raw IDs for logging/debugging only."""

    values: np.ndarray
    workzone_slot_ids: Tuple[Optional[str], ...]
    other_agent_slot_ids: Tuple[Optional[str], ...]
    lane_segment_ids: Tuple[Optional[str], ...]

    @property
    def observation_ids(self) -> Dict[str, Tuple[Optional[str], ...]]:
        return {
            "workzone": self.workzone_slot_ids,
            "other_agents": self.other_agent_slot_ids,
            "lanes": self.lane_segment_ids,
        }


class _EpisodeIdentityCodecV2:
    """Map object keys to shuffled, unique episode-local binary tokens."""

    def __init__(self, bits: int, rng: np.random.Generator) -> None:
        self.bits = int(bits)
        tokens = np.arange(1, 2 ** self.bits, dtype=np.uint32)
        rng.shuffle(tokens)
        self._available = iter(int(value) for value in tokens)
        self._tokens: Dict[ObjectKeyV2, int] = {}

    def encode(self, key: ObjectKeyV2) -> np.ndarray:
        if key not in self._tokens:
            try:
                self._tokens[key] = next(self._available)
            except StopIteration as error:
                raise RuntimeError(
                    "Episode-local object-ID capacity was exhausted; "
                    "increase local_id_bits"
                ) from error
        token = self._tokens[key]
        return np.asarray(
            [(token >> bit) & 1 for bit in range(self.bits)],
            dtype=np.float32,
        )


VisibleRecordV2 = Tuple[ObjectSampleV2, float, float, float]
VisibleLaneRecordV2 = Tuple[LaneSegmentSampleV2, float, float, float]


@dataclass(frozen=True)
class _HistoryFrameV2:
    ego: EgoSampleV2
    other_agents: Dict[ObjectKeyV2, VisibleRecordV2]


class _StableSlotsV2:
    """Keep selected raw object keys in stable nearest-object slots."""

    def __init__(
        self,
        capacity: int,
        *,
        missing_ttl_steps: Optional[int],
    ) -> None:
        self.capacity = int(capacity)
        self.missing_ttl_steps = missing_ttl_steps
        self._slots: list[Optional[ObjectKeyV2]] = [None] * self.capacity
        self._missed: Dict[ObjectKeyV2, int] = {}

    @property
    def keys(self) -> Tuple[Optional[ObjectKeyV2], ...]:
        return tuple(self._slots)

    @property
    def ids(self) -> Tuple[Optional[str], ...]:
        return tuple(
            None if key is None else f"{key[0]}:{key[1]}"
            for key in self._slots
        )

    def update(self, visible: Dict[ObjectKeyV2, VisibleRecordV2]) -> None:
        ranked = [
            key
            for key, _ in sorted(
                visible.items(),
                key=lambda item: (
                    item[1][3],
                    item[1][1],
                    item[1][2],
                    KIND_ORDER_V2.index(item[0][0]),
                    item[0][1],
                ),
            )[: self.capacity]
        ]
        selected = set(ranked)

        for slot, key in enumerate(self._slots):
            if key is None:
                continue
            if key in selected:
                self._missed.pop(key, None)
                continue
            if key in visible:
                # A visible but farther object must yield to a nearer hazard.
                self._slots[slot] = None
                self._missed.pop(key, None)
                continue
            if self.missing_ttl_steps is None:
                continue
            missed = self._missed.get(key, 0) + 1
            if missed > self.missing_ttl_steps:
                self._slots[slot] = None
                self._missed.pop(key, None)
            else:
                self._missed[key] = missed

        tracked = {key for key in self._slots if key is not None}
        for key in ranked:
            if key in tracked:
                continue
            try:
                free_slot = self._slots.index(None)
            except ValueError:
                # Missing objects may retain a slot for temporal continuity,
                # but a newly visible nearest hazard always has priority.
                missing_slots = [
                    index
                    for index, slot_key in enumerate(self._slots)
                    if slot_key not in visible
                ]
                if not missing_slots:
                    raise RuntimeError("Nearest-object selection has no free slot")
                free_slot = max(
                    missing_slots,
                    key=lambda index: self._missed.get(
                        self._slots[index] or ("traffic_cone", ""), 0
                    ),
                )
                old_key = self._slots[free_slot]
                if old_key is not None:
                    self._missed.pop(old_key, None)
            self._slots[free_slot] = key
            tracked.add(key)


class ObservationEncoderV2:
    """Build one finite, normalized observation per synchronous 10-Hz frame."""

    def __init__(self, spec: ObservationSpecV2 = ObservationSpecV2()) -> None:
        self.spec = spec
        start = 0
        current = slice(start, start + CURRENT_EGO_DIM_V2)
        start = current.stop
        history = slice(
            start,
            start + len(spec.history_lags) * EGO_HISTORY_FRAME_DIM_V2,
        )
        start = history.stop
        workzone = slice(
            start,
            start + spec.max_workzone_elements * spec.workzone_entity_dim,
        )
        start = workzone.stop
        agents = slice(
            start,
            start + spec.max_other_agents * spec.dynamic_entity_dim,
        )
        start = agents.stop
        lanes = slice(
            start,
            start + spec.max_lane_segments * spec.lane_segment_dim,
        )
        self.layout = ObservationLayoutV2(
            current_ego=current,
            ego_history=history,
            workzone_elements=workzone,
            other_agents=agents,
            lane_segments=lanes,
        )
        self._start_ego: Optional[EgoSampleV2] = None
        self._history: deque[_HistoryFrameV2] = deque(maxlen=spec.max_history_lag)
        self._workzone_slots = _StableSlotsV2(
            spec.max_workzone_elements,
            missing_ttl_steps=None,
        )
        self._agent_slots = _StableSlotsV2(
            spec.max_other_agents,
            missing_ttl_steps=spec.missing_actor_ttl_steps,
        )
        self._workzone_ids: Optional[_EpisodeIdentityCodecV2] = None
        self._agent_ids: Optional[_EpisodeIdentityCodecV2] = None

    def reset(self, initial_ego: EgoSampleV2, *, seed: Optional[int] = None) -> None:
        """Clear temporal state and create fresh episode-local ID salts."""
        _validate_ego_v2(initial_ego)
        seed_sequence = np.random.SeedSequence(seed)
        workzone_seed, agent_seed = seed_sequence.spawn(2)
        self._start_ego = initial_ego
        self._history.clear()
        self._workzone_slots = _StableSlotsV2(
            self.spec.max_workzone_elements,
            missing_ttl_steps=None,
        )
        self._agent_slots = _StableSlotsV2(
            self.spec.max_other_agents,
            missing_ttl_steps=self.spec.missing_actor_ttl_steps,
        )
        self._workzone_ids = _EpisodeIdentityCodecV2(
            self.spec.local_id_bits,
            np.random.default_rng(workzone_seed),
        )
        self._agent_ids = _EpisodeIdentityCodecV2(
            self.spec.local_id_bits,
            np.random.default_rng(agent_seed),
        )

    def encode(
        self,
        ego: EgoSampleV2,
        *,
        workzone_elements: Iterable[ObjectSampleV2] = (),
        other_agents: Iterable[ObjectSampleV2] = (),
        lane_segments: Iterable[LaneSegmentSampleV2] = (),
    ) -> EncodedObservationV2:
        """Encode one frame; call exactly once per synchronous environment step."""
        if (
            self._start_ego is None
            or self._workzone_ids is None
            or self._agent_ids is None
        ):
            raise RuntimeError("reset() must be called before encode()")
        _validate_ego_v2(ego)

        workzone_visible = self._visible_objects(
            ego,
            workzone_elements,
            allowed_kinds=WORKZONE_KINDS_V2,
        )
        agent_visible = self._visible_objects(
            ego,
            other_agents,
            allowed_kinds=OTHER_AGENT_KINDS_V2,
        )
        lane_visible = self._visible_lane_segments(ego, lane_segments)
        self._workzone_slots.update(workzone_visible)
        self._agent_slots.update(agent_visible)

        values = np.zeros(self.spec.observation_dim, dtype=np.float32)
        values[self.layout.current_ego] = self._encode_current_ego(ego)
        values[self.layout.ego_history] = self._encode_ego_history(ego)
        values[self.layout.workzone_elements] = self._encode_workzone_entities(
            self._workzone_slots.keys,
            workzone_visible,
            self._workzone_ids,
        )
        values[self.layout.other_agents] = self._encode_dynamic_entities(
            ego,
            self._agent_slots.keys,
            agent_visible,
            self._agent_ids,
        )
        lane_values, lane_ids = self._encode_lane_segments(ego, lane_visible)
        values[self.layout.lane_segments] = lane_values

        # Append only after encoding: lag 1 always means the previous call.
        self._history.append(
            _HistoryFrameV2(ego=ego, other_agents=dict(agent_visible))
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Observation encoder produced a non-finite value")
        np.clip(values, -1.0, 1.0, out=values)
        return EncodedObservationV2(
            values=values,
            workzone_slot_ids=self._workzone_slots.ids,
            other_agent_slot_ids=self._agent_slots.ids,
            lane_segment_ids=lane_ids,
        )

    def _encode_current_ego(self, ego: EgoSampleV2) -> np.ndarray:
        assert self._start_ego is not None
        dx = float(ego.x) - float(self._start_ego.x)
        dy = float(ego.y) - float(self._start_ego.y)
        local_x, local_y = _rotate_to_frame_v2(
            dx,
            dy,
            self._start_ego.heading_deg,
        )
        heading_delta = math.radians(
            _wrapped_heading_delta_v2(
                ego.heading_deg,
                self._start_ego.heading_deg,
            )
        )
        return np.asarray(
            [
                np.clip(local_x / self.spec.ego_position_scale_m, -1.0, 1.0),
                np.clip(local_y / self.spec.ego_position_scale_m, -1.0, 1.0),
                math.sin(heading_delta),
                math.cos(heading_delta),
                _normalize_speed_v2(ego.speed_mps, self.spec.max_ego_speed_mps),
            ],
            dtype=np.float32,
        )

    def _encode_ego_history(self, ego: EgoSampleV2) -> np.ndarray:
        encoded = np.zeros(
            len(self.spec.history_lags) * EGO_HISTORY_FRAME_DIM_V2,
            dtype=np.float32,
        )
        for index, lag in enumerate(self.spec.history_lags):
            frame = self._frame_at_lag(lag)
            if frame is None:
                continue
            sample = frame.ego
            dx = float(sample.x) - float(ego.x)
            dy = float(sample.y) - float(ego.y)
            rel_x, rel_y = _rotate_to_frame_v2(dx, dy, ego.heading_deg)
            heading_delta = math.radians(
                _wrapped_heading_delta_v2(sample.heading_deg, ego.heading_deg)
            )
            start = index * EGO_HISTORY_FRAME_DIM_V2
            encoded[start : start + EGO_HISTORY_FRAME_DIM_V2] = (
                1.0,
                float(
                    np.clip(
                        rel_x / self.spec.history_distance_scale_m,
                        -1.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        rel_y / self.spec.history_distance_scale_m,
                        -1.0,
                        1.0,
                    )
                ),
                math.sin(heading_delta),
                math.cos(heading_delta),
                _normalize_speed_v2(
                    sample.speed_mps,
                    self.spec.max_ego_speed_mps,
                ),
            )
        return encoded

    def _visible_objects(
        self,
        ego: EgoSampleV2,
        objects: Iterable[ObjectSampleV2],
        *,
        allowed_kinds: frozenset[str],
    ) -> Dict[ObjectKeyV2, VisibleRecordV2]:
        visible: Dict[ObjectKeyV2, VisibleRecordV2] = {}
        radius_sq = self.spec.perception_radius_m ** 2
        for item in objects:
            if item.kind not in allowed_kinds:
                raise ValueError(
                    f"{item.kind!r} is not valid in this observation group"
                )
            raw_id = str(item.object_id)
            if not raw_id:
                raise ValueError("Object IDs must be non-empty")
            if not all(math.isfinite(float(value)) for value in (item.x, item.y)):
                continue
            dx = float(item.x) - float(ego.x)
            dy = float(item.y) - float(ego.y)
            distance_sq = dx * dx + dy * dy
            if distance_sq > radius_sq:
                continue
            rel_x, rel_y = _rotate_to_frame_v2(dx, dy, ego.heading_deg)
            key: ObjectKeyV2 = (item.kind, raw_id)
            if key in visible:
                raise ValueError(
                    f"Duplicate object identity in one frame: {item.kind}:{raw_id}"
                )
            visible[key] = (item, rel_x, rel_y, distance_sq)
        return visible

    def _visible_lane_segments(
        self,
        ego: EgoSampleV2,
        segments: Iterable[LaneSegmentSampleV2],
    ) -> Tuple[VisibleLaneRecordV2, ...]:
        visible: list[VisibleLaneRecordV2] = []
        seen_ids: set[str] = set()
        radius_sq = self.spec.perception_radius_m ** 2
        for segment in segments:
            segment_id = str(segment.segment_id)
            if not segment_id:
                raise ValueError("Lane segment IDs must be non-empty")
            if segment_id in seen_ids:
                raise ValueError(f"Duplicate lane segment identity: {segment_id}")
            seen_ids.add(segment_id)
            numeric = (
                segment.x,
                segment.y,
                segment.axis_heading_deg,
                segment.length_m,
                segment.width_m,
            )
            if not all(math.isfinite(float(value)) for value in numeric):
                continue
            if segment.length_m <= 0.0 or segment.width_m <= 0.0:
                continue
            distance_sq = _point_to_lane_segment_distance_sq_v2(
                float(ego.x),
                float(ego.y),
                segment,
            )
            if distance_sq > radius_sq:
                continue
            dx = float(segment.x) - float(ego.x)
            dy = float(segment.y) - float(ego.y)
            rel_x, rel_y = _rotate_to_frame_v2(dx, dy, ego.heading_deg)
            visible.append((segment, rel_x, rel_y, distance_sq))
        visible.sort(
            key=lambda item: (item[3], item[1], item[2], item[0].segment_id)
        )
        return tuple(visible[: self.spec.max_lane_segments])

    def _encode_lane_segments(
        self,
        ego: EgoSampleV2,
        visible: Sequence[VisibleLaneRecordV2],
    ) -> Tuple[np.ndarray, Tuple[Optional[str], ...]]:
        encoded = np.zeros(
            self.spec.max_lane_segments * self.spec.lane_segment_dim,
            dtype=np.float32,
        )
        ids: list[Optional[str]] = [None] * self.spec.max_lane_segments
        for slot, (segment, rel_x, rel_y, _) in enumerate(visible):
            start = slot * self.spec.lane_segment_dim
            relative_axis = math.radians(
                _wrapped_heading_delta_v2(
                    segment.axis_heading_deg,
                    ego.heading_deg,
                )
            )
            encoded[start : start + self.spec.lane_segment_dim] = (
                1.0,
                _normalize_distance_v2(rel_x, self.spec.perception_radius_m),
                _normalize_distance_v2(rel_y, self.spec.perception_radius_m),
                float(np.clip(segment.length_m / self.spec.lane_length_scale_m, 0.0, 1.0)),
                float(np.clip(segment.width_m / self.spec.lane_width_scale_m, 0.0, 1.0)),
                math.cos(2.0 * relative_axis),
                math.sin(2.0 * relative_axis),
            )
            ids[slot] = str(segment.segment_id)
        return encoded, tuple(ids)

    def _encode_workzone_entities(
        self,
        slot_keys: Sequence[Optional[ObjectKeyV2]],
        visible: Dict[ObjectKeyV2, VisibleRecordV2],
        identity_codec: _EpisodeIdentityCodecV2,
    ) -> np.ndarray:
        encoded = np.zeros(
            len(slot_keys) * self.spec.workzone_entity_dim,
            dtype=np.float32,
        )
        for slot, key in enumerate(slot_keys):
            if key is None or key not in visible:
                continue
            _, rel_x, rel_y, _ = visible[key]
            start = slot * self.spec.workzone_entity_dim
            encoded[start] = 1.0
            encoded[
                start + 1 + WORKZONE_KIND_ORDER_V2.index(key[0])
            ] = 1.0
            id_start = start + 1 + GROUP_KIND_DIM_V2
            id_end = id_start + self.spec.local_id_bits
            encoded[id_start:id_end] = identity_codec.encode(key)
            encoded[id_end] = _normalize_distance_v2(
                rel_x,
                self.spec.perception_radius_m,
            )
            encoded[id_end + 1] = _normalize_distance_v2(
                rel_y,
                self.spec.perception_radius_m,
            )
        return encoded

    def _encode_dynamic_entities(
        self,
        ego: EgoSampleV2,
        slot_keys: Sequence[Optional[ObjectKeyV2]],
        visible: Dict[ObjectKeyV2, VisibleRecordV2],
        identity_codec: _EpisodeIdentityCodecV2,
    ) -> np.ndarray:
        encoded = np.zeros(
            len(slot_keys) * self.spec.dynamic_entity_dim,
            dtype=np.float32,
        )
        for slot, key in enumerate(slot_keys):
            if key is None:
                continue
            start = slot * self.spec.dynamic_entity_dim
            encoded[start + 1 + OTHER_AGENT_KIND_ORDER_V2.index(key[0])] = 1.0
            id_start = start + 1 + GROUP_KIND_DIM_V2
            id_end = id_start + self.spec.local_id_bits
            encoded[id_start:id_end] = identity_codec.encode(key)
            current = visible.get(key)
            if current is not None:
                _, rel_x, rel_y, _ = current
                encoded[start] = 1.0
                encoded[id_end] = _normalize_distance_v2(
                    rel_x,
                    self.spec.perception_radius_m,
                )
                encoded[id_end + 1] = _normalize_distance_v2(
                    rel_y,
                    self.spec.perception_radius_m,
                )

            history_start = id_end + 2
            for index, lag in enumerate(self.spec.history_lags):
                frame = self._frame_at_lag(lag)
                if frame is None:
                    continue
                historical = frame.other_agents.get(key)
                if historical is None:
                    continue
                item = historical[0]
                dx = float(item.x) - float(ego.x)
                dy = float(item.y) - float(ego.y)
                rel_x, rel_y = _rotate_to_frame_v2(dx, dy, ego.heading_deg)
                lag_start = history_start + index * DYNAMIC_HISTORY_FRAME_DIM_V2
                encoded[lag_start] = 1.0
                encoded[lag_start + 1] = _normalize_distance_v2(
                    rel_x,
                    self.spec.perception_radius_m,
                )
                encoded[lag_start + 2] = _normalize_distance_v2(
                    rel_y,
                    self.spec.perception_radius_m,
                )
        return encoded

    def _frame_at_lag(self, lag: int) -> Optional[_HistoryFrameV2]:
        return self._history[-lag] if len(self._history) >= lag else None


DEFAULT_OBSERVATION_SPEC_V2 = ObservationSpecV2()
DEFAULT_OBSERVATION_DIM_V2 = DEFAULT_OBSERVATION_SPEC_V2.observation_dim


def _validate_ego_v2(ego: EgoSampleV2) -> None:
    values = (ego.x, ego.y, ego.heading_deg, ego.speed_mps)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Ego observation values must be finite")
    if ego.speed_mps < 0.0:
        raise ValueError("ego speed must be non-negative")


def _normalize_speed_v2(speed_mps: float, maximum_mps: float) -> float:
    return float(2.0 * np.clip(float(speed_mps) / maximum_mps, 0.0, 1.0) - 1.0)


def _normalize_distance_v2(distance_m: float, scale_m: float) -> float:
    return float(np.clip(float(distance_m) / scale_m, -1.0, 1.0))


def _wrapped_heading_delta_v2(heading_deg: float, reference_deg: float) -> float:
    return (float(heading_deg) - float(reference_deg) + 180.0) % 360.0 - 180.0


def _rotate_to_frame_v2(
    world_dx: float,
    world_dy: float,
    frame_heading_deg: float,
) -> Tuple[float, float]:
    yaw = math.radians(float(frame_heading_deg))
    c, s = math.cos(yaw), math.sin(yaw)
    return (
        c * float(world_dx) + s * float(world_dy),
        -s * float(world_dx) + c * float(world_dy),
    )


def _point_to_lane_segment_distance_sq_v2(
    point_x: float,
    point_y: float,
    segment: LaneSegmentSampleV2,
) -> float:
    yaw = math.radians(float(segment.axis_heading_deg))
    half_length = 0.5 * float(segment.length_m)
    axis_x = math.cos(yaw)
    axis_y = math.sin(yaw)
    first_x = float(segment.x) - half_length * axis_x
    first_y = float(segment.y) - half_length * axis_y
    second_x = float(segment.x) + half_length * axis_x
    second_y = float(segment.y) + half_length * axis_y
    edge_x = second_x - first_x
    edge_y = second_y - first_y
    length_sq = edge_x * edge_x + edge_y * edge_y
    if length_sq <= 1e-12:
        return (float(point_x) - first_x) ** 2 + (float(point_y) - first_y) ** 2
    projection = (
        (float(point_x) - first_x) * edge_x
        + (float(point_y) - first_y) * edge_y
    ) / length_sq
    projection = float(np.clip(projection, 0.0, 1.0))
    closest_x = first_x + projection * edge_x
    closest_y = first_y + projection * edge_y
    return (float(point_x) - closest_x) ** 2 + (float(point_y) - closest_y) ** 2


__all__ = [
    "DEFAULT_OBSERVATION_DIM_V2",
    "DEFAULT_OBSERVATION_SPEC_V2",
    "EncodedObservationV2",
    "EgoSampleV2",
    "KIND_ORDER_V2",
    "LANE_SEGMENT_DIM_V2",
    "LaneSegmentSampleV2",
    "OBSERVATION_CONTRACT_VERSION_V2",
    "ObjectSampleV2",
    "ObservationEncoderV2",
    "ObservationLayoutV2",
    "ObservationSpecV2",
]
