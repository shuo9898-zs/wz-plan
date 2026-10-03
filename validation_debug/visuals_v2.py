"""Validation-native CARLA props and finite-lived work-zone laser overlays."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym

from baseline.PPO.train_s3_wz1_origin2_debug_v2 import (
    OVERLAY_CLOSE_TIMEOUT_S_V2,
    OVERLAY_LIFETIME_S_V2,
    OVERLAY_REFRESH_PERIOD_S_V2,
    _wait_for_debug_overlay_expiry_v2,
)
from tools.show_traffic_cones import (
    CONE_BLUEPRINT_CANDIDATES,
    SIGN_BLUEPRINT_CANDIDATES,
    _choose_blueprint,
    _surface_transform,
    _visual_prop_blueprint_ids,
)
from tools.show_workzone_polygons_v2 import (
    FINISH_YELLOW_V2,
    ORIGIN_CYAN_V2,
    VisualizationGeometryV2,
    build_visualization_geometry_v2,
    hatch_segments_v2,
)
from validation_debug.loader_v2 import ValidationCaseV2


class ValidationWorkZonePropsV2:
    """Spawn and own props from an already-materialized validation case."""

    def __init__(
        self,
        case: ValidationCaseV2,
        *,
        enabled: bool = True,
        cones_only: bool = False,
        z_offset: float = 0.08,
        sign_yaw_offset: float = 180.0,
    ) -> None:
        self.case = case
        self.enabled = bool(enabled)
        self.cones_only = bool(cones_only)
        self.z_offset = float(z_offset)
        self.sign_yaw_offset = float(sign_yaw_offset)
        self.world: Any | None = None
        self.actors: list[Any] = []

    def start(self, world: Any) -> None:
        if not self.enabled:
            return
        live = []
        for actor in self.actors:
            try:
                if actor.is_alive:
                    live.append(actor)
            except RuntimeError:
                pass
        self.actors = live
        if self.actors and self.world is world:
            return
        if self.actors:
            self.close()
        self.world = world

        cfg = self.case.config
        library = world.get_blueprint_library()
        cone_bp = _choose_blueprint(
            library, CONE_BLUEPRINT_CANDIDATES, ("*constructioncone*",)
        )
        if cone_bp is None and cfg.workzone.traffic_cones:
            raise RuntimeError("No CARLA construction-cone blueprint was found")
        carla_map = world.get_map()
        for index, (x, y) in enumerate(cfg.workzone.traffic_cones, start=1):
            transform = _surface_transform(
                carla_map,
                float(x),
                float(y),
                float(cfg.carla.road_heading_deg),
                self.z_offset,
            )
            self._spawn(cone_bp, transform, f"cone {index}")

        if not self.cones_only:
            visual_debug = self.case.materialized_data.get("visual_debug") or {}
            visual_props = tuple(
                dict(prop) for prop in visual_debug.get("props", ())
            )
            if visual_props:
                for index, prop in enumerate(visual_props, start=1):
                    hint = str(prop.get("asset_hint_supplied", ""))
                    prop_bp = _choose_blueprint(
                        library, _visual_prop_blueprint_ids(hint), ()
                    )
                    if prop_bp is None:
                        print(
                            f"WARN no CARLA blueprint for validation prop "
                            f"{index}: {hint or 'missing asset hint'}",
                            flush=True,
                        )
                        continue
                    location = prop.get("location") or ()
                    if len(location) < 2:
                        print(
                            f"WARN validation prop {index} has no XY location",
                            flush=True,
                        )
                        continue
                    rotation = prop.get("rotation_deg") or ()
                    pitch = float(rotation[0]) if len(rotation) > 0 else 0.0
                    roll = float(rotation[1]) if len(rotation) > 1 else 0.0
                    yaw = (
                        float(rotation[2])
                        if len(rotation) > 2
                        else float(cfg.carla.road_heading_deg)
                    )
                    authored_z = float(location[2]) if len(location) > 2 else 0.0
                    transform = _surface_transform(
                        carla_map,
                        float(location[0]),
                        float(location[1]),
                        yaw,
                        self.z_offset + authored_z,
                        pitch=pitch,
                        roll=roll,
                    )
                    self._spawn(prop_bp, transform, f"visual prop {index}")
            else:
                sign_bp = _choose_blueprint(
                    library,
                    SIGN_BLUEPRINT_CANDIDATES,
                    ("*warning*", "*construction*"),
                )
                if sign_bp is None and cfg.workzone.warning_signs:
                    print("WARN no CARLA work-zone sign blueprint was found", flush=True)
                elif sign_bp is not None:
                    sign_yaw = (
                        float(cfg.carla.road_heading_deg) + self.sign_yaw_offset
                    )
                    for index, (x, y) in enumerate(
                        cfg.workzone.warning_signs, start=1
                    ):
                        transform = _surface_transform(
                            carla_map,
                            float(x),
                            float(y),
                            sign_yaw,
                            self.z_offset,
                        )
                        self._spawn(sign_bp, transform, f"warning sign {index}")

        print(
            f"validation_props spawned={len(self.actors)} "
            f"setting={cfg.setting_id}",
            flush=True,
        )

    def _spawn(self, blueprint: Any, transform: Any, label: str) -> None:
        if blueprint is None or self.world is None:
            return
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "debug_workzone_prop")
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            print(
                f"WARN could not spawn validation {label} at "
                f"({transform.location.x:.2f}, {transform.location.y:.2f})",
                flush=True,
            )
            return
        try:
            actor.set_simulate_physics(False)
            actor.set_enable_gravity(False)
        except (AttributeError, RuntimeError):
            pass
        self.actors.append(actor)

    def close(self) -> None:
        destroyed = 0
        total = len(self.actors)
        for actor in reversed(self.actors):
            try:
                if actor.is_alive and actor.destroy():
                    destroyed += 1
            except RuntimeError:
                pass
        self.actors.clear()
        self.world = None
        if total:
            print(f"validation_props destroyed={destroyed}/{total}", flush=True)


class ValidationDebugVisualWrapperV2(gym.Wrapper):
    """Draw exact shared geometry without ever moving the spectator camera."""

    def __init__(
        self,
        env: gym.Env,
        *,
        case: ValidationCaseV2,
        selector: Any,
        show_props: bool,
        cones_only: bool,
    ) -> None:
        super().__init__(env)
        self.case = case
        self._selector = selector
        self._active_origin_index = 0
        self.props = ValidationWorkZonePropsV2(
            case, enabled=show_props, cones_only=cones_only
        )
        control_dt_s = float(self.env.engine.cfg.episode.sim_dt)
        self._overlay_lifetime_s = OVERLAY_LIFETIME_S_V2
        self._overlay_refresh_interval_steps = max(
            1,
            int(round(OVERLAY_REFRESH_PERIOD_S_V2 / control_dt_s)),
        )
        self._overlay_refresh_countdown = 0
        self._overlay_key: tuple[Any, ...] | None = None
        self._geometry_cache: (
            tuple[tuple[Any, ...], VisualizationGeometryV2] | None
        ) = None

    @property
    def active_origin_index(self) -> int:
        return self._active_origin_index

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._active_origin_index = int(
            self._selector.current_ticket.origin_index
        )
        world = self.env.engine.world
        if world is None:
            raise RuntimeError("validation reset completed without a CARLA world")
        self.props.start(world)
        self._overlay_refresh_countdown = 0
        self._draw_overlay_if_needed()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._draw_overlay_if_needed()
        return result

    def close(self) -> None:
        world = getattr(self.env.engine, "world", None)
        overlay_was_drawn = self._overlay_key is not None
        try:
            self.props.close()
        finally:
            try:
                super().close()
            finally:
                if overlay_was_drawn and world is not None:
                    _wait_for_debug_overlay_expiry_v2(
                        world,
                        life_time_s=self._overlay_lifetime_s,
                        wall_timeout_s=OVERLAY_CLOSE_TIMEOUT_S_V2,
                    )

    def _geometry(self, world: Any) -> VisualizationGeometryV2:
        cfg = self.case.config
        spawn = cfg.origin.spawn_points[0]
        cache_key = (
            id(world),
            self._active_origin_index,
            float(spawn.x),
            float(spawn.y),
            float(spawn.yaw_deg),
        )
        if self._geometry_cache is not None and self._geometry_cache[0] == cache_key:
            return self._geometry_cache[1]
        geometry = build_visualization_geometry_v2(
            cfg,
            origin_index=0 if cfg.scenario_id == "s3" else None,
            carla_map=world.get_map(),
        )
        if geometry.origin is None:
            geometry = replace(
                geometry,
                origin=(float(spawn.x), float(spawn.y)),
            )
        self._geometry_cache = (cache_key, geometry)
        return geometry

    def _draw_overlay_if_needed(self) -> None:
        try:
            import carla

            world = self.env.engine.world
            if world is None:
                return
            geometry = self._geometry(world)
            key = (
                id(world),
                tuple(
                    (layer.label, layer.vertices, layer.holes, layer.color_rgb)
                    for layer in geometry.layers
                ),
                geometry.finish_line.first,
                geometry.finish_line.second,
                geometry.origin,
            )
            if self._overlay_key == key and self._overlay_refresh_countdown > 0:
                self._overlay_refresh_countdown -= 1
                return

            carla_map = world.get_map()

            def location(point: tuple[float, float], z_offset: float = 0.18):
                query = carla.Location(
                    x=float(point[0]), y=float(point[1]), z=1.0
                )
                try:
                    waypoint = carla_map.get_waypoint(
                        query,
                        project_to_road=True,
                        lane_type=carla.LaneType.Any,
                    )
                except TypeError:
                    waypoint = carla_map.get_waypoint(
                        query, project_to_road=True
                    )
                ground_z = (
                    waypoint.transform.location.z
                    if waypoint is not None
                    else 0.0
                )
                return carla.Location(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=float(ground_z + z_offset),
                )

            for layer in geometry.layers:
                color = carla.Color(*layer.color_rgb)
                for start, end in zip(
                    layer.vertices, layer.vertices[1:] + layer.vertices[:1]
                ):
                    # Ground outline plus a raised, thin outline makes the
                    # exact same polygon visible as a laser fence.  These are
                    # debug primitives only; neither outline has collision.
                    world.debug.draw_line(
                        location(start, 0.20),
                        location(end, 0.20),
                        thickness=0.16,
                        color=color,
                        life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
                    world.debug.draw_line(
                        location(start, 0.82),
                        location(end, 0.82),
                        thickness=0.07,
                        color=color,
                        life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
                    world.debug.draw_line(
                        location(start, 0.20),
                        location(start, 0.82),
                        thickness=0.055,
                        color=color,
                        life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
                spacing = 2.0 if geometry.scenario_id == "s3" else 0.75
                for start, end in hatch_segments_v2(
                    layer.vertices,
                    spacing,
                    geometry.scenario_id != "s3",
                    holes=layer.holes,
                ):
                    world.debug.draw_line(
                        location(start, 0.21),
                        location(end, 0.21),
                        thickness=0.045,
                        color=color,
                        life_time=self._overlay_lifetime_s,
                        persistent_lines=False,
                    )
                center = (
                    sum(point[0] for point in layer.vertices) / len(layer.vertices),
                    sum(point[1] for point in layer.vertices) / len(layer.vertices),
                )
                meaning = (
                    "GREEN = DRIVABLE AREA"
                    if geometry.scenario_id == "s3"
                    else "RED = FORBIDDEN / VIOLATION"
                )
                world.debug.draw_string(
                    location(center, 1.05),
                    meaning,
                    draw_shadow=True,
                    color=color,
                    life_time=self._overlay_lifetime_s,
                    persistent_lines=False,
                )

            yellow = carla.Color(*FINISH_YELLOW_V2)
            world.debug.draw_line(
                location(geometry.finish_line.first, 0.25),
                location(geometry.finish_line.second, 0.25),
                thickness=0.20,
                color=yellow,
                life_time=self._overlay_lifetime_s,
                persistent_lines=False,
            )
            if geometry.origin is not None:
                cyan = carla.Color(*ORIGIN_CYAN_V2)
                world.debug.draw_point(
                    location(geometry.origin, 0.32),
                    size=0.24,
                    color=cyan,
                    life_time=self._overlay_lifetime_s,
                    persistent_lines=False,
                )
                world.debug.draw_string(
                    location(geometry.origin, 0.60),
                    f"{self.case.spec.scenario_id.upper()} VALIDATION "
                    f"ORIGIN {self._active_origin_index}",
                    draw_shadow=True,
                    color=cyan,
                    life_time=self._overlay_lifetime_s,
                    persistent_lines=False,
                )

            self._overlay_key = key
            self._overlay_refresh_countdown = (
                self._overlay_refresh_interval_steps - 1
            )
        except Exception as error:
            print(f"WARN validation laser overlay failed: {error}", flush=True)


__all__ = [
    "ValidationDebugVisualWrapperV2",
    "ValidationWorkZonePropsV2",
]
