"""Offline portability checks for the self-contained Aug24 PPO bundle."""
from __future__ import annotations

import importlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from bootstrap import BUNDLE_ROOT, activate_portable_runtime


REQUIRED_TREES = (
    "baseline", "config", "env", "logic", "models", "monitoring", "sync", "tools",
    "scenarios", "validation", "validation_debug",
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    activate_portable_runtime()
    errors: list[str] = []
    for name in REQUIRED_TREES:
        if not (BUNDLE_ROOT / name).is_dir():
            errors.append(f"missing bundled tree: {name}")

    for name in ("numpy", "gymnasium", "stable_baselines3", "torch", "shapely", "traci", "carla"):
        try:
            module = importlib.import_module(name)
            print(f"EXTERNAL {name}={getattr(module, '__version__', 'installed')}")
        except Exception as error:
            errors.append(f"cannot import {name}: {error}")

    for name in ("baseline.PPO.train_three_servers_v2", "config.scenario_catalog", "env.carla_sumo_env_v2", "validation_debug.runner_v2"):
        try:
            module = importlib.import_module(name)
            origin = Path(module.__file__).resolve()
            print(f"BUNDLED {name}={origin}")
            if not _inside(origin, BUNDLE_ROOT):
                errors.append(f"external source leak: {name} -> {origin}")
        except Exception as error:
            errors.append(f"cannot import bundled module {name}: {error}")

    try:
        from config.scenario_catalog import list_setting_ids, resolve_setting, scenario_ids
        settings = [setting for sid in scenario_ids() for setting in list_setting_ids(sid)]
        if len(settings) != 54:
            errors.append(f"expected 54 training settings, got {len(settings)}")
        for setting in settings:
            record, _ = resolve_setting(setting)
            for path in (record.config_path, record.network_path, record.route_path):
                if path is not None and not _inside(path, BUNDLE_ROOT):
                    errors.append(f"training path leak: {setting} -> {path}")
    except Exception as error:
        errors.append(f"training catalog check failed: {error}")

    try:
        from validation_debug.loader_v2 import load_validation_case_v2
        for sid in ("s1", "s2", "s3", "s4", "s5", "s6"):
            case = load_validation_case_v2(sid)
            if not _inside(case.config_path, BUNDLE_ROOT):
                errors.append(f"validation config leak: {sid} -> {case.config_path}")
            if case.config.sumo is not None:
                route = BUNDLE_ROOT / case.config.sumo.route_file
                routes = {node.attrib["id"] for node in ET.parse(route).getroot().findall("route")}
                missing = set(case.config.sumo.bg_routes) - routes
                if missing:
                    errors.append(f"validation {sid} missing routes: {sorted(missing)}")
    except Exception as error:
        errors.append(f"validation catalog check failed: {error}")

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"ok": True, "bundle_root": str(BUNDLE_ROOT), "python": sys.version.split()[0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
