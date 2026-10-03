"""Bootstrap helpers that keep all project imports inside this bundle."""
from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent


def _sumo_tools_path() -> Path | None:
    """Find TraCI tools from SUMO_HOME or the locked eclipse-sumo wheel."""
    configured = os.environ.get("SUMO_HOME")
    if configured:
        tools = Path(configured).expanduser() / "tools"
        if tools.is_dir():
            return tools.resolve()

    spec = importlib.util.find_spec("sumo")
    if spec is None or spec.origin is None:
        return None
    sumo_root = Path(spec.origin).resolve().parent
    tools = sumo_root / "tools"
    if not tools.is_dir():
        return None
    os.environ["SUMO_HOME"] = str(sumo_root)
    return tools


def activate_portable_runtime() -> Path:
    """Prepend the bundle to this process and all child Python processes."""
    root = str(BUNDLE_ROOT)
    sumo_tools = _sumo_tools_path()
    prepend = [root]
    if sumo_tools is not None:
        prepend.append(str(sumo_tools))
    sys.path[:] = [item for item in sys.path if item not in prepend]
    sys.path[:0] = prepend

    inherited = os.environ.get("PYTHONPATH", "")
    parts = [
        item
        for item in inherited.split(os.pathsep)
        if item and item not in prepend
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join([*prepend, *parts])
    os.environ["AUG24_PPO_BUNDLE_ROOT"] = root
    return BUNDLE_ROOT


def portable_subprocess_environment() -> dict[str, str]:
    activate_portable_runtime()
    return dict(os.environ)


__all__ = ["BUNDLE_ROOT", "activate_portable_runtime", "portable_subprocess_environment"]
