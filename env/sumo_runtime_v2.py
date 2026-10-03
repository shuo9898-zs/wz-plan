"""Pinned SUMO runtime discovery for the PPO V2 environment.

The machine-wide SUMO installation is intentionally not changed.  PPO V2
uses the version installed in this project's virtual environment so the SUMO
server and Python TraCI client come from one release.
"""
from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MIN_SUMO_VERSION_V2 = (1, 27, 1)


@dataclass(frozen=True)
class SumoRuntimeV2:
    home: Path
    tools_dir: Path
    binary_dir: Path
    sumo_binary: Path
    sumo_gui_binary: Path


def resolve_sumo_runtime_v2() -> SumoRuntimeV2:
    try:
        import sumo
    except ImportError as exc:
        raise RuntimeError(
            "PPO V2 requires eclipse-sumo==1.27.1 in the project virtual "
            "environment; run `python -m pip install -r requirements_v2.txt`."
        ) from exc

    home = Path(sumo.SUMO_HOME).resolve()
    runtime = SumoRuntimeV2(
        home=home,
        tools_dir=home / "tools",
        binary_dir=home / "bin",
        sumo_binary=home / "bin" / "sumo.exe",
        sumo_gui_binary=home / "bin" / "sumo-gui.exe",
    )
    missing = [
        path
        for path in (
            runtime.tools_dir,
            runtime.sumo_binary,
            runtime.sumo_gui_binary,
        )
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "Incomplete PPO V2 SUMO installation; missing: "
            + ", ".join(str(path) for path in missing)
        )
    return runtime


SUMO_RUNTIME_V2 = resolve_sumo_runtime_v2()


def prefer_sumo_python_tools_v2() -> None:
    """Put the pinned TraCI implementation ahead of machine-wide tools.

    This function must run before importing the shared simulator engine.
    Existing imports are not replaced because doing so would leave partially
    rebound TraCI domain objects in the process.
    """
    tools = str(SUMO_RUNTIME_V2.tools_dir)
    if "traci" in sys.modules:
        return
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.normcase(entry) != os.path.normcase(tools)
    ]
    sys.path.insert(0, tools)


@contextmanager
def pinned_sumo_binary_path_v2() -> Iterator[None]:
    """Temporarily make the pinned SUMO binaries first on ``PATH``."""
    previous = os.environ.get("PATH")
    prefix = str(SUMO_RUNTIME_V2.binary_dir)
    os.environ["PATH"] = prefix if not previous else prefix + os.pathsep + previous
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous


def parse_sumo_server_version_v2(description: str) -> tuple[int, int, int]:
    match = re.search(r"\bSUMO\s+(\d+)\.(\d+)\.(\d+)\b", description)
    if match is None:
        raise ValueError(f"Unrecognized SUMO server version: {description!r}")
    return tuple(int(part) for part in match.groups())


__all__ = [
    "MIN_SUMO_VERSION_V2",
    "SUMO_RUNTIME_V2",
    "SumoRuntimeV2",
    "parse_sumo_server_version_v2",
    "pinned_sumo_binary_path_v2",
    "prefer_sumo_python_tools_v2",
    "resolve_sumo_runtime_v2",
]
