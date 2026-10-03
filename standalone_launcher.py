"""Load this directory as ``Aug24_ppo`` even when the folder was renamed."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from bootstrap import activate_portable_runtime


ROOT = Path(__file__).resolve().parent


def load_module(module_name: str) -> ModuleType:
    activate_portable_runtime()
    if "Aug24_ppo" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "Aug24_ppo",
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load portable package from {ROOT}")
        package = importlib.util.module_from_spec(spec)
        sys.modules["Aug24_ppo"] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"Aug24_ppo.{module_name}")


__all__ = ["ROOT", "load_module"]
