"""Run validation S1 Normal Traffic (legacy/runtime folder identity S5)."""
from __future__ import annotations

from validation_debug.runner_v2 import main_for_validation_scenario_v2


def main(argv: list[str] | None = None) -> int:
    return main_for_validation_scenario_v2("s1", argv)


if __name__ == "__main__":
    raise SystemExit(main())
