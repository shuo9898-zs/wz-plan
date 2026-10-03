"""Run the S4 jaywalker validation debug."""
from __future__ import annotations

from validation_debug.runner_v2 import main_for_validation_scenario_v2


def main(argv: list[str] | None = None) -> int:
    return main_for_validation_scenario_v2("s4", argv)


if __name__ == "__main__":
    raise SystemExit(main())
