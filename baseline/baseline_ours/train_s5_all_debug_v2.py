"""Visible 10k-step debug over every S5 WZ/layout/origin ticket."""
from baseline.baseline_ours.train_scenario_all_debug_v2 import (
    build_parser_for_scenario_v2,
    main_for_scenario_v2,
    scenario_settings_v2,
    scenario_ticket_ids_v2,
)

SCENARIO_ID = "s5"
SETTINGS_V2 = scenario_settings_v2(SCENARIO_ID)
TICKET_IDS_V2 = scenario_ticket_ids_v2(SCENARIO_ID)


def build_parser():
    return build_parser_for_scenario_v2(SCENARIO_ID)


def main(argv=None) -> int:
    return main_for_scenario_v2(SCENARIO_ID, argv)


if __name__ == "__main__":
    raise SystemExit(main())
