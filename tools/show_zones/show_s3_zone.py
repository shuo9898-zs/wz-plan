"""Show every S3 green drivable union for every setting and origin."""
from tools.show_zones._zone_carousel_v2 import main_for_scenario_v2

SCENARIO_ID = "s3"


def main(argv=None) -> int:
    return main_for_scenario_v2(SCENARIO_ID, argv)


if __name__ == "__main__":
    raise SystemExit(main())

