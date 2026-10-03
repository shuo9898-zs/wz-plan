"""Path-independent dashboard launcher."""
from __future__ import annotations

import sys

from standalone_launcher import ROOT, load_module


experiment_config = load_module("experiment_config")
if "--log-dir" not in sys.argv:
    sys.argv.extend(("--log-dir", str(ROOT / "runs" / "logs")))
if "--total-updates" not in sys.argv:
    sys.argv.extend(("--total-updates", str(experiment_config.TOTAL_UPDATES)))
if "--steps-per-update" not in sys.argv:
    sys.argv.extend((
        "--steps-per-update",
        str(experiment_config.STEPS_PER_UPDATE),
    ))

main = load_module("dashboard").main


if __name__ == "__main__":
    raise SystemExit(main())
