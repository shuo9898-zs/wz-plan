"""Launch the existing localhost dashboard for the Aug-24 run."""
from __future__ import annotations

import sys

from tools.live_training_web_v2 import main

from .experiment_config import DEFAULT_RUN_ROOT, STEPS_PER_UPDATE, TOTAL_UPDATES


if __name__ == "__main__":
    if "--log-dir" not in sys.argv:
        sys.argv.extend(("--log-dir", str(DEFAULT_RUN_ROOT / "logs")))
    if "--total-updates" not in sys.argv:
        sys.argv.extend(("--total-updates", str(TOTAL_UPDATES)))
    if "--steps-per-update" not in sys.argv:
        sys.argv.extend(("--steps-per-update", str(STEPS_PER_UPDATE)))
    raise SystemExit(main())
