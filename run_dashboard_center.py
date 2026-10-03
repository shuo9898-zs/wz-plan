"""Dashboard launcher for the centre-ego control experiment."""
from __future__ import annotations

import sys

from standalone_launcher import ROOT, load_module


if "--log-dir" not in sys.argv:
    sys.argv.extend(("--log-dir", str(ROOT / "runs_center_ego" / "logs")))
if "--total-updates" not in sys.argv:
    sys.argv.extend(("--total-updates", "35"))

main = load_module("dashboard").main


if __name__ == "__main__":
    raise SystemExit(main())
