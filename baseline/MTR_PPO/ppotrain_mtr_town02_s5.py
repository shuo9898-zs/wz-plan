"""CMD-friendly entry point for the Town02/S5 MTR-PPO debug run."""
from __future__ import annotations

import sys
from pathlib import Path


_BUNDLE_ROOT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_ROOT))

from baseline.MTR_PPO.train_town02_s5_debug import main


if __name__ == "__main__":
    raise SystemExit(main())
