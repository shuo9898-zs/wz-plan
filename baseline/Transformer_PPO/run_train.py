"""Direct launcher for ``baseline.Transformer_PPO.train``."""
from __future__ import annotations

import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from bootstrap import activate_portable_runtime  # noqa: E402

activate_portable_runtime()

from baseline.Transformer_PPO.train import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
