"""Path-independent launcher: ``python run_train.py [arguments]``."""
from __future__ import annotations

from standalone_launcher import load_module


main = load_module("train").main


if __name__ == "__main__":
    raise SystemExit(main())
