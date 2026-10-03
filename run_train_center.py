"""Path-independent launcher for the centre-ego control experiment."""
from __future__ import annotations

from standalone_launcher import load_module


main = load_module("train_center").main


if __name__ == "__main__":
    raise SystemExit(main())
