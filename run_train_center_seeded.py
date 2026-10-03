"""Run the frozen Old-OD center-ego experiment with an explicit seed."""
from __future__ import annotations

import argparse
import sys

from standalone_launcher import load_module


def suppress_native_crash_dialogs() -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


def main(argv: list[str] | None = None) -> int:
    seed_parser = argparse.ArgumentParser(add_help=False)
    seed_parser.add_argument("--seed", type=int, required=True)
    seed_args, remaining = seed_parser.parse_known_args(argv)

    suppress_native_crash_dialogs()
    train_center = load_module("train_center")
    original_command = train_center._training_command

    def seeded_command(args, **kwargs):
        command = original_command(args, **kwargs)
        command.extend(("--seed", str(seed_args.seed)))
        return command

    train_center._training_command = seeded_command
    return int(train_center.main(remaining))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
