"""Load and verify one CARLA map on an already-running server."""
from __future__ import annotations

import argparse
import sys
import time

import carla


SUPPORTED_MAPS = {
    "town02": "Town02",
    "town05": "Town05",
    "town10hd": "Town10HD",
}


def _map_name(value: str) -> str:
    try:
        return SUPPORTED_MAPS[value.strip().casefold()]
    except KeyError as error:
        raise argparse.ArgumentTypeError("choose Town02, Town05, or Town10HD") from error


def _key(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].split(".", 1)[0].casefold()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=_map_name)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rpc-timeout", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.rpc_timeout)
    deadline = time.monotonic() + args.startup_timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            world = client.get_world()
            break
        except RuntimeError as error:
            last_error = error
            time.sleep(1.0)
    else:
        raise RuntimeError(f"CARLA {args.host}:{args.port} not ready: {last_error}")

    if _key(world.get_map().name) != _key(args.map):
        world = client.load_world(args.map)
    actual = world.get_map().name
    if _key(actual) != _key(args.map):
        raise RuntimeError(f"expected {args.map}, got {actual}")
    print(f"Ready: {actual} on {args.host}:{args.port}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
