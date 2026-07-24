#!/usr/bin/env python3
"""Change or inspect the running Zer0 voice mode."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from control_plane import default_control_socket, request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=default_control_socket())
    commands = parser.add_subparsers(dest="command", required=True)
    mic = commands.add_parser("mic")
    mic.add_argument("mode", choices=("push-to-talk", "continuous", "muted"))
    notify = commands.add_parser("notify")
    notify.add_argument("mode", choices=("conversational", "updates", "critical"))
    reload = commands.add_parser("reload")
    reload.add_argument("--live-model", default=None)
    reload.add_argument("--live-effort", default=None)
    commands.add_parser("press")
    commands.add_parser("release")
    commands.add_parser("status")
    args = parser.parse_args()

    if args.command == "mic":
        command = {"mic": args.mode}
    elif args.command == "notify":
        command = {"notifications": args.mode}
    elif args.command == "press":
        command = {"push_held": True}
    elif args.command == "release":
        command = {"push_held": False}
    elif args.command == "status":
        command = {}
    else:
        command = {
            **({"live_model": args.live_model} if args.live_model is not None else {}),
            **({"live_effort": args.live_effort} if args.live_effort is not None else {}),
        }
    if args.command == "reload" and not command:
        raise SystemExit("reload requires --live-model and/or --live-effort")
    result = asyncio.run(request(args.socket, command))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
