#!/usr/bin/env python3
"""Terminate a stuck voice worker so its supervisor can restart it cleanly."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
import urllib.request
from pathlib import Path

from health import assess, read_snapshot


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def http_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.5) as response:
            if response.status >= 300:
                return False
            body = json.loads(response.read())
            return body.get("status") == "healthy"
    except (OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--health", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--startup-grace", type=float, default=25.0)
    parser.add_argument("--stale", type=float, default=5.0)
    parser.add_argument("--kokoro-health")
    parser.add_argument("--dependency-failures", type=int, default=3)
    parser.add_argument("--terminate-grace", type=float, default=2.0)
    args = parser.parse_args()

    started = time.monotonic()
    dependency_failures = 0
    while alive(args.pid):
        record = read_snapshot(args.health)
        matching = record and int(record.get("pid", -1)) == args.pid
        if not matching and time.monotonic() - started < args.startup_grace:
            time.sleep(args.interval)
            continue
        if matching:
            healthy, reason = assess(
                record,
                heartbeat_timeout_seconds=args.stale,
            )
        else:
            healthy, reason = False, "heartbeat-missing"
        if not healthy:
            print(f"voice watchdog restarting pid={args.pid}: {reason}", flush=True)
            try:
                os.kill(args.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + args.terminate_grace
            while alive(args.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if alive(args.pid):
                try:
                    os.kill(args.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return 2
        if args.kokoro_health:
            dependency_failures = (
                0
                if http_healthy(args.kokoro_health)
                else dependency_failures + 1
            )
            if dependency_failures >= args.dependency_failures:
                print(
                    "voice watchdog restarting "
                    f"pid={args.pid}: kokoro_unhealthy",
                    flush=True,
                )
                try:
                    os.kill(args.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + args.terminate_grace
                while alive(args.pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if alive(args.pid):
                    try:
                        os.kill(args.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                return 2
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
