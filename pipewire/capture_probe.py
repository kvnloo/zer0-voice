#!/usr/bin/env python3
"""Prove a PipeWire/Pulse source emits frames without retaining microphone audio."""

from __future__ import annotations

import argparse
import os
import selectors
import subprocess
import sys
import time
from collections.abc import Callable


def wait_for_byte(
    descriptor: int,
    timeout: float,
    *,
    selector_factory: Callable[[], selectors.BaseSelector] = selectors.DefaultSelector,
    read: Callable[[int, int], bytes] = os.read,
) -> bool:
    deadline = time.monotonic() + timeout
    with selector_factory() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if not selector.select(remaining):
                return False
            if read(descriptor, 1):
                return True
            return False
    return False


def probe(device: str, timeout: float, executable: str = "parec") -> bool:
    process = subprocess.Popen(
        [
            executable,
            "--raw",
            f"--device={device}",
            "--rate=48000",
            "--channels=1",
            "--format=float32le",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert process.stdout is not None
        return wait_for_byte(process.stdout.fileno(), timeout)
    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    if probe(args.device, args.timeout):
        return os.EX_OK
    print(f"capture stalled: {args.device}", file=sys.stderr, flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
