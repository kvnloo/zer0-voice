#!/usr/bin/env python3
"""Restart a voice worker whose direct ALSA capture cursor stops advancing."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from health import read_snapshot


@dataclass(frozen=True, slots=True)
class Cursor:
    state: str
    owner_pid: int
    hardware: int
    application: int


def parse_cursor(text: str) -> Cursor | None:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    try:
        return Cursor(
            fields["state"],
            int(fields["owner_pid"]),
            int(fields["hw_ptr"]),
            int(fields["appl_ptr"]),
        )
    except (KeyError, ValueError):
        return None


def card_number(cards: str, card_id: str) -> int | None:
    match = re.search(
        rf"(?m)^\s*(\d+)\s+\[{re.escape(card_id)}\s*\]\s*:",
        cards,
    )
    return int(match.group(1)) if match else None


def pcm_path(cards_path: Path, proc_root: Path, card_id: str) -> Path | None:
    try:
        number = card_number(cards_path.read_text(encoding="utf-8"), card_id)
    except OSError:
        return None
    if number is None:
        return None
    path = proc_root / f"card{number}/pcm0c/sub0/status"
    return path if path.is_file() else None


def read_cursor(path: Path | None) -> Cursor | None:
    if path is None:
        return None
    try:
        return parse_cursor(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def assess_progress(
    before: Cursor | None,
    after: Cursor | None,
    expected_pid: int,
) -> tuple[bool, str]:
    if before is None or after is None:
        return False, "alsa-cursor-missing"
    if after.state != "RUNNING":
        return False, f"alsa-state:{after.state.lower()}"
    if after.owner_pid != expected_pid:
        return False, "alsa-owner-mismatch"
    if after.hardware <= before.hardware:
        return False, "alsa-hardware-stalled"
    if after.application <= before.application:
        return False, "alsa-consumer-stalled"
    return True, "progressing"


def capture_active(path: Path) -> bool | None:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(0.5)
    try:
        client.connect(str(path))
        client.sendall(b"{}\n")
        raw = b""
        while b"\n" not in raw and len(raw) <= 65_536:
            chunk = client.recv(4096)
            if not chunk:
                break
            raw += chunk
        response = json.loads(raw)
        if response.get("ok") is not True:
            return None
        return response.get("capture_active") is True
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    finally:
        client.close()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def terminate(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health", required=True, type=Path)
    parser.add_argument("--control-socket", required=True, type=Path)
    parser.add_argument("--cards", type=Path, default=Path("/proc/asound/cards"))
    parser.add_argument("--proc-root", type=Path, default=Path("/proc/asound"))
    parser.add_argument("--card-id", default="Snowball")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--startup-grace", type=float, default=25.0)
    parser.add_argument("--failures", type=int, default=3)
    args = parser.parse_args()
    if args.interval <= 0 or args.startup_grace < 0 or args.failures <= 0:
        parser.error("interval and failures must be positive")

    worker_pid = 0
    worker_started = 0.0
    failures = 0
    control_failures = 0
    previous: Cursor | None = None
    while True:
        record = read_snapshot(args.health)
        pid = int(record.get("pid", 0))
        if pid <= 0 or not alive(pid):
            worker_pid = 0
            previous = None
            failures = 0
            control_failures = 0
            time.sleep(args.interval)
            continue
        if pid != worker_pid:
            worker_pid = pid
            worker_started = time.monotonic()
            failures = 0
            control_failures = 0
            previous = None

        expected = capture_active(args.control_socket)
        if expected is None:
            if time.monotonic() - worker_started >= args.startup_grace:
                control_failures += 1
                if control_failures >= args.failures:
                    print(
                        "alsa watchdog restarting "
                        f"pid={worker_pid}: control-unavailable",
                        flush=True,
                    )
                    terminate(worker_pid)
                    return 2
            time.sleep(args.interval)
            continue
        control_failures = 0
        if expected is False:
            previous = None
            failures = 0
            time.sleep(args.interval)
            continue

        path = pcm_path(args.cards, args.proc_root, args.card_id)
        current = read_cursor(path)
        if previous is None:
            previous = current
            time.sleep(args.interval)
            continue
        healthy, reason = assess_progress(previous, current, worker_pid)
        previous = current
        if healthy:
            failures = 0
        elif time.monotonic() - worker_started >= args.startup_grace:
            failures += 1
            if failures >= args.failures:
                print(
                    f"alsa watchdog restarting pid={worker_pid}: {reason}",
                    flush=True,
                )
                terminate(worker_pid)
                return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
