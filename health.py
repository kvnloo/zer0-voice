#!/usr/bin/env python3
"""Privacy-safe runtime heartbeat and watchdog policy for continuous voice."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 5.0
DEFAULT_CAPTURE_TIMEOUT_SECONDS = 3.0
DEFAULT_PHASE_DEADLINES_SECONDS = {
    "starting": 180.0,
    "attaching": 20.0,
    "transcribing": 20.0,
    "syncing": 20.0,
    "generating": 35.0,
    "speaking": 30.0,
    "recovering": 20.0,
}


def read_snapshot(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def assess(
    snapshot: dict[str, Any],
    *,
    now_ns: int | None = None,
    heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    capture_timeout_seconds: float = DEFAULT_CAPTURE_TIMEOUT_SECONDS,
    phase_deadlines_seconds: dict[str, float] | None = None,
) -> tuple[bool, str]:
    """Return functional health without inspecting transcript or response text."""
    if not snapshot:
        return False, "heartbeat-missing"
    now_ns = now_ns or time.time_ns()
    updated_ns = int(snapshot.get("updated_ns", 0))
    if updated_ns <= 0:
        return False, "heartbeat-invalid"
    age = max(0.0, (now_ns - updated_ns) / 1_000_000_000)
    if age > heartbeat_timeout_seconds:
        return False, f"heartbeat-stale:{age:.1f}s"
    if snapshot.get("capture_expected") is True:
        capture_updated_ns = int(snapshot.get("capture_updated_ns", 0))
        if capture_updated_ns <= 0:
            return False, "capture-missing"
        capture_age = max(
            0.0,
            (now_ns - capture_updated_ns) / 1_000_000_000,
        )
        if capture_age > capture_timeout_seconds:
            return False, f"capture-stalled:{capture_age:.1f}s"
    phase = str(snapshot.get("phase", "unknown"))
    deadlines = phase_deadlines_seconds or DEFAULT_PHASE_DEADLINES_SECONDS
    deadline = deadlines.get(phase)
    if deadline is not None:
        since_ns = int(snapshot.get("phase_since_ns", updated_ns))
        phase_age = max(0.0, (now_ns - since_ns) / 1_000_000_000)
        if phase_age > deadline:
            return False, f"phase-stale:{phase}:{phase_age:.1f}s"
    return True, "healthy"


@dataclass(slots=True)
class RuntimeHealth:
    """Atomically publish one worker's phase; never stores conversation content."""

    path: Path
    run_id: str
    phase: str = "starting"
    phase_since_ns: int = 0
    revision: int = 0
    lane: str = "runtime"
    reason: str = ""
    capture_expected: bool = False
    capture_frames: int = 0
    capture_updated_ns: int = 0
    mic_mode: str = "unknown"
    notification_mode: str = "unknown"
    capture_active: bool = False
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.phase_since_ns = self.phase_since_ns or time.time_ns()
        self.touch()

    def transition(
        self,
        phase: str,
        *,
        lane: str = "runtime",
        reason: str = "",
    ) -> None:
        with self._lock:
            if phase != self.phase or lane != self.lane:
                self.phase_since_ns = time.time_ns()
            self.phase = phase
            self.lane = lane
            self.reason = reason
            self.revision += 1
            self._touch_unlocked()

    def touch(self) -> None:
        with self._lock:
            self._touch_unlocked()

    def expect_capture(self, expected: bool) -> None:
        """Declare whether audio callbacks must currently make progress."""
        with self._lock:
            if expected and not self.capture_expected:
                self.capture_updated_ns = time.time_ns()
            self.capture_expected = expected
            self._touch_unlocked()

    def captured(self, frames: int) -> None:
        """Record callback progress in memory; the heartbeat publishes it."""
        if frames <= 0:
            return
        with self._lock:
            self.capture_frames += frames
            self.capture_updated_ns = time.time_ns()

    def control(
        self,
        *,
        mic_mode: str,
        notification_mode: str,
        capture_active: bool,
    ) -> None:
        """Publish operator mode independently from transient worker phase."""
        with self._lock:
            self.mic_mode = mic_mode
            self.notification_mode = notification_mode
            self.capture_active = capture_active
            self._touch_unlocked()

    def _touch_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "schema": 1,
            "pid": os.getpid(),
            "run_id": self.run_id,
            "phase": self.phase,
            "lane": self.lane,
            "reason": self.reason,
            "capture_expected": self.capture_expected,
            "capture_frames": self.capture_frames,
            "capture_updated_ns": self.capture_updated_ns,
            "mic_mode": self.mic_mode,
            "notification_mode": self.notification_mode,
            "capture_active": self.capture_active,
            "phase_since_ns": self.phase_since_ns,
            "updated_ns": time.time_ns(),
            "revision": self.revision,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.{os.getpid()}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class HealthRecord(RuntimeHealth):
    """Tiny runtime-facing compatibility API for the duplex worker."""

    def set(self, phase: str, *, lane: str = "runtime", reason: str = "") -> None:
        self.transition(phase, lane=lane, reason=reason)

    def beat(self) -> None:
        self.touch()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    )
    args = parser.parse_args()
    snapshot = read_snapshot(args.path)
    ok, reason = assess(
        snapshot,
        heartbeat_timeout_seconds=args.heartbeat_timeout,
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "reason": reason,
                "phase": snapshot.get("phase", "missing"),
                "lane": snapshot.get("lane", "runtime"),
                "pid": snapshot.get("pid"),
                "run_id": snapshot.get("run_id"),
                "mic_mode": snapshot.get("mic_mode", "unknown"),
                "capture_active": snapshot.get("capture_active", False),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
