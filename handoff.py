#!/usr/bin/env python3
"""Fail-closed, turn-boundary handoff between warm voice generations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from control_plane import request
from health import assess, read_snapshot

ControlRequest = Callable[
    [Path, dict[str, object]], Awaitable[dict[str, object]]
]


class HandoffError(RuntimeError):
    """A candidate was not safe to receive microphone ownership."""


@dataclass(frozen=True, slots=True)
class Candidate:
    control: Path
    health: Path


def ready(
    health: dict[str, object],
    status: dict[str, object],
    *,
    now_ns: int | None = None,
) -> tuple[bool, str]:
    """Require a fresh, attached, muted generation before any switch."""
    healthy, reason = assess(health, now_ns=now_ns)
    if not healthy:
        return False, reason
    if health.get("phase") != "listening":
        return False, f"candidate-phase:{health.get('phase', 'missing')}"
    if int(health.get("pid", 0)) <= 0:
        return False, "candidate-pid-missing"
    if status.get("mic") != "muted" or status.get("capture_active") is not False:
        return False, "candidate-already-capturing"
    if not status.get("live_model"):
        return False, "candidate-live-model-missing"
    return True, "ready"


async def wait_ready(
    candidate: Candidate,
    *,
    timeout: float = 30.0,
    interval: float = 0.05,
    request_timeout: float = 1.0,
    control_request: ControlRequest = request,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_reason = "candidate-not-started"
    while time.monotonic() < deadline:
        try:
            status = await asyncio.wait_for(
                control_request(candidate.control, {}),
                timeout=min(
                    request_timeout,
                    max(0.001, deadline - time.monotonic()),
                ),
            )
            health = read_snapshot(candidate.health)
            ok, last_reason = ready(health, status)
            if ok:
                return {
                    "health": health,
                    "status": status,
                    "reason": last_reason,
                }
        except (OSError, RuntimeError, TimeoutError, ValueError):
            last_reason = "candidate-control-unavailable"
        await asyncio.sleep(interval)
    raise HandoffError(f"candidate readiness timed out: {last_reason}")


async def handoff(
    active: Candidate,
    candidate: Candidate,
    *,
    control_request: ControlRequest = request,
    readiness_timeout: float = 30.0,
    request_timeout: float = 1.0,
    probation_seconds: float = 0.5,
) -> dict[str, object]:
    """Move capture at an idle turn boundary; restore active on any failure."""
    try:
        old = await asyncio.wait_for(
            control_request(active.control, {}),
            timeout=request_timeout,
        )
    except TimeoutError as error:
        raise HandoffError("active control request timed out") from error
    if old.get("capture_active") is not True:
        raise HandoffError("active generation does not own capture")
    old_health = read_snapshot(active.health)
    old_healthy, old_reason = assess(old_health)
    if not old_healthy or old_health.get("phase") != "listening":
        raise HandoffError(f"active generation is not idle: {old_reason}")
    candidate_ready = await wait_ready(
        candidate,
        timeout=readiness_timeout,
        request_timeout=request_timeout,
        control_request=control_request,
    )
    old_mode = str(old.get("mic", "continuous"))
    try:
        muted = await asyncio.wait_for(
            control_request(active.control, {"mic": "muted"}),
            timeout=request_timeout,
        )
        confirmed = await asyncio.wait_for(
            control_request(active.control, {}),
            timeout=request_timeout,
        )
        for response in (muted, confirmed):
            if (
                response.get("mic") != "muted"
                or response.get("capture_active") is not False
            ):
                raise HandoffError("active mute was not confirmed")
    except BaseException as mute_error:
        try:
            restored = await asyncio.wait_for(
                control_request(active.control, {"mic": old_mode}),
                timeout=request_timeout,
            )
        except (OSError, RuntimeError, TimeoutError) as restore_error:
            raise HandoffError(
                "active mute failed and rollback was unavailable"
            ) from restore_error
        if restored.get("capture_active") is not True:
            raise HandoffError("active mute failed and rollback failed")
        if isinstance(mute_error, TimeoutError):
            raise HandoffError("active mute request timed out") from mute_error
        if isinstance(mute_error, HandoffError):
            raise mute_error
        raise HandoffError("active mute confirmation failed") from mute_error
    try:
        promoted = await asyncio.wait_for(
            control_request(candidate.control, {"mic": old_mode}),
            timeout=request_timeout,
        )
        if promoted.get("capture_active") is not True:
            raise HandoffError("candidate did not acquire capture")
        await asyncio.sleep(max(0.0, probation_seconds))
        stable = await asyncio.wait_for(
            control_request(candidate.control, {}),
            timeout=request_timeout,
        )
        stable_health = read_snapshot(candidate.health)
        stable_healthy, stable_reason = assess(stable_health)
        expected_pid = int(candidate_ready["health"].get("pid", 0))
        if (
            not stable_healthy
            or int(stable_health.get("pid", 0)) != expected_pid
            or stable.get("mic") != old_mode
            or stable.get("capture_active") is not True
        ):
            raise HandoffError(
                "candidate failed post-handoff probation: "
                + (
                    stable_reason
                    if not stable_healthy
                    else "generation-or-capture-changed"
                )
            )
    except BaseException as acquisition_error:
        try:
            muted_candidate = await asyncio.wait_for(
                control_request(candidate.control, {"mic": "muted"}),
                timeout=request_timeout,
            )
            confirmed_candidate = await asyncio.wait_for(
                control_request(candidate.control, {}),
                timeout=request_timeout,
            )
            for response in (muted_candidate, confirmed_candidate):
                if (
                    response.get("mic") != "muted"
                    or response.get("capture_active") is not False
                ):
                    raise HandoffError("candidate mute was not confirmed")
        except (HandoffError, OSError, RuntimeError, TimeoutError) as mute_error:
            # Never restore the old capture owner while the candidate might
            # still own the device. The caller must stop/prove the candidate
            # dead, then explicitly restore the retained active generation.
            raise HandoffError(
                "candidate capture state unknown; active remains muted "
                "for manual recovery"
            ) from mute_error
        try:
            restored = await asyncio.wait_for(
                control_request(active.control, {"mic": old_mode}),
                timeout=request_timeout,
            )
        except TimeoutError as error:
            raise HandoffError(
                "candidate failed and active rollback timed out"
            ) from error
        if restored.get("capture_active") is not True:
            raise HandoffError("candidate failed and active rollback failed")
        if isinstance(acquisition_error, TimeoutError):
            raise HandoffError(
                "candidate acquisition request timed out"
            ) from acquisition_error
        raise
    return {
        "schema": 1,
        "status": "candidate-active",
        "old_pid": int(old_health.get("pid", 0)),
        "candidate_pid": int(read_snapshot(candidate.health).get("pid", 0)),
        "old_retained_for_rollback": True,
        "post_handoff_probation_seconds": max(0.0, probation_seconds),
        "capture_gap_bound": "two-bounded-local-control-round-trips",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-control", required=True, type=Path)
    parser.add_argument("--active-health", required=True, type=Path)
    parser.add_argument("--candidate-control", required=True, type=Path)
    parser.add_argument("--candidate-health", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(
        handoff(
            Candidate(args.active_control, args.active_health),
            Candidate(args.candidate_control, args.candidate_health),
        )
    )
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
