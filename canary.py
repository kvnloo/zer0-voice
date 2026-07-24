#!/usr/bin/env python3
"""Read-only dual-lane voice canary auditor; never emits transcript content."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from health import (
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    assess as assess_health,
    read_snapshot,
)
from verify_metrics import (
    DEFAULT_MINIMUM_SAMPLES,
    PRODUCTION_PIPELINE,
    percentile,
    read_jsonl,
    verify,
)

DEFAULT_DUPLICATE_WINDOW_SECONDS = 1.5
DEFAULT_MAX_INTERRUPTION_COUNT = 1
DEFAULT_MAX_ERROR_COUNT = 0
MINIMUM_PROMOTION_TURNS = 10


def current_run(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts = [
        index
        for index, event in enumerate(events)
        if event.get("kind") == "voice.starting"
    ]
    return events[starts[-1] :] if starts else events


def timestamp_ns(row: dict[str, Any]) -> int:
    for field in ("ts_ns", "recorded_at_ns", "updated_ns"):
        try:
            value = int(row.get(field, 0))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def after(
    rows: list[dict[str, Any]],
    start_ns: int | None,
) -> list[dict[str, Any]]:
    if start_ns is None:
        return rows
    return [row for row in rows if timestamp_ns(row) >= start_ns]


def canonical_transcript(value: object) -> str:
    """Normalize only in memory for duplicate detection; never return it."""
    return " ".join(re.findall(r"[a-z0-9']+", str(value).lower()))


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "median": None, "p95": None, "max": None}
    return {
        "samples": len(values),
        "median": round(statistics.median(values), 6),
        "p95": round(percentile(values, 0.95), 6),
        "max": round(max(values), 6),
    }


def sequence_audit(
    events: list[dict[str, Any]],
    *,
    duplicate_window_seconds: float,
) -> tuple[dict[str, int], dict[str, dict[str, float | int | None]]]:
    """Count canonical final → live-running → response cycles."""
    pending_final_ns: int | None = None
    live_running_ns: int | None = None
    seen_final = False
    sequence_violations = 0
    completions = 0
    rapid_duplicates = 0
    empty_model_outputs = 0
    previous_by_canonical: dict[str, int] = {}
    final_to_running: list[float] = []
    running_to_response: list[float] = []
    final_to_response: list[float] = []
    duplicate_window_ns = round(duplicate_window_seconds * 1_000_000_000)

    for event in events:
        kind = str(event.get("kind", ""))
        event_ns = timestamp_ns(event)
        if kind == "asr.final":
            if pending_final_ns is not None:
                sequence_violations += 1
            pending_final_ns = event_ns or None
            live_running_ns = None
            seen_final = True
            canonical = canonical_transcript(event.get("text", ""))
            previous_ns = previous_by_canonical.get(canonical) if canonical else None
            if (
                canonical
                and previous_ns is not None
                and event_ns >= previous_ns
                and event_ns - previous_ns <= duplicate_window_ns
            ):
                rapid_duplicates += 1
            if canonical and event_ns:
                previous_by_canonical[canonical] = event_ns
            continue

        if (
            kind == "voice.lane.state"
            and event.get("lane") == "live"
            and event.get("state") == "running"
        ):
            if pending_final_ns is None:
                if seen_final:
                    sequence_violations += 1
                continue
            if live_running_ns is not None:
                sequence_violations += 1
                continue
            live_running_ns = event_ns or pending_final_ns
            final_to_running.append(
                max(0.0, live_running_ns - pending_final_ns) / 1_000_000_000
            )
            continue

        if kind == "voice.response" and event.get("lane") == "live":
            if pending_final_ns is None or live_running_ns is None:
                if seen_final:
                    sequence_violations += 1
                continue
            if not str(event.get("text", "")).strip():
                empty_model_outputs += 1
                pending_final_ns = None
                live_running_ns = None
                continue
            response_ns = event_ns or live_running_ns
            running_to_response.append(
                max(0.0, response_ns - live_running_ns) / 1_000_000_000
            )
            final_to_response.append(
                max(0.0, response_ns - pending_final_ns) / 1_000_000_000
            )
            completions += 1
            pending_final_ns = None
            live_running_ns = None
            continue

        if kind in ("voice.interrupted", "voice.live.error", "voice.error"):
            pending_final_ns = None
            live_running_ns = None

    return (
        {
            "canonical_completions": completions,
            "sequence_violations": sequence_violations,
            "rapid_duplicate_finals": rapid_duplicates,
            "empty_model_outputs": empty_model_outputs,
            "in_flight": int(pending_final_ns is not None),
        },
        {
            "final_to_live_running_seconds": latency_summary(final_to_running),
            "live_running_to_response_seconds": latency_summary(running_to_response),
            "final_to_response_seconds": latency_summary(final_to_response),
        },
    )


def audit(
    metrics: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
    now_ns: int | None = None,
    start_ns: int | None = None,
    health: dict[str, Any] | None = None,
    heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    duplicate_window_seconds: float = DEFAULT_DUPLICATE_WINDOW_SECONDS,
    max_interruptions: int = DEFAULT_MAX_INTERRUPTION_COUNT,
    max_errors: int = DEFAULT_MAX_ERROR_COUNT,
    bundle_sha256: str | None = None,
) -> dict[str, Any]:
    required_samples = max(minimum_samples, MINIMUM_PROMOTION_TURNS)
    run = current_run(events)
    run_id = next(
        (
            str(event["run_id"])
            for event in run
            if event.get("kind") == "voice.starting" and event.get("run_id")
        ),
        None,
    )
    run_bundle = next(
        (
            str(event["bundle_sha256"])
            for event in run
            if event.get("kind") == "voice.starting"
            and event.get("bundle_sha256")
        ),
        None,
    )
    cohort = after(run, start_ns)
    metric_cohort = after(metrics, start_ns)
    kinds = [str(event.get("kind", "")) for event in cohort]
    errors = [
        kind
        for kind in kinds
        if kind == "voice.error"
        or kind.endswith(".error")
        or kind == "audio.warning"
    ]
    gate = verify(
        metric_cohort,
        minimum_samples=required_samples,
        pipeline=PRODUCTION_PIPELINE,
        run_id=run_id,
        window_samples=required_samples,
    )
    sequence_counts, latencies = sequence_audit(
        cohort,
        duplicate_window_seconds=duplicate_window_seconds,
    )
    latest_ns = max(
        (timestamp_ns(event) for event in cohort),
        default=0,
    )
    now_ns = time.time_ns() if now_ns is None else now_ns
    activity_age_seconds = (
        round(max(0, now_ns - latest_ns) / 1_000_000_000, 3)
        if latest_ns
        else None
    )
    counts = {
        "speech_started": kinds.count("speech.started"),
        "partials": kinds.count("asr.partial"),
        "finals": kinds.count("asr.final"),
        "live_responses": sum(
            event.get("kind") == "voice.response"
            and event.get("lane") == "live"
            for event in cohort
        ),
        "authoritative_submissions": kinds.count("codex.authoritative.submitted"),
        "interruptions": kinds.count("voice.interrupted"),
        "errors": len(errors),
        **sequence_counts,
    }
    invariants = []
    if counts["live_responses"] > counts["finals"]:
        invariants.append("live responses exceed committed finals")
    if counts["authoritative_submissions"] > counts["finals"]:
        invariants.append("authoritative submissions exceed committed finals")
    if counts["rapid_duplicate_finals"]:
        invariants.append(
            f"rapid duplicate finals {counts['rapid_duplicate_finals']} > allowed 0"
        )
    if counts["sequence_violations"]:
        invariants.append(
            f"canonical sequence violations {counts['sequence_violations']} > allowed 0"
        )
    if counts["empty_model_outputs"]:
        invariants.append(
            f"empty model outputs {counts['empty_model_outputs']} > allowed 0"
        )
    if counts["interruptions"] > max_interruptions:
        invariants.append(
            f"interruptions {counts['interruptions']} > allowed {max_interruptions}"
        )
    if counts["errors"] > max_errors:
        invariants.append(f"errors {counts['errors']} > allowed {max_errors}")
    if bundle_sha256 and run_bundle != bundle_sha256:
        invariants.append("runtime bundle does not match requested candidate")

    health_snapshot = health or {}
    health_ok, health_reason = assess_health(
        health_snapshot,
        now_ns=now_ns,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    )
    health_run_id = health_snapshot.get("run_id")
    if health_ok and run_id and health_run_id != run_id:
        health_ok = False
        health_reason = "heartbeat-run-mismatch"
    if not health_ok:
        invariants.append(f"runtime health not fresh: {health_reason}")

    enough_sequences = counts["canonical_completions"] >= required_samples
    if gate["ok"] and enough_sequences and not invariants:
        status = "passed"
    elif invariants or (
        gate["completed_samples"] >= required_samples and not gate["ok"]
    ):
        status = "failed"
    else:
        status = "collecting"
    return {
        "schema": 1,
        "pipeline": PRODUCTION_PIPELINE,
        "bundle_sha256": run_bundle,
        "run_id": run_id,
        "status": status,
        "start_ns": start_ns,
        "minimum_samples": required_samples,
        "limits": {
            "max_interruptions": max_interruptions,
            "max_errors": max_errors,
            "duplicate_window_seconds": duplicate_window_seconds,
            "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        },
        "activity_age_seconds": activity_age_seconds,
        "counts": counts,
        "latencies": latencies,
        "health": {
            "ok": health_ok,
            "reason": health_reason,
            "age_seconds": (
                round(
                    max(0, now_ns - timestamp_ns(health_snapshot))
                    / 1_000_000_000,
                    3,
                )
                if timestamp_ns(health_snapshot)
                else None
            ),
        },
        "promotion": {
            "eligible": status == "passed",
            "verdict": (
                "promote"
                if status == "passed"
                else "reject"
                if status == "failed"
                else "hold"
            ),
            "required_completed_turns": required_samples,
            "observed_completed_turns": min(
                int(gate["completed_samples"]),
                counts["canonical_completions"],
            ),
        },
        "invariants": invariants,
        "gate": gate,
    }


def main() -> int:
    root = Path(__file__).parents[1]
    user_state = Path.home() / ".local/state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=root / "bench/voice-history.jsonl",
    )
    parser.add_argument(
        "--debug",
        type=Path,
        default=user_state / "zer0-voice/voice-debug.jsonl",
    )
    parser.add_argument(
        "--health",
        type=Path,
        default=user_state / "zer0-voice/health.json",
    )
    parser.add_argument(
        "--bundle-sha256",
        required=True,
        help="exact staged bundle expected in the current worker generation",
    )
    parser.add_argument(
        "--start-ns",
        type=int,
        help="evaluate only metrics/events at or after this timestamp",
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=DEFAULT_MINIMUM_SAMPLES,
    )
    parser.add_argument(
        "--max-interruptions",
        type=int,
        default=DEFAULT_MAX_INTERRUPTION_COUNT,
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERROR_COUNT,
    )
    parser.add_argument(
        "--duplicate-window",
        type=float,
        default=DEFAULT_DUPLICATE_WINDOW_SECONDS,
        help="seconds in which the same canonical final counts as a duplicate",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    )
    args = parser.parse_args()
    result = audit(
        read_jsonl(args.metrics) if args.metrics.exists() else [],
        read_jsonl(args.debug) if args.debug.exists() else [],
        minimum_samples=args.minimum_samples,
        start_ns=args.start_ns,
        health=read_snapshot(args.health),
        heartbeat_timeout_seconds=args.heartbeat_timeout,
        duplicate_window_seconds=args.duplicate_window,
        max_interruptions=args.max_interruptions,
        max_errors=args.max_errors,
        bundle_sha256=args.bundle_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return {"passed": 0, "failed": 1}.get(result["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
