#!/usr/bin/env python3
"""Deterministic latency/reliability gate for privacy-safe voice metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


DEFAULT_BUDGETS = {
    "speech_to_partial_seconds": 0.35,
    "asr_seconds": 0.50,
    "tts_first_seconds": 0.50,
    "audio_onset_after_endpoint_seconds": 2.50,
    "estimated_audio_onset_after_user_stop_seconds": 3.00,
}
PRODUCTION_PIPELINE = "codex-continuous-pcm-v5"
DEFAULT_MINIMUM_SAMPLES = 10


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def verify(
    rows: list[dict[str, object]],
    *,
    minimum_samples: int = 1,
    budgets: dict[str, float] | None = None,
    pipeline: str | None = None,
    run_id: str | None = None,
    window_samples: int | None = None,
) -> dict[str, object]:
    budgets = budgets or DEFAULT_BUDGETS
    pipeline_rows = (
        [row for row in rows if row.get("pipeline") == pipeline]
        if pipeline
        else rows
    )
    selected = (
        [row for row in pipeline_rows if row.get("run_id") == run_id]
        if run_id
        else pipeline_rows
    )
    completed_all = [
        row
        for row in selected
        if not row.get("interrupted")
        and all(isinstance(row.get(field), (int, float)) for field in budgets)
    ]
    completed = (
        completed_all[-window_samples:]
        if window_samples is not None
        else completed_all
    )
    metrics: dict[str, dict[str, float]] = {}
    violations: list[str] = []
    if len(completed_all) < minimum_samples:
        violations.append(
            f"completed samples {len(completed_all)} < required {minimum_samples}"
        )
    for field, budget in budgets.items():
        values = [float(row[field]) for row in completed]
        if not values:
            continue
        median = statistics.median(values)
        p95 = percentile(values, 0.95)
        metrics[field] = {
            "median": round(median, 6),
            "p95": round(p95, 6),
            "budget": budget,
        }
        if p95 > budget:
            violations.append(f"{field} p95 {p95:.6f} > {budget:.6f}")
    return {
        "schema": 1,
        "ok": not violations,
        "pipeline": pipeline,
        "run_id": run_id,
        "rows": len(selected),
        "ledger_rows": len(rows),
        "completed_samples": len(completed_all),
        "gate_window_samples": len(completed),
        "interrupted_samples": sum(bool(row.get("interrupted")) for row in selected),
        "metrics": metrics,
        "violations": violations,
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger",
        type=Path,
        nargs="?",
        default=Path(__file__).parents[1] / "bench/voice-history.jsonl",
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=DEFAULT_MINIMUM_SAMPLES,
    )
    parser.add_argument("--pipeline", default=PRODUCTION_PIPELINE)
    args = parser.parse_args()
    result = verify(
        read_jsonl(args.ledger),
        minimum_samples=args.minimum_samples,
        pipeline=args.pipeline,
        window_samples=args.minimum_samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
