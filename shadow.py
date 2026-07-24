#!/usr/bin/env python3
"""Hermetic shadow-candidate voice runner with transcript-free results."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from floor import TurnOwner

SHADOW_PIPELINE = "voice-shadow-v1"
MINIMUM_PROMOTION_CASES = 10
EXPECTED_STAGES = ("asr", "turn_owner", "model", "tts", "sink")
SAFE_CASE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
SAFE_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
FORBIDDEN_REPORT_KEYS = {
    "content",
    "prompt",
    "response",
    "text",
    "transcript",
}


class CorpusAsr(Protocol):
    input_mode: str

    async def transcribe(self, audio: bytes) -> str: ...


class EphemeralModel(Protocol):
    mode: str

    async def generate(
        self,
        prompt: str,
        *,
        completed_turn_snapshot: bytes | None,
    ) -> str: ...


class PcmSynthesizer(Protocol):
    output_mode: str

    async def synthesize(self, text: str) -> bytes: ...


class DiscardSink(Protocol):
    mode: str

    async def consume(self, pcm: bytes) -> str: ...


@dataclass(frozen=True, slots=True)
class ShadowFragment:
    audio: bytes
    at_seconds: float


@dataclass(frozen=True, slots=True)
class ShadowCase:
    case_id: str
    fragments: tuple[ShadowFragment, ...]
    completed_turn_snapshot: bytes | None = None


@dataclass(frozen=True, slots=True)
class _StageFailure(Exception):
    stage: str
    code: str


@dataclass(frozen=True, slots=True)
class _CaseOutcome:
    report: dict[str, Any]
    canonical_final: str = ""
    secrets: tuple[str, ...] = ()


class TranscriptLeakError(RuntimeError):
    pass


class HashDiscardSink:
    """Hash PCM in memory and discard it; never opens an audio device or file."""

    mode = "discard-hash"

    async def consume(self, pcm: bytes) -> str:
        return hashlib.sha256(pcm).hexdigest()


def canonical(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.lower()))


def assert_transcript_free(
    report: object,
    *,
    secrets: tuple[str, ...] = (),
) -> None:
    """Reject a report schema that could carry candidate conversation text."""
    leaves: list[str] = []

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN_REPORT_KEYS:
                    raise TranscriptLeakError("forbidden report field")
                inspect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)
        elif isinstance(value, str):
            leaves.append(value)

    inspect(report)
    secret_values = {value for value in secrets if value}
    if secret_values.intersection(leaves):
        raise TranscriptLeakError("candidate text reached report values")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = max(0, (len(ordered) * 95 + 99) // 100 - 1)
    return {
        "samples": len(values),
        "median": round(statistics.median(values), 6),
        "p95": round(ordered[p95_index], 6),
        "max": round(ordered[-1], 6),
    }


def _safe_failure(
    case_id: str,
    *,
    stage: str,
    code: str,
    sequence: list[str] | None = None,
    latencies: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "ok": False,
        "stage_sequence": list(sequence or ()),
        "latencies": dict(latencies or {}),
        "failure": {"stage": stage, "code": code},
    }


async def _timed(
    stage: str,
    awaitable,
    *,
    timeout_seconds: float,
    clock: Callable[[], float],
):
    started = clock()
    try:
        value = await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except TimeoutError as error:
        raise _StageFailure(stage, "timeout") from error
    except Exception as error:
        raise _StageFailure(stage, "adapter-error") from error
    return value, max(0.0, clock() - started)


def _validate_case(case: ShadowCase) -> str | None:
    if not SAFE_CASE_ID.fullmatch(case.case_id):
        return "unsafe-case-id"
    if not case.fragments:
        return "empty-corpus-case"
    if (
        case.completed_turn_snapshot is not None
        and (
            not isinstance(case.completed_turn_snapshot, bytes)
            or not case.completed_turn_snapshot
        )
    ):
        return "invalid-completed-turn-snapshot"
    previous = -1.0
    for fragment in case.fragments:
        if (
            not isinstance(fragment.audio, bytes)
            or not fragment.audio
            or not math.isfinite(fragment.at_seconds)
            or fragment.at_seconds < 0
            or fragment.at_seconds < previous
        ):
            return "invalid-corpus-fragment"
        previous = fragment.at_seconds
    return None


def _adapter_contract_error(asr, model, tts, sink) -> str | None:
    contracts = (
        (getattr(asr, "input_mode", None), "explicit-corpus", "unsafe-asr"),
        (
            getattr(model, "mode", None),
            "isolated-ephemeral",
            "non-ephemeral-model",
        ),
        (getattr(tts, "output_mode", None), "pcm-bytes", "unsafe-tts"),
        (getattr(sink, "mode", None), "discard-hash", "unsafe-sink"),
    )
    return next(
        (code for actual, expected, code in contracts if actual != expected),
        None,
    )


async def _run_case(
    case: ShadowCase,
    *,
    asr: CorpusAsr,
    model: EphemeralModel,
    tts: PcmSynthesizer,
    sink: DiscardSink,
    timeout_seconds: float,
    clock: Callable[[], float],
) -> _CaseOutcome:
    sequence: list[str] = []
    latencies: dict[str, float] = {}
    case_started = clock()
    owner = TurnOwner()
    committed = None
    asr_total = 0.0
    secrets: list[str] = []

    try:
        for index, fragment in enumerate(case.fragments):
            transcript, elapsed = await _timed(
                "asr",
                asr.transcribe(fragment.audio),
                timeout_seconds=timeout_seconds,
                clock=clock,
            )
            asr_total += elapsed
            if not isinstance(transcript, str):
                raise _StageFailure("asr", "invalid-output")
            secrets.append(transcript)
            decision = owner.observe(transcript, now=fragment.at_seconds)
            if decision.action == "submit":
                if index != len(case.fragments) - 1:
                    raise _StageFailure("turn_owner", "premature-commit")
                committed = decision

        latencies["asr_seconds"] = round(asr_total, 6)
        sequence.append("asr")
        owner_started = clock()
        if committed is None:
            if owner.deadline is None:
                raise _StageFailure("turn_owner", "missing-commit")
            committed = owner.due(now=owner.deadline)
        latencies["turn_owner_seconds"] = round(
            max(0.0, clock() - owner_started),
            6,
        )
        if committed.action != "submit" or not committed.text:
            raise _StageFailure("turn_owner", "missing-commit")
        sequence.append("turn_owner")

        final_canonical = canonical(committed.text)
        if not final_canonical:
            raise _StageFailure("turn_owner", "empty-final")

        response, elapsed = await _timed(
            "model",
            model.generate(
                committed.text,
                completed_turn_snapshot=case.completed_turn_snapshot,
            ),
            timeout_seconds=timeout_seconds,
            clock=clock,
        )
        if not isinstance(response, str):
            raise _StageFailure("model", "invalid-output")
        if not response.strip():
            raise _StageFailure("model", "empty-model-output")
        secrets.append(response)
        latencies["model_seconds"] = round(elapsed, 6)
        sequence.append("model")

        pcm, elapsed = await _timed(
            "tts",
            tts.synthesize(response),
            timeout_seconds=timeout_seconds,
            clock=clock,
        )
        if not isinstance(pcm, bytes) or not pcm:
            raise _StageFailure("tts", "invalid-output")
        latencies["tts_seconds"] = round(elapsed, 6)
        sequence.append("tts")

        digest, elapsed = await _timed(
            "sink",
            sink.consume(pcm),
            timeout_seconds=timeout_seconds,
            clock=clock,
        )
        if not isinstance(digest, str) or not SAFE_DIGEST.fullmatch(digest):
            raise _StageFailure("sink", "invalid-digest")
        latencies["sink_seconds"] = round(elapsed, 6)
        sequence.append("sink")
        latencies["total_seconds"] = round(
            max(0.0, clock() - case_started),
            6,
        )
        if tuple(sequence) != EXPECTED_STAGES:
            raise _StageFailure("sequence", "missing-stage")
        return _CaseOutcome(
            {
                "case_id": case.case_id,
                "ok": True,
                "stage_sequence": sequence,
                "latencies": latencies,
                "pcm": {"bytes": len(pcm), "sha256": digest},
            },
            final_canonical,
            tuple(secrets),
        )
    except _StageFailure as failure:
        owner.cancel()
        return _CaseOutcome(
            _safe_failure(
                case.case_id,
                stage=failure.stage,
                code=failure.code,
                sequence=sequence,
                latencies=latencies,
            ),
            secrets=tuple(secrets),
        )


def _minimal_leak_report(total_cases: int) -> dict[str, Any]:
    return {
        "schema": 1,
        "pipeline": SHADOW_PIPELINE,
        "status": "failed",
        "counts": {
            "cases": total_cases,
            "completed": 0,
            "failed": total_cases,
            "duplicates": 0,
            "timeouts": 0,
            "adapter_errors": 0,
            "empty_model_outputs": 0,
            "text_leaks": 1,
            "context_cases": 0,
        },
        "latencies": {},
        "promotion": {
            "eligible": False,
            "verdict": "reject",
            "required_cases": MINIMUM_PROMOTION_CASES,
            "completed_cases": 0,
            "observed_completed_turns": 0,
        },
        "cases": [],
    }


async def run_shadow(
    corpus: tuple[ShadowCase, ...],
    *,
    asr: CorpusAsr,
    model: EphemeralModel,
    tts: PcmSynthesizer,
    sink: DiscardSink | None = None,
    timeout_seconds: float = 10.0,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate explicit audio without production I/O or authoritative context."""
    sink = sink or HashDiscardSink()
    cases: list[dict[str, Any]] = []
    canonical_finals: set[str] = set()
    secrets: list[str] = []
    duplicates = 0

    contract_error = _adapter_contract_error(asr, model, tts, sink)
    case_ids = [case.case_id for case in corpus]
    duplicate_ids = len(case_ids) - len(set(case_ids))
    if contract_error or duplicate_ids or timeout_seconds <= 0:
        code = contract_error or (
            "duplicate-case-id" if duplicate_ids else "invalid-timeout"
        )
        cases = [
            _safe_failure(
                case.case_id if SAFE_CASE_ID.fullmatch(case.case_id) else "invalid",
                stage="configuration",
                code=code,
            )
            for case in corpus
        ]
    else:
        for case in corpus:
            validation_error = _validate_case(case)
            if validation_error:
                outcome = _CaseOutcome(
                    _safe_failure(
                        case.case_id if SAFE_CASE_ID.fullmatch(case.case_id) else "invalid",
                        stage="corpus",
                        code=validation_error,
                    )
                )
            else:
                outcome = await _run_case(
                    case,
                    asr=asr,
                    model=model,
                    tts=tts,
                    sink=sink,
                    timeout_seconds=timeout_seconds,
                    clock=clock,
                )
            if outcome.canonical_final:
                if outcome.canonical_final in canonical_finals:
                    duplicates += 1
                    outcome = _CaseOutcome(
                        _safe_failure(
                            case.case_id,
                            stage="turn_owner",
                            code="duplicate-final",
                            sequence=outcome.report.get("stage_sequence"),
                            latencies=outcome.report.get("latencies"),
                        )
                    )
                else:
                    canonical_finals.add(outcome.canonical_final)
            secrets.extend(outcome.secrets)
            cases.append(outcome.report)

    completed = sum(case.get("ok") is True for case in cases)
    failed = len(cases) - completed
    codes = [
        case.get("failure", {}).get("code")
        for case in cases
        if case.get("failure")
    ]
    latency_fields = sorted(
        {
            field
            for case in cases
            if case.get("ok")
            for field in case.get("latencies", {})
        }
    )
    latencies = {
        field: _summary(
            [
                float(case["latencies"][field])
                for case in cases
                if case.get("ok") and field in case.get("latencies", {})
            ]
        )
        for field in latency_fields
    }
    cohort_complete = len(corpus) >= MINIMUM_PROMOTION_CASES
    if failed or duplicates:
        status = "failed"
    elif cohort_complete and completed == len(corpus):
        status = "passed"
    else:
        status = "collecting"
    report = {
        "schema": 1,
        "pipeline": SHADOW_PIPELINE,
        "status": status,
        "counts": {
            "cases": len(corpus),
            "completed": completed,
            "failed": failed,
            "duplicates": duplicates + duplicate_ids,
            "timeouts": codes.count("timeout"),
            "adapter_errors": codes.count("adapter-error"),
            "empty_model_outputs": codes.count("empty-model-output"),
            "text_leaks": 0,
            "context_cases": sum(
                case.completed_turn_snapshot is not None for case in corpus
            ),
        },
        "latencies": latencies,
        "promotion": {
            "eligible": status == "passed",
            "verdict": (
                "promote"
                if status == "passed"
                else "reject"
                if status == "failed"
                else "hold"
            ),
            "required_cases": MINIMUM_PROMOTION_CASES,
            "completed_cases": completed,
            "observed_completed_turns": completed,
        },
        "cases": cases,
    }
    try:
        assert_transcript_free(report, secrets=tuple(secrets))
    except TranscriptLeakError:
        return _minimal_leak_report(len(corpus))
    return report


def dumps(report: dict[str, Any]) -> str:
    assert_transcript_free(report)
    return json.dumps(report, separators=(",", ":"), sort_keys=True)
