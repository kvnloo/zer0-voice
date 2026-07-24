#!/usr/bin/env python3
"""Run the hermetic voice shadow gate against real local candidate adapters.

There are deliberately no microphone, audio-output, session, thread, or shared
daemon options in this program.  Corpus and response audio live only in memory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "voice"),
    str(ROOT / "contracts"),
    str(ROOT / "adapters" / "codex"),
    str(ROOT / "adapters" / "llm"),
]

from app_server import CodexAppServer
from conversation import decode_wav, kokoro_wav, transcribe
from providers import CodexSubscription
from shadow import (
    HashDiscardSink,
    MINIMUM_PROMOTION_CASES,
    ShadowCase,
    ShadowFragment,
    dumps,
    run_shadow,
)


# Fixed, synthetic, punctuation-complete turns.  They are source material, not
# user data, and are never emitted in the privacy-safe report.
SYNTHETIC_TURNS = (
    "Confirm the first isolated shadow check.",
    "Summarize why deterministic ownership matters.",
    "Name one benefit of a discard hash sink.",
    "Explain the purpose of an ephemeral thread.",
    "Confirm that no microphone is connected.",
    "Describe one safe promotion invariant.",
    "State why empty model output must fail.",
    "Confirm that response audio is never played.",
    "Explain why duplicate final turns are rejected.",
    "Finish the tenth synthetic shadow check.",
    "Describe a bounded latency measurement.",
    "Confirm that production remains unchanged.",
)

SHADOW_INSTRUCTIONS = """\
You are the isolated Zer0 voice shadow candidate.
Reply to each synthetic test turn with exactly one short natural sentence.
Do not call tools, alter files, or interact with any production conversation.
"""


class FasterWhisperCorpusAsr:
    """Candidate Faster Whisper decoder accepting explicit in-memory WAV only."""

    input_mode = "explicit-corpus"

    def __init__(self, model) -> None:
        self.model = model

    async def transcribe(self, audio: bytes) -> str:
        samples, rate = decode_wav(audio)
        if rate != 16_000:
            samples = _resample(samples, rate, 16_000)
        return await asyncio.to_thread(
            transcribe,
            self.model,
            samples,
            "en",
            accurate=True,
        )


class EphemeralCodexModel:
    """One fresh ephemeral thread per turn on a private app-server process."""

    mode = "isolated-ephemeral"

    def __init__(
        self,
        server: CodexAppServer,
        cwd: Path,
        model: str,
        effort: str,
    ) -> None:
        if server.shared:
            raise ValueError("shadow model requires a private app-server")
        self.server = server
        self.cwd = cwd
        self.model = model
        self.effort = effort

    async def generate(
        self,
        prompt: str,
        *,
        completed_turn_snapshot: bytes | None,
    ) -> str:
        # Snapshots are explicit synthetic corpus data, never an authoritative
        # thread identifier. Decode strictly and place them in this one request.
        context: tuple[str, ...] = ()
        if completed_turn_snapshot is not None:
            context = (
                completed_turn_snapshot.decode("utf-8", errors="strict"),
            )
        thread = await self.server.start_thread(
            cwd=self.cwd,
            model=self.model,
            developer_instructions=SHADOW_INSTRUCTIONS,
            ephemeral=True,
        )
        pieces: list[str] = []
        completed = False
        async for event in CodexSubscription(self.server, thread).stream(
            prompt,
            context=context,
            effort=self.effort,
        ):
            if event.kind == "delta":
                pieces.append(event.text)
            elif event.kind == "completed":
                completed = True
        answer = "".join(pieces).strip()
        if not completed or not answer:
            raise RuntimeError("ephemeral model returned no completed text")
        return answer


class KokoroPcmSynthesizer:
    """Candidate Kokoro synthesis returning in-memory WAV/PCM bytes."""

    output_mode = "pcm-bytes"

    def __init__(self, base_url: str, voice: str) -> None:
        self.base_url = base_url
        self.voice = voice

    async def synthesize(self, text: str) -> bytes:
        audio = await asyncio.to_thread(
            kokoro_wav,
            text,
            base_url=self.base_url,
            voice=self.voice,
        )
        # Decode once to prove the payload is nonempty PCM without persisting it.
        samples, rate = decode_wav(audio)
        if rate <= 0 or samples.size == 0:
            raise RuntimeError("Kokoro returned empty PCM")
        return audio


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or samples.size == 0:
        raise ValueError("invalid source audio")
    target_size = max(1, round(samples.size * target_rate / source_rate))
    source = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    target = np.linspace(0.0, 1.0, target_size, endpoint=False)
    return np.interp(target, source, samples).astype(np.float32)


async def build_synthetic_corpus(
    *,
    kokoro_url: str,
    voice: str,
    cases: int,
) -> tuple[ShadowCase, ...]:
    if cases < MINIMUM_PROMOTION_CASES or cases > len(SYNTHETIC_TURNS):
        raise ValueError(
            f"cases must be {MINIMUM_PROMOTION_CASES}..{len(SYNTHETIC_TURNS)}"
        )

    async def synthesize(index: int, turn: str) -> ShadowCase:
        wav = await asyncio.to_thread(
            kokoro_wav,
            turn,
            base_url=kokoro_url,
            voice=voice,
        )
        samples, rate = decode_wav(wav)
        if rate <= 0 or samples.size == 0:
            raise RuntimeError("synthetic corpus generation returned empty PCM")
        return ShadowCase(
            case_id=f"synthetic-{index + 1:02d}",
            fragments=(ShadowFragment(audio=wav, at_seconds=0.0),),
            completed_turn_snapshot=(
                b"Synthetic prior context: answer briefly."
                if index % 3 == 2
                else None
            ),
        )

    # Bound generation concurrency to avoid competing Kokoro GPU requests.
    return tuple(
        [
            await synthesize(index, turn)
            for index, turn in enumerate(SYNTHETIC_TURNS[:cases])
        ]
    )


async def real_shadow(args: argparse.Namespace) -> dict[str, object]:
    from faster_whisper import WhisperModel

    corpus = await build_synthetic_corpus(
        kokoro_url=args.kokoro_url,
        voice=args.voice,
        cases=args.cases,
    )
    whisper_started = time.perf_counter()
    whisper = await asyncio.to_thread(
        WhisperModel,
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=True,
    )
    load_seconds = round(time.perf_counter() - whisper_started, 6)

    # shared=False starts a new stdio app-server subprocess. No production
    # daemon connection, existing thread ID, or harness context is reachable.
    async with CodexAppServer(
        cwd=ROOT,
        shared=False,
        startup_timeout=args.startup_timeout,
    ) as server:
        report = await run_shadow(
            corpus,
            asr=FasterWhisperCorpusAsr(whisper),
            model=EphemeralCodexModel(server, ROOT, args.model, args.effort),
            tts=KokoroPcmSynthesizer(args.kokoro_url, args.voice),
            sink=HashDiscardSink(),
            timeout_seconds=args.case_timeout,
        )
    report["setup"] = {
        "corpus": "synthetic-in-memory",
        "asr": "faster-whisper-local",
        "model": "private-app-server-ephemeral-luna",
        "tts": "kokoro-in-memory",
        "sink": "discard-hash",
        "whisper_load_seconds": load_seconds,
        "production_resources": 0,
    }
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cases", type=int, default=MINIMUM_PROMOTION_CASES)
    result.add_argument("--whisper-model", default="small.en")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--compute-type", default="float16")
    result.add_argument("--kokoro-url", default="http://127.0.0.1:8880")
    result.add_argument("--voice", default="af_heart")
    result.add_argument("--model", default="gpt-5.6-luna")
    result.add_argument("--effort", default="low")
    result.add_argument("--startup-timeout", type=float, default=15.0)
    result.add_argument("--case-timeout", type=float, default=30.0)
    result.add_argument(
        "--report",
        type=Path,
        help="atomically persist the transcript-free JSON report",
    )
    return result


def atomic_report(path: Path, report: dict[str, object]) -> None:
    payload = dumps(report).encode() + b"\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parser().parse_args()
    try:
        report = asyncio.run(real_shadow(args))
    except Exception as error:
        # Adapter exception text may include candidate data; never serialize it.
        report = {
            "schema": 1,
            "pipeline": "voice-shadow-v1-real",
            "status": "failed",
            "failure": {"stage": "setup", "code": type(error).__name__},
            "promotion": {"eligible": False, "verdict": "reject"},
        }
    if args.report:
        atomic_report(args.report, report)
    print(dumps(report))
    return 0 if report.get("promotion", {}).get("eligible") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
