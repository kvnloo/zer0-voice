#!/usr/bin/env python3
"""Real local ASR -> Codex stream -> TTS smoke with latency output."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [
    str(HERE.parent),
    str(HERE.parents[1] / "contracts"),
    str(HERE.parents[1] / "adapters/codex"),
]

from app_server import CodexAppServer
from conversation import decode_wav, kokoro_wav, transcribe
from orchestrator import OllamaLiveLane


async def run() -> dict:
    from faster_whisper import WhisperModel

    timings = {}
    started = time.perf_counter()
    model = WhisperModel(
        "small.en", device="cuda", compute_type="float16", local_files_only=True
    )
    timings["asr_model_load_ms"] = round((time.perf_counter() - started) * 1000, 1)

    started = time.perf_counter()
    user_wav = await asyncio.to_thread(
        kokoro_wav, "Say exactly, full duplex is connected."
    )
    audio, _ = decode_wav(user_wav)
    timings["input_synthesis_ms"] = round((time.perf_counter() - started) * 1000, 1)

    started = time.perf_counter()
    prompt = await asyncio.to_thread(transcribe, model, audio, "en")
    timings["asr_ms"] = round((time.perf_counter() - started) * 1000, 1)

    live = OllamaLiveLane()
    live_deltas = []
    deep_deltas = []
    live_first = deep_first = None
    parallel_started = time.perf_counter()

    async def run_live():
        nonlocal live_first
        async for delta in live.stream(prompt, ()):
            if live_first is None:
                live_first = time.perf_counter()
            live_deltas.append(delta)

    async with CodexAppServer(cwd=HERE.parents[1]) as server:
        thread = await server.start_thread(
            developer_instructions=(
                "You are the deep lane. Reply with exactly: Deep lane is connected."
            )
        )

        async def run_deep():
            nonlocal deep_first
            async for event in server.stream_turn(thread, prompt, effort="low"):
                if event.kind == "assistant.delta":
                    if deep_first is None:
                        deep_first = time.perf_counter()
                    deep_deltas.append(str(event.payload["text"]))

        await asyncio.gather(run_live(), run_deep())
    timings["live_first_delta_ms"] = round(
        (live_first - parallel_started) * 1000, 1
    )
    timings["deep_first_delta_ms"] = round(
        (deep_first - parallel_started) * 1000, 1
    )
    timings["parallel_complete_ms"] = round(
        (time.perf_counter() - parallel_started) * 1000, 1
    )
    response = "".join(live_deltas)
    deep_response = "".join(deep_deltas)

    started = time.perf_counter()
    output_wav = await asyncio.to_thread(kokoro_wav, response)
    timings["output_synthesis_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return {
        "ok": bool(prompt and response and deep_response and len(output_wav) > 44),
        "transcript": prompt,
        "live_response": response,
        "deep_response": deep_response,
        "output_bytes": len(output_wav),
        "timings": timings,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), indent=2))
