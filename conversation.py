#!/usr/bin/env python3
"""Hands-free local voice loop around a persistent Codex conversation."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

KOKORO_URL = os.getenv("ZERO_KOKORO_URL", "http://127.0.0.1:8880")
SAMPLE_RATE = 16_000
BLOCK_MS = 30


@dataclass(frozen=True)
class ListenConfig:
    sample_rate: int = SAMPLE_RATE
    block_ms: int = BLOCK_MS
    threshold: float = 0.018
    start_blocks: int = 2
    silence_ms: int = 650
    max_seconds: float = 45.0
    pre_roll_ms: int = 240

    @property
    def block_size(self) -> int:
        return self.sample_rate * self.block_ms // 1000

    @property
    def silence_blocks(self) -> int:
        return max(1, self.silence_ms // self.block_ms)

    @property
    def pre_roll_blocks(self) -> int:
        return max(1, self.pre_roll_ms // self.block_ms)


def rms(block: np.ndarray) -> float:
    samples = block.astype(np.float32, copy=False)
    return math.sqrt(float(np.mean(samples * samples))) if samples.size else 0.0


def segment_blocks(
    blocks: Iterable[np.ndarray], config: ListenConfig
) -> np.ndarray | None:
    """Return the first energy-delimited utterance from a finite block stream."""
    before: list[np.ndarray] = []
    utterance: list[np.ndarray] = []
    voiced = silent = 0
    active = False
    max_blocks = int(config.max_seconds * 1000 / config.block_ms)

    for block in blocks:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        is_voice = rms(block) >= config.threshold
        if not active:
            before.append(block)
            before = before[-config.pre_roll_blocks :]
            voiced = voiced + 1 if is_voice else 0
            if voiced >= config.start_blocks:
                active = True
                utterance.extend(before)
        else:
            utterance.append(block)
            silent = 0 if is_voice else silent + 1
            if silent >= config.silence_blocks or len(utterance) >= max_blocks:
                keep = max(0, len(utterance) - silent)
                return np.concatenate(utterance[:keep]) if keep else None
    return None


def listen(config: ListenConfig, stop: threading.Event | None = None) -> np.ndarray | None:
    """Capture one utterance. A stop event enables playback barge-in."""
    import sounddevice as sd

    blocks: queue.Queue[np.ndarray] = queue.Queue()
    stop = stop or threading.Event()

    def callback(indata: np.ndarray, _frames: int, _time: Any, status: Any) -> None:
        if status:
            print(f"\r[audio: {status}]", file=sys.stderr)
        blocks.put(indata[:, 0].copy())

    def source() -> Iterable[np.ndarray]:
        while not stop.is_set():
            yield blocks.get()

    with sd.InputStream(
        samplerate=config.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=config.block_size,
        callback=callback,
    ):
        return segment_blocks(source(), config)


def transcribe(
    model: "WhisperModel",
    audio: np.ndarray,
    language: str = "en",
    *,
    lexicon: Any | None = None,
    accurate: bool = False,
) -> str:
    """Decode quickly for drafts or carefully for a committed utterance."""
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=3 if accurate else 1,
        best_of=3 if accurate else 1,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
        initial_prompt=lexicon.prompt if lexicon else None,
        hotwords=lexicon.hotwords if lexicon else None,
    )
    credible = (
        segment
        for segment in segments
        if not (
            (getattr(segment, "no_speech_prob", 0.0) or 0.0) > 0.60
            and (getattr(segment, "avg_logprob", 0.0) or 0.0) < -1.0
        )
    )
    text = " ".join(segment.text.strip() for segment in credible).strip()
    return lexicon.correct(text) if lexicon else text


def codex_turn(
    prompt: str,
    *,
    session: str | None,
    cwd: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt") as output:
        command = ["codex", "exec", "resume"]
        if session:
            command.append(session)
        else:
            command.append("--last")
        command.extend(
            [
                "--skip-git-repo-check",
                "-c",
                'model_verbosity="low"',
                "-o",
                output.name,
                prompt,
            ]
        )
        result = runner(command, cwd=cwd, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"Codex exited with status {result.returncode}")
        return Path(output.name).read_text(encoding="utf-8").strip()


def kokoro_wav(
    text: str,
    *,
    base_url: str = KOKORO_URL,
    voice: str = "af_heart",
    timeout: float = 120.0,
) -> bytes:
    body = json.dumps(
        {
            "model": "kokoro",
            "voice": voice,
            "input": text,
            "response_format": "wav",
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def decode_wav(audio: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got sample width {width}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, rate


def speak(text: str, *, base_url: str, voice: str) -> None:
    import sounddevice as sd

    samples, rate = decode_wav(kokoro_wav(text, base_url=base_url, voice=voice))
    sd.play(samples, rate)
    sd.wait()


def health(base_url: str) -> None:
    import sounddevice as sd

    with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=2) as response:
        status = json.load(response)
    if status.get("status") != "healthy":
        raise RuntimeError(f"Kokoro is not healthy: {status}")
    sd.check_input_settings(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.check_output_settings()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="Codex session UUID/name; defaults to --last")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--whisper-model", default="small.en")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--kokoro-url", default=KOKORO_URL)
    parser.add_argument("--threshold", type=float, default=0.018)
    parser.add_argument("--silence-ms", type=int, default=650)
    parser.add_argument("--once", action="store_true", help="process one utterance and exit")
    args = parser.parse_args()

    try:
        health(args.kokoro_url)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"voice health check failed: {exc}", file=sys.stderr)
        return 2

    print(f"Loading Whisper {args.whisper_model} on {args.device}…", flush=True)
    from faster_whisper import WhisperModel

    model = WhisperModel(
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=True,
    )
    config = ListenConfig(threshold=args.threshold, silence_ms=args.silence_ms)
    print("Ready. Speak naturally; Ctrl-C exits.", flush=True)

    try:
        while True:
            print("\nListening…", flush=True)
            audio = listen(config)
            if audio is None:
                continue
            started = time.monotonic()
            prompt = transcribe(model, audio)
            if not prompt:
                continue
            print(f"You: {prompt}", flush=True)
            response = codex_turn(prompt, session=args.session, cwd=args.cwd)
            print(f"Codex: {response}", flush=True)
            print(f"turn latency before speech: {time.monotonic() - started:.2f}s", flush=True)
            speak(response, base_url=args.kokoro_url, voice=args.voice)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
