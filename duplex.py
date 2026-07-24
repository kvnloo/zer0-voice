#!/usr/bin/env python3
"""Full-duplex local voice runtime: mic -> Whisper -> Codex -> Kokoro."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np

HERE = Path(__file__).resolve()
sys.path[:0] = [
    str(HERE.parents[1] / "contracts"),
    str(HERE.parents[1] / "adapters/codex"),
]

from app_server import AppServerError, CodexAppServer
from conversation import ListenConfig, kokoro_wav, rms, transcribe
from events import Event
from preflight import preflight
from workspace_router import Route, WorkspaceRouter, load_routes


@dataclass(frozen=True)
class AudioEvent:
    kind: str
    audio: np.ndarray | None = None


def append_metric(path: Path, metric: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, separators=(",", ":"), sort_keys=True) + "\n")


def publish_transcript(url: str, thread: str, text: str, ts_ns: int) -> None:
    event = Event(
        source="codex.voice",
        kind="voice.transcript.final",
        subject=f"voice:{thread}:{ts_ns}",
        payload={"text": text, "thread": thread},
        ts_ns=ts_ns,
    )
    request = urllib.request.Request(
        url,
        data=event.json().encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=0.5) as response:
        if response.status >= 300:
            raise RuntimeError(f"event relay returned HTTP {response.status}")


async def publish_transcript_best_effort(
    url: str, thread: str, text: str, ts_ns: int
) -> None:
    try:
        await asyncio.to_thread(publish_transcript, url, thread, text, ts_ns)
    except (OSError, RuntimeError) as error:
        print(f"Event relay unavailable: {error}", flush=True)


def voice_control(text: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    stop = re.fullmatch(
        r"(?:(?:can|could|would) you )?"
        r"(?:please )?"
        r"(?:stop|exit|quit)"
        r"(?: (?:voice|voice mode|listening|conversation mode))?"
        r"(?: please)?",
        normalized,
    )
    return "stop" if stop else None


class UtteranceDetector:
    """Stateful endpoint detector that reports speech start before final audio."""

    def __init__(self, config: ListenConfig, *, min_speech_ms: int = 180):
        self.config = config
        self.min_voiced_blocks = max(
            config.start_blocks,
            (min_speech_ms + config.block_ms - 1) // config.block_ms,
        )
        self.before: deque[np.ndarray] = deque(maxlen=config.pre_roll_blocks)
        self.utterance: list[np.ndarray] = []
        self.voiced = 0
        self.active_voiced = 0
        self.silent = 0
        self.active = False
        self.confirmed = False

    def push(self, block: np.ndarray) -> list[AudioEvent]:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        voice = rms(block) >= self.config.threshold
        events: list[AudioEvent] = []
        if not self.active:
            self.before.append(block)
            self.voiced = self.voiced + 1 if voice else 0
            if not voice:
                self.voiced = 0
            if self.voiced >= self.config.start_blocks:
                self.active = True
                self.utterance = list(self.before)
                self.active_voiced = self.voiced
                self.silent = 0
                events.append(AudioEvent("speech.started"))
                if self.active_voiced >= self.min_voiced_blocks:
                    self.confirmed = True
                    events.append(AudioEvent("speech.confirmed"))
            return events

        self.utterance.append(block)
        if voice:
            self.active_voiced += 1
            if (
                not self.confirmed
                and self.active_voiced >= self.min_voiced_blocks
            ):
                self.confirmed = True
                events.append(AudioEvent("speech.confirmed"))
        self.silent = 0 if voice else self.silent + 1
        max_blocks = int(self.config.max_seconds * 1000 / self.config.block_ms)
        if self.silent >= self.config.silence_blocks or len(self.utterance) >= max_blocks:
            keep = max(0, len(self.utterance) - self.silent)
            audio = np.concatenate(self.utterance[:keep]) if keep else None
            voiced_blocks = self.active_voiced
            self.reset()
            if audio is not None and voiced_blocks >= self.min_voiced_blocks:
                events.append(AudioEvent("speech.final", audio))
        return events

    def reset(self) -> None:
        self.before.clear()
        self.utterance = []
        self.voiced = self.active_voiced = self.silent = 0
        self.active = False
        self.confirmed = False


def adaptive_threshold(levels: np.ndarray) -> float:
    """Choose a conservative speech threshold from a short ambient sample."""
    values = np.asarray(levels, dtype=np.float32).reshape(-1)
    if not values.size:
        return 0.018
    # Median rejects a cough, keypress, or startup transient in the short sample.
    noise = float(np.median(values))
    return min(0.02, max(0.004, noise * 2.5))


def calibrate_microphone(
    *,
    device: str | int | None,
    sample_rate: int,
    block_size: int,
    seconds: float,
) -> tuple[float, str]:
    import sounddevice as sd

    info = sd.query_devices(device, "input")
    frames = max(block_size, int(sample_rate * seconds))
    sample = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        blocking=True,
    )
    levels = np.asarray(
        [
            rms(sample[start : start + block_size, 0])
            for start in range(0, sample.shape[0], block_size)
        ]
    )
    return adaptive_threshold(levels), str(info["name"])


class SentenceChunker:
    """Turn token deltas into low-latency, natural TTS units."""

    boundary = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+")

    def __init__(self, max_chars: int = 180):
        self.buffer = ""
        self.max_chars = max_chars

    def feed(self, delta: str) -> list[str]:
        self.buffer += delta
        chunks: list[str] = []
        while True:
            match = self.boundary.search(self.buffer)
            if match:
                chunks.append(self.buffer[: match.end()].strip())
                self.buffer = self.buffer[match.end() :]
                continue
            if len(self.buffer) >= self.max_chars:
                split = self.buffer.rfind(" ", 0, self.max_chars)
                if split > 0:
                    chunks.append(self.buffer[:split].strip())
                    self.buffer = self.buffer[split + 1 :]
                    continue
            return chunks

    def flush(self) -> str:
        tail, self.buffer = self.buffer.strip(), ""
        return tail


class Microphone:
    def __init__(self, config: ListenConfig, *, device: str | int | None = None):
        self.config = config
        self.device = device
        self.detector = UtteranceDetector(config)
        self.queue: asyncio.Queue[AudioEvent] = asyncio.Queue()
        self.stream = None
        self.loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        import sounddevice as sd

        self.loop = asyncio.get_running_loop()

        def callback(indata, _frames, _time, status):
            if status:
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait, AudioEvent("audio.warning")
                )
            for event in self.detector.push(indata[:, 0].copy()):
                self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.config.block_size,
            device=self.device,
            callback=callback,
        )
        self.stream.start()

    def close(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()

    async def next(self) -> AudioEvent:
        return await self.queue.get()


class Speaker:
    def __init__(
        self,
        base_url: str,
        voice: str,
        *,
        backend: str = "pipewire",
        output: str | None = None,
        latency: str = "40ms",
    ):
        self.base_url = base_url
        self.voice = voice
        self.backend = backend
        self.output = output
        self.latency = latency
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._player: subprocess.Popen | None = None
        self.first_play_at: float | None = None
        self.first_synthesis_seconds: float | None = None

    def interrupt(self) -> None:
        self.cancel.set()
        with self._lock:
            player = self._player
        if player and player.poll() is None:
            player.terminate()
        if self.backend == "sounddevice":
            import sounddevice as sd

            sd.stop()

    def _pipewire_play(self, wav: bytes) -> None:
        command = ["pw-play", "--latency", self.latency]
        if self.output:
            command.extend(("--target", self.output))
        command.append("-")
        player = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        with self._lock:
            if self.cancel.is_set():
                player.terminate()
            self._player = player
        _, stderr = player.communicate(wav)
        with self._lock:
            if self._player is player:
                self._player = None
        if player.returncode and not self.cancel.is_set():
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"PipeWire playback failed: {detail or player.returncode}")

    def _pipewire_stream(self, text: str, synthesis_started: float) -> None:
        body = json.dumps(
            {
                "model": "kokoro",
                "voice": self.voice,
                "input": text,
                "response_format": "pcm",
                "stream": True,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/audio/speech",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        command = [
            "pw-play",
            "--raw",
            "--format",
            "s16",
            "--rate",
            "24000",
            "--channels",
            "1",
            "--latency",
            self.latency,
        ]
        if self.output:
            command.extend(("--target", self.output))
        command.append("-")
        player = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        with self._lock:
            if self.cancel.is_set():
                player.terminate()
            self._player = player
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                while not self.cancel.is_set():
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    if self.first_synthesis_seconds is None:
                        now = time.monotonic()
                        self.first_synthesis_seconds = now - synthesis_started
                        self.first_play_at = now
                    assert player.stdin
                    player.stdin.write(chunk)
                    player.stdin.flush()
        except BrokenPipeError:
            if not self.cancel.is_set():
                raise
        finally:
            if player.stdin:
                player.stdin.close()
            player.wait()
            stderr = player.stderr.read() if player.stderr else b""
            with self._lock:
                if self._player is player:
                    self._player = None
        if player.returncode and not self.cancel.is_set():
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"PipeWire playback failed: {detail or player.returncode}")

    async def say(self, text: str) -> None:
        if self.cancel.is_set() or not text:
            return
        synthesis_started = time.monotonic()
        if self.backend == "pipewire":
            await asyncio.to_thread(
                self._pipewire_stream,
                text,
                synthesis_started,
            )
            return
        audio = await asyncio.to_thread(
            kokoro_wav, text, base_url=self.base_url, voice=self.voice
        )
        if self.first_synthesis_seconds is None:
            self.first_synthesis_seconds = time.monotonic() - synthesis_started
        if self.cancel.is_set():
            return
        if self.first_play_at is None:
            self.first_play_at = time.monotonic()
        import sounddevice as sd
        from conversation import decode_wav

        samples, rate = decode_wav(audio)
        await asyncio.to_thread(sd.play, samples, rate)
        await asyncio.to_thread(sd.wait)


class DuplexTurn:
    def __init__(self, server: CodexAppServer, thread: str, speaker: Speaker):
        self.server = server
        self.thread = thread
        self.speaker = speaker
        self.turn_id: str | None = None

    async def run(self, text: str, effort: str) -> str:
        chunks: asyncio.Queue[str | None] = asyncio.Queue()
        response: list[str] = []

        async def generate() -> None:
            chunker = SentenceChunker()
            async for event in self.server.stream_turn(self.thread, text, effort=effort):
                self.turn_id = event.subject.removeprefix("turn:")
                if event.kind == "assistant.delta":
                    delta = str(event.payload["text"])
                    response.append(delta)
                    for chunk in chunker.feed(delta):
                        await chunks.put(chunk)
            if tail := chunker.flush():
                await chunks.put(tail)
            await chunks.put(None)

        async def speak() -> None:
            while (chunk := await chunks.get()) is not None:
                await self.speaker.say(chunk)

        await asyncio.gather(generate(), speak())
        return "".join(response).strip()

    async def interrupt(self) -> None:
        self.speaker.interrupt()
        if self.turn_id:
            await self.server.interrupt(self.thread, self.turn_id)


@dataclass(frozen=True, slots=True)
class ThreadBinding:
    key: str
    thread: str
    route: Route | None
    reason: str


class HarnessRouter:
    """Resolve workspace focus, then attach one persistent Codex thread per route."""

    def __init__(
        self,
        server: CodexAppServer,
        workspace: WorkspaceRouter | None,
        fallback_thread: str,
        *,
        timeout: float,
    ):
        self.server = server
        self.workspace = workspace
        self.fallback_thread = fallback_thread
        self.timeout = timeout
        self.bindings: dict[str, ThreadBinding] = {}
        self.pinned_project: str | None = None

    def _voice_route(self, text: str) -> Route | None:
        if self.workspace is None:
            return None
        normalized = re.sub(r"[^a-z0-9.]+", " ", text.lower()).strip()
        if re.search(r"\b(?:follow|use) (?:the )?focus\b", normalized):
            self.pinned_project = None
            print("Voice routing returned to workspace focus.", flush=True)
            return None
        aliases = {
            "pm": "pm",
            "product manager": "pm",
            "zeros": "zerOS",
            "zero s": "zerOS",
            "flowkit": "flowkit",
            "dotfiles": ".files",
            "files": ".files",
        }
        command = re.search(
            r"\b(?:switch|route|talk)(?: me)? to ([a-z0-9. ]+?)(?: please)?$",
            normalized,
        )
        if command:
            requested = aliases.get(command.group(1).strip())
            if requested and requested in self.workspace.routes:
                self.pinned_project = requested
                print(f"Voice routing pinned to {requested}.", flush=True)
        if self.pinned_project:
            return Route(
                project=self.pinned_project,
                cwd=self.workspace.routes[self.pinned_project],
                pane_id="voice-pin",
                session="",
                window="",
            )
        return None

    async def resolve(self, text: str = "") -> ThreadBinding:
        if self.workspace is None:
            return ThreadBinding("default", self.fallback_thread, None, "fixed")
        if route := self._voice_route(text):
            if route.project in self.bindings:
                binding = self.bindings[route.project]
                binding = ThreadBinding(
                    binding.key,
                    binding.thread,
                    route,
                    "spoken_pin",
                )
                return binding
            binding = await self._attach(route, "spoken_pin")
            self.bindings[route.project] = binding
            return binding
        try:
            resolution = await asyncio.to_thread(self.workspace.resolve)
        except Exception as exc:
            print(f"Workspace route unavailable: {exc}", flush=True)
            return ThreadBinding(
                "default", self.fallback_thread, None, "workspace_error"
            )
        if resolution.route is None:
            projects = ",".join(route.project for route in resolution.candidates)
            print(
                f"Workspace route {resolution.reason}; candidates={projects or 'none'}. "
                "Keeping the launch-context thread.",
                flush=True,
            )
            return ThreadBinding(
                "default", self.fallback_thread, None, resolution.reason
            )
        route = resolution.route
        if route.project in self.bindings:
            return self.bindings[route.project]
        binding = await self._attach(route, resolution.reason)
        self.bindings[route.project] = binding
        return binding

    async def _attach(self, route: Route, reason: str) -> ThreadBinding:
        try:
            threads = await asyncio.wait_for(
                self.server.list_threads(route.cwd),
                timeout=self.timeout,
            )
            if threads:
                thread = await asyncio.wait_for(
                    self.server.resume_thread(
                        threads[0]["id"],
                        cwd=route.cwd,
                    ),
                    timeout=self.timeout,
                )
            else:
                raise AppServerError(
                    f"no existing Codex harness thread for {route.project}"
                )
            print(
                f"Voice route {route.project} pane={route.pane_id} thread={thread}",
                flush=True,
            )
            return ThreadBinding(route.project, thread, route, reason)
        except (TimeoutError, AppServerError, KeyError) as exc:
            raise AppServerError(
                f"could not attach the existing {route.project} harness: "
                f"{exc or 'timed out'}"
            )


async def existing_harness_thread(
    server: CodexAppServer,
    cwd: Path,
    requested: str | None,
    *,
    timeout: float,
) -> str:
    """Resolve a real harness thread and never manufacture a detached chat."""

    thread_id = requested or os.environ.get("CODEX_THREAD_ID")
    if thread_id:
        return await asyncio.wait_for(
            server.resume_thread(thread_id, cwd=cwd),
            timeout=timeout,
        )
    threads = await asyncio.wait_for(server.list_threads(cwd), timeout=timeout)
    if not threads:
        raise AppServerError(
            "no existing Codex harness thread; pass --session or launch from Codex"
        )
    return await asyncio.wait_for(
        server.resume_thread(threads[0]["id"], cwd=cwd),
        timeout=timeout,
    )


async def next_final(
    mic: Microphone,
    turn: DuplexTurn | None = None,
    *,
    interrupt_mode: str = "final",
) -> np.ndarray:
    while True:
        event = await mic.next()
        should_interrupt = (
            interrupt_mode == "immediate" and event.kind == "speech.started"
        ) or (
            interrupt_mode == "sustained" and event.kind == "speech.confirmed"
        )
        if turn and should_interrupt:
            await turn.interrupt()
        if event.kind == "speech.final" and event.audio is not None:
            return event.audio


async def run(args) -> None:
    from faster_whisper import WhisperModel

    if not args.skip_preflight:
        report = await asyncio.to_thread(
            preflight,
            whisper_python=sys.executable,
            kokoro_url=args.kokoro_url,
            ollama_url=None,
            live_model=None,
            input_device=args.input,
            output_device=args.output,
            workspace_routing=args.workspace_routing,
            routes=args.routes,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if not report["ok"]:
            raise RuntimeError(
                "voice preflight failed: " + "; ".join(report["failures"])
            )

    model = WhisperModel(
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=True,
    )
    base_config = ListenConfig(silence_ms=args.silence_ms)
    threshold = args.threshold
    input_name = "system default"
    if threshold is None:
        threshold, input_name = await asyncio.to_thread(
            calibrate_microphone,
            device=args.input,
            sample_rate=base_config.sample_rate,
            block_size=base_config.block_size,
            seconds=args.calibration_seconds,
        )
    config = ListenConfig(threshold=threshold, silence_ms=args.silence_ms)
    mic = Microphone(config, device=args.input)
    async with CodexAppServer(
        cwd=args.cwd,
        shared=args.app_server == "shared",
    ) as server:
        harness_thread = await existing_harness_thread(
            server,
            args.cwd,
            args.session,
            timeout=args.session_timeout,
        )
        print(f"Voice attached to existing Codex thread {harness_thread}.", flush=True)
        workspace = (
            WorkspaceRouter(load_routes(args.routes))
            if args.workspace_routing
            else None
        )
        harness_router = HarnessRouter(
            server,
            workspace,
            harness_thread,
            timeout=args.session_timeout,
        )
        if args.startup_phrase:
            startup_speaker = Speaker(
                args.kokoro_url,
                args.voice,
                backend=args.playback,
                output=args.output,
                latency=args.playback_latency,
            )
            await startup_speaker.say(args.startup_phrase)
        mic.start()
        print(
            f"Ready—just speak. input={input_name!r} "
            f"threshold={config.threshold:.4f} Ctrl-C exits.",
            flush=True,
        )
        pending_audio = None
        try:
            while True:
                audio = pending_audio if pending_audio is not None else await next_final(mic)
                pending_audio = None
                started = time.monotonic()
                asr_started = time.monotonic()
                text = await asyncio.to_thread(transcribe, model, audio, args.language)
                asr_seconds = time.monotonic() - asr_started
                if not text:
                    continue
                if voice_control(text) == "stop":
                    print(f"You[control]: {text}\nStopping voice mode.", flush=True)
                    break
                binding = await harness_router.resolve(text)
                print(
                    f"You[{binding.key} thread={binding.thread}]: {text}",
                    flush=True,
                )
                if args.event_url:
                    asyncio.create_task(
                        publish_transcript_best_effort(
                            args.event_url,
                            binding.thread,
                            text,
                            time.time_ns(),
                        )
                    )
                speaker = Speaker(
                    args.kokoro_url,
                    args.voice,
                    backend=args.playback,
                    output=args.output,
                    latency=args.playback_latency,
                )
                turn = DuplexTurn(server, binding.thread, speaker)
                task = asyncio.create_task(turn.run(text, args.effort))
                listen_task = asyncio.create_task(
                    next_final(
                        mic,
                        turn,
                        interrupt_mode=args.barge_in,
                    )
                )
                done, _ = await asyncio.wait(
                    (task, listen_task), return_when=asyncio.FIRST_COMPLETED
                )
                if listen_task in done:
                    pending_audio = listen_task.result()
                    await turn.interrupt()
                response = await task
                first_audio = (
                    speaker.first_play_at - started
                    if speaker.first_play_at is not None
                    else None
                )
                onset = f"{first_audio:.2f}s" if first_audio is not None else "none"
                synthesis = (
                    f"{speaker.first_synthesis_seconds:.2f}s"
                    if speaker.first_synthesis_seconds is not None
                    else "none"
                )
                print(
                    f"Codex: {response}\n"
                    f"latency asr={asr_seconds:.2f}s "
                    f"tts_first={synthesis} audio_onset={onset} "
                    f"total={time.monotonic() - started:.2f}s",
                    flush=True,
                )
                if args.metrics:
                    await asyncio.to_thread(
                        append_metric,
                        args.metrics,
                        {
                            "schema": 1,
                            "pipeline": "codex-harness-pcm-v1",
                            "recorded_at_ns": time.time_ns(),
                            "route": binding.key,
                            "routing_reason": binding.reason,
                            "asr_seconds": round(asr_seconds, 6),
                            "tts_first_seconds": (
                                round(speaker.first_synthesis_seconds, 6)
                                if speaker.first_synthesis_seconds is not None
                                else None
                            ),
                            "audio_onset_after_endpoint_seconds": (
                                round(first_audio, 6)
                                if first_audio is not None
                                else None
                            ),
                            "estimated_audio_onset_after_user_stop_seconds": (
                                round(first_audio + config.silence_ms / 1000, 6)
                                if first_audio is not None
                                else None
                            ),
                            "total_seconds": round(
                                time.monotonic() - started,
                                6,
                            ),
                            "interrupted": pending_audio is not None,
                        },
                    )
                if not listen_task.done():
                    listen_task.cancel()
        finally:
            mic.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--whisper-model", default="small.en")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="en")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--kokoro-url", default="http://127.0.0.1:8880")
    parser.add_argument(
        "--session",
        help="existing Codex harness thread; defaults to CODEX_THREAD_ID",
    )
    parser.add_argument(
        "--app-server",
        choices=("shared", "isolated"),
        default="shared",
        help="shared uses the managed Codex daemon; isolated is diagnostic only",
    )
    parser.add_argument("--session-timeout", type=float, default=8.0)
    parser.add_argument(
        "--workspace-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="route each utterance from privacy-safe workspace/tmux focus",
    )
    parser.add_argument(
        "--routes",
        type=Path,
        default=HERE.parent / "routes.json",
        help="project-to-cwd routing map",
    )
    parser.add_argument("--effort", default="medium")
    parser.add_argument(
        "--event-url",
        default="http://127.0.0.1:8787/v1/events",
        help="best-effort transcript event relay; pass an empty string to disable",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip dependency checks when a supervisor has already verified them",
    )
    parser.add_argument(
        "--playback",
        default="pipewire",
        choices=("pipewire", "sounddevice"),
    )
    parser.add_argument("--output", help="PipeWire sink node name or serial")
    parser.add_argument("--playback-latency", default="40ms")
    parser.add_argument(
        "--startup-phrase",
        default="Voice mode ready.",
        help="audible readiness check; pass an empty string to disable",
    )
    parser.add_argument(
        "--barge-in",
        default="sustained",
        choices=("final", "sustained", "immediate"),
        help="sustained yields after confirmed speech; final/immediate are explicit alternatives",
    )
    parser.add_argument("--input", help="sounddevice input index or device-name match")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="fixed VAD RMS threshold; default calibrates from ambient sound",
    )
    parser.add_argument("--calibration-seconds", type=float, default=1.0)
    parser.add_argument(
        "--metrics",
        type=Path,
        help="append one privacy-safe JSON latency record per completed turn",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=850,
        help="endpoint silence; 850 ms preserves natural mid-sentence pauses",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
