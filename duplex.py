#!/usr/bin/env python3
"""Full-duplex local voice runtime: mic -> Whisper -> Codex -> Kokoro."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
    str(HERE.parents[1] / "adapters/llm"),
    str(HERE.parents[1] / "adapters/voice_pm"),
]

from app_server import AppServerError, CodexAppServer
from conversation import ListenConfig, kokoro_wav, rms, transcribe
from control_plane import ControlServer, VoiceControl, default_control_socket
from events import Event
from floor import (
    AdaptiveEndpoint,
    EndpointHint,
    SubmissionDecision,
    TurnOwner,
    endpoint_hint,
)
from health import RuntimeHealth
from indicator import VoiceState, make_indicator
from lexicon import load_lexicon
from modes import MicMode, NotificationMode, VoiceModes
from turn_contract import require_audible_reply
from preflight import preflight
from providers import CodexSubscription, Provider
from workspace_router import Route, WorkspaceRouter, load_routes
from wiring import VoicePMWiring

LIVE_LANE_INSTRUCTIONS = """\
You are Zer0's realtime conversational voice lane. Respond immediately and
substantively to the user's latest spoken thought in natural conversational
English. Use one or two short sentences, usually under 35 words. Never emit
markdown, status boilerplate, canned acknowledgments, or phrases such as
"got it" and "I'm with you". Do not call tools. A separate authoritative
reasoning agent is doing the deep work, so answer what you can now and clearly
name only genuinely necessary uncertainty.
"""


@dataclass(frozen=True)
class AudioEvent:
    kind: str
    audio: np.ndarray | None = None
    generation: int | None = None
    detail: str | None = None


def append_debug(path: Path, kind: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"schema": 1, "ts_ns": time.time_ns(), "kind": kind, **fields}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def append_metric(path: Path, metric: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, separators=(",", ":"), sort_keys=True) + "\n")


def mark_capture_active(health: RuntimeHealth | None) -> None:
    """Keep long or continued speech out of the bounded ASR phase."""

    if health:
        health.transition("listening", lane="mic")


def pm_lifecycle_sink(path: Path | None):
    """Write publisher state only; the adapter never supplies turn content."""

    if path is None:
        return None

    def emit(row: dict[str, object]) -> None:
        values = dict(row)
        kind = str(values.pop("kind"))
        values.pop("schema", None)
        append_debug(path, kind, **values)

    return emit


def voice_pm_wiring(
    cache: dict[str, VoicePMWiring],
    *,
    endpoint: str,
    lane: str,
    thread: str,
    run_id: str,
    lifecycle,
) -> VoicePMWiring:
    """Return one stable publisher sequence per routed harness thread."""

    wiring = cache.get(thread)
    if wiring is None:
        route = hashlib.blake2s(
            thread.encode(),
            digest_size=6,
        ).hexdigest()
        wiring = VoicePMWiring.for_relay(
            endpoint,
            thread=thread,
            conversation=thread,
            run_id=f"{run_id}-{route}",
            lifecycle=lifecycle,
            active=lane,
        )
        cache[thread] = wiring
    elif wiring.active != lane:
        wiring.select(lane)
    return wiring


def schedule_pm_decision(
    cache: dict[str, VoicePMWiring],
    *,
    endpoint: str,
    lane: str,
    thread: str,
    run_id: str,
    lifecycle,
    decision: SubmissionDecision,
) -> tuple[VoicePMWiring, str]:
    """Schedule the live loop's committed owner decision without relay I/O."""

    wiring = voice_pm_wiring(
        cache,
        endpoint=endpoint,
        lane=lane,
        thread=thread,
        run_id=run_id,
        lifecycle=lifecycle,
    )
    return wiring, wiring.schedule(decision)


async def monitor_control(
    control: VoiceControl,
    debug_path: Path | None,
    health: RuntimeHealth | None = None,
) -> None:
    revision = -1
    while True:
        state = control.state if revision < 0 else await control.wait_after(revision)
        revision = state.revision
        if health:
            await asyncio.to_thread(
                health.control,
                mic_mode=state.modes.mic.value,
                notification_mode=state.modes.notifications.value,
                capture_active=state.capture_active,
            )
        if debug_path:
            await asyncio.to_thread(
                append_debug,
                debug_path,
                "voice.mode",
                mic=state.modes.mic.value,
                notifications=state.modes.notifications.value,
                push_held=state.push_held,
                live_model=state.live_model,
                live_effort=state.live_effort,
                capture_active=state.capture_active,
                revision=state.revision,
            )


async def monitor_health(
    record: RuntimeHealth,
    server: CodexAppServer | None = None,
) -> None:
    while True:
        # A closed shared app-server transport can leave the worker process
        # alive but unable to complete any future turn. Stop heartbeating so
        # the external watchdog rebuilds this disposable client.
        if server and server.reader and server.reader.done():
            await asyncio.to_thread(
                record.transition,
                "recovering",
                lane="app-server",
                reason="transport-closed",
            )
            return
        await asyncio.to_thread(record.touch)
        await asyncio.sleep(1)


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

    def __init__(
        self,
        config: ListenConfig,
        *,
        min_speech_ms: int = 180,
        endpoint: AdaptiveEndpoint | None = None,
    ):
        self.config = config
        self.endpoint = endpoint or AdaptiveEndpoint()
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
        self.preview_blocks = max(1, 240 // config.block_ms)
        self.last_preview_at = 0
        self._hint: EndpointHint | None = None
        self._hint_lock = threading.Lock()
        self.generation = 0
        self.last_endpoint_silence_ms = config.silence_ms

    def update_transcript(self, text: str, generation: int | None = None) -> bool:
        """Update semantic pacing from an unstable visual-only partial."""
        with self._hint_lock:
            if generation is not None and generation != self.generation:
                return False
            self._hint = endpoint_hint(text)
        return True

    def silence_needed_blocks(self) -> int:
        with self._hint_lock:
            hint = self._hint
        if hint is None:
            milliseconds = self.config.silence_ms
        elif hint.force:
            milliseconds = 180
        else:
            milliseconds = self.endpoint.silence_needed_ms(
                syntactically_complete=hint.complete,
                thinking_words=hint.thinking,
            )
        return max(1, milliseconds // self.config.block_ms)

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
                events.append(AudioEvent("speech.started", generation=self.generation))
                if self.active_voiced >= self.min_voiced_blocks:
                    self.confirmed = True
                    events.append(
                        AudioEvent("speech.confirmed", generation=self.generation)
                    )
            return events

        self.utterance.append(block)
        if voice:
            if self.silent:
                self.endpoint.observe_within_turn_pause(
                    self.silent * self.config.block_ms
                )
            self.active_voiced += 1
            if (
                not self.confirmed
                and self.active_voiced >= self.min_voiced_blocks
            ):
                self.confirmed = True
                events.append(AudioEvent("speech.confirmed", generation=self.generation))
        if (
            self.confirmed
            and len(self.utterance) - self.last_preview_at >= self.preview_blocks
        ):
            self.last_preview_at = len(self.utterance)
            events.append(
                AudioEvent(
                    "speech.preview",
                    np.concatenate(self.utterance),
                    self.generation,
                )
            )
        self.silent = 0 if voice else self.silent + 1
        max_blocks = int(self.config.max_seconds * 1000 / self.config.block_ms)
        if self.silent >= self.silence_needed_blocks() or len(self.utterance) >= max_blocks:
            keep = max(0, len(self.utterance) - self.silent)
            audio = np.concatenate(self.utterance[:keep]) if keep else None
            voiced_blocks = self.active_voiced
            generation = self.generation
            self.last_endpoint_silence_ms = self.silent * self.config.block_ms
            self.reset()
            if audio is not None and voiced_blocks >= self.min_voiced_blocks:
                events.append(AudioEvent("speech.final", audio, generation))
        return events

    def reset(self) -> None:
        self.before.clear()
        self.utterance = []
        self.voiced = self.active_voiced = self.silent = 0
        self.active = False
        self.confirmed = False
        self.last_preview_at = 0
        with self._hint_lock:
            self._hint = None
            self.generation += 1

    def finish(self) -> AudioEvent | None:
        """Finalize held push-to-talk audio without waiting for endpoint silence."""
        keep = max(0, len(self.utterance) - self.silent)
        audio = np.concatenate(self.utterance[:keep]) if keep else None
        voiced_blocks = self.active_voiced
        generation = self.generation
        self.reset()
        if audio is not None and voiced_blocks >= self.min_voiced_blocks:
            return AudioEvent("speech.final", audio, generation)
        return None


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
    def __init__(
        self,
        config: ListenConfig,
        *,
        device: str | int | None = None,
        health: RuntimeHealth | None = None,
    ):
        self.config = config
        self.device = device
        self.detector = UtteranceDetector(config)
        self.queue: asyncio.Queue[AudioEvent] = asyncio.Queue()
        self.stream = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stream_generation = 0
        self.health = health

    def _process_block(
        self,
        block: np.ndarray,
        warning: str | None,
        generation: int,
    ) -> None:
        if generation != self.stream_generation or self.stream is None:
            return
        if self.health:
            self.health.captured(len(block))
        if warning:
            self.queue.put_nowait(AudioEvent("audio.warning", detail=warning))
        for event in self.detector.push(block):
            self.queue.put_nowait(event)

    def start(self) -> None:
        if self.stream:
            return
        import sounddevice as sd

        self.loop = asyncio.get_running_loop()
        self.stream_generation += 1
        generation = self.stream_generation

        def callback(indata, _frames, _time, status):
            warning = str(status).strip().lower().replace(" ", "_") if status else None
            self.loop.call_soon_threadsafe(
                self._process_block,
                indata[:, 0].copy(),
                warning,
                generation,
            )

        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.config.block_size,
            device=self.device,
            callback=callback,
        )
        self.stream.start()
        if self.health:
            self.health.expect_capture(True)

    def close(self) -> None:
        if self.health:
            self.health.expect_capture(False)
        self.stream_generation += 1
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.detector.reset()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def pause(self, *, finalize: bool = False) -> AudioEvent | None:
        if self.health:
            self.health.expect_capture(False)
        self.stream_generation += 1
        event = self.detector.finish() if finalize else None
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if not finalize:
            self.detector.reset()
        return event

    async def next(self) -> AudioEvent:
        return await self.queue.get()

    def update_transcript(self, text: str, generation: int | None = None) -> bool:
        return self.detector.update_transcript(text, generation)

    @property
    def endpoint_silence_ms(self) -> int:
        return self.detector.last_endpoint_silence_ms


def apply_live_reload(
    *,
    live_context: object,
    live_model: str | None,
    live_effort: str,
    control_state,
) -> tuple[str | None, str]:
    if (
        control_state.live_model is not None
        and control_state.live_model != live_context.model
    ):
        live_context.model = control_state.live_model
        live_model = control_state.live_model
    if control_state.live_effort:
        live_effort = control_state.live_effort
    return live_model, live_effort


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

    def _fallback_audio(self, text: str, synthesis_started: float) -> None:
        """Best-effort audio fallback using in-memory WAV playback."""

        import sounddevice as sd

        from conversation import decode_wav

        if not text:
            return
        audio = kokoro_wav(text, base_url=self.base_url, voice=self.voice)
        if self.first_synthesis_seconds is None:
            self.first_synthesis_seconds = time.monotonic() - synthesis_started
        if self.cancel.is_set():
            return
        if self.first_play_at is None:
            self.first_play_at = time.monotonic()
        samples, rate = decode_wav(audio)
        sd.play(samples, rate)
        sd.wait()

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
            try:
                await asyncio.to_thread(
                    self._pipewire_stream,
                    text,
                    synthesis_started,
                )
            except RuntimeError:
                # Keep the live system responsive if PipeWire fails transiently.
                # This commonly happens with sink reconfiguration, racey sink
                # targets, or temporary sound server state.
                await asyncio.to_thread(
                    self._fallback_audio,
                    text,
                    synthesis_started,
                )
            return
        await asyncio.to_thread(self._fallback_audio, text, synthesis_started)


class DuplexTurn:
    _ABORT_TIMEOUT_SECONDS = 0.75

    def __init__(
        self,
        provider: Provider,
        speaker: Speaker,
        *,
        on_speaking=None,
        on_delta=None,
        prefix_speech: str = "",
    ):
        self.provider = provider
        self.speaker = speaker
        self.on_speaking = on_speaking
        self.on_delta = on_delta
        self.prefix_speech = prefix_speech
        self.turn_id: str | None = None

    async def run(self, text: str, effort: str) -> str:
        chunks: asyncio.Queue[str | None] = asyncio.Queue()
        response: list[str] = []

        async def generate() -> None:
            chunker = SentenceChunker()
            async for event in self.provider.stream(text, effort=effort):
                self.turn_id = event.turn_id
                if event.kind == "delta":
                    delta = event.text
                    response.append(delta)
                    if self.on_delta:
                        self.on_delta("".join(response))
                    for chunk in chunker.feed(delta):
                        await chunks.put(chunk)
            if tail := chunker.flush():
                await chunks.put(tail)
            await chunks.put(None)

        async def speak() -> None:
            announced = False
            if self.prefix_speech:
                if self.on_speaking:
                    self.on_speaking()
                    announced = True
                await self.speaker.say(self.prefix_speech)
            while (chunk := await chunks.get()) is not None:
                if not announced and self.on_speaking:
                    self.on_speaking()
                    announced = True
                await self.speaker.say(chunk)

        generate_task = asyncio.create_task(generate(), name="voice-turn-generate")
        speak_task = asyncio.create_task(speak(), name="voice-turn-speak")
        tasks = (generate_task, speak_task)
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # asyncio.gather does not cancel sibling tasks when one fails. Stop
            # playback first (including its worker thread), then close both
            # coroutines and best-effort interrupt any remote generation.
            try:
                self.speaker.interrupt()
            except Exception:
                pass
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.turn_id:
                try:
                    await asyncio.wait_for(
                        self.provider.interrupt(self.turn_id),
                        timeout=self._ABORT_TIMEOUT_SECONDS,
                    )
                except (Exception, asyncio.CancelledError):
                    pass
            raise
        return "".join(response).strip()

    async def interrupt(self) -> None:
        self.speaker.interrupt()
        if self.turn_id:
            await self.provider.interrupt(self.turn_id)


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


class LiveContextMirror:
    """Fork the authoritative thread so live speech sees keyboard history."""

    def __init__(
        self,
        server: CodexAppServer,
        *,
        cwd: Path,
        model: str | None,
        developer_instructions: str,
        fixed_thread: str | None = None,
    ) -> None:
        self.server = server
        self.cwd = cwd
        self.model = model
        self.developer_instructions = developer_instructions
        self.fixed_thread = fixed_thread

    async def refresh(self, authoritative_thread: str) -> str:
        if self.fixed_thread:
            return self.fixed_thread
        # The visible harness is usually in-progress while voice is speaking.
        # Forking its moving head copies that active turn; stream_turn then
        # steers a half-finished response and may never receive a clean
        # completion. Branch from the newest completed turn instead. This keeps
        # all stable keyboard/voice history while making the realtime fork idle.
        last_completed: str | None = None
        thread = await self.server.read_thread(authoritative_thread)
        for turn in reversed(thread.get("turns", [])):
            if turn.get("status") == "completed" and turn.get("id"):
                last_completed = str(turn["id"])
                break
        return await self.server.fork_thread(
            authoritative_thread,
            cwd=self.cwd,
            model=self.model,
            developer_instructions=self.developer_instructions,
            ephemeral=True,
            last_turn_id=last_completed,
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
    on_event=None,
) -> np.ndarray:
    while True:
        event = await mic.next()
        if on_event:
            on_event(event)
        should_interrupt = (
            interrupt_mode == "immediate" and event.kind == "speech.started"
        ) or (
            interrupt_mode == "sustained" and event.kind == "speech.confirmed"
        )
        # A continuation fragment captured before any reply audio exists must
        # not repeatedly kill the live lane. Preserve that audio for the next
        # turn; sustained barge-in becomes active once playback has actually
        # begun. Explicit `immediate` remains an opt-in hard preemption mode.
        if (
            should_interrupt
            and interrupt_mode == "sustained"
            and getattr(getattr(turn, "speaker", None), "first_play_at", None)
            is None
        ):
            should_interrupt = False
        if turn and should_interrupt:
            await turn.interrupt()
        if event.kind == "speech.final" and event.audio is not None:
            return event.audio


async def next_controlled_final(
    mic: Microphone,
    control: VoiceControl,
    indicator,
    turn: DuplexTurn | None = None,
    *,
    interrupt_mode: str = "final",
    on_event=None,
) -> np.ndarray:
    """Capture one utterance while enforcing live mute/PTT transitions."""
    while True:
        state = control.state
        if not state.capture_active:
            mic.close()
            indicator.clear()
            state = await control.wait_after(state.revision)
            continue
        mic.start()
        indicator.set(VoiceState.LISTENING)
        audio_task = asyncio.create_task(
            next_final(
                mic,
                turn,
                interrupt_mode=interrupt_mode,
                on_event=on_event,
            )
        )
        change_task = asyncio.create_task(control.wait_after(state.revision))
        try:
            done, _ = await asyncio.wait(
                (audio_task, change_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if audio_task in done:
                return audio_task.result()
            changed = change_task.result()
            if changed.capture_active:
                continue
            released_ptt = (
                state.modes.mic is MicMode.PUSH_TO_TALK
                and state.push_held
                and not changed.push_held
            )
            final = mic.pause(finalize=released_ptt)
            indicator.clear()
            if final and final.audio is not None:
                if on_event:
                    on_event(final)
                return final.audio
        finally:
            for task in (audio_task, change_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                audio_task,
                change_task,
                return_exceptions=True,
            )


@dataclass(frozen=True)
class OwnedTranscript:
    """One committed thought plus capture already running for the next thought."""

    decision: SubmissionDecision
    next_audio: asyncio.Task
    asr_seconds: float


async def collect_owned_transcript(
    owner: TurnOwner,
    capture,
    transcribe_audio,
    *,
    initial_audio: np.ndarray | None = None,
    on_event=None,
    on_segment=None,
    clock=time.monotonic,
) -> OwnedTranscript:
    """Pipeline capture and ASR while one owner decides the harness boundary.

    Capture for fragment N+1 starts before ASR for fragment N. A speech-start
    event wins over the commit timer, so a person resuming before the deadline
    cannot be split merely because their next VAD endpoint arrives later.
    """

    activity = asyncio.Event()
    speech_active = False
    next_audio: asyncio.Task | None = None
    handed_off = False
    asr_seconds = 0.0

    def observe(event: AudioEvent) -> None:
        nonlocal speech_active
        if event.kind == "speech.started":
            speech_active = True
            activity.set()
        elif event.kind == "speech.final":
            speech_active = False
        if on_event:
            on_event(event)

    def begin_capture() -> asyncio.Task:
        nonlocal speech_active
        speech_active = False
        activity.clear()
        return asyncio.create_task(capture(observe), name="voice-owned-capture")

    try:
        if initial_audio is None:
            next_audio = begin_capture()
            audio = await next_audio
        else:
            audio = initial_audio
        next_audio = begin_capture()

        while True:
            asr_started = time.monotonic()
            text = await transcribe_audio(audio)
            asr_seconds += time.monotonic() - asr_started
            decision = (
                owner.observe(text, now=clock())
                if text
                else SubmissionDecision("reject", owner.pending, "empty")
            )
            if on_segment:
                await on_segment(text, decision)
            if decision.action == "submit":
                handed_off = True
                return OwnedTranscript(decision, next_audio, asr_seconds)

            # Audio that endpointed or speech that began during ASR belongs to
            # this owner regardless of whether the prior deadline has elapsed.
            if next_audio.done() or activity.is_set() or speech_active:
                audio = await next_audio
                next_audio = begin_capture()
                continue

            remaining = owner.remaining(now=clock())
            if remaining is None:
                audio = await next_audio
                next_audio = begin_capture()
                continue

            activity_wait = asyncio.create_task(
                activity.wait(),
                name="voice-owned-activity",
            )
            try:
                done, _ = await asyncio.wait(
                    (next_audio, activity_wait),
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not activity_wait.done():
                    activity_wait.cancel()
                await asyncio.gather(activity_wait, return_exceptions=True)

            if next_audio in done or activity.is_set() or speech_active:
                audio = await next_audio
                next_audio = begin_capture()
                continue

            decision = owner.due(now=clock())
            if decision.action == "submit":
                handed_off = True
                return OwnedTranscript(decision, next_audio, asr_seconds)
    except BaseException:
        owner.cancel()
        raise
    finally:
        if not handed_off and next_audio is not None and not next_audio.done():
            next_audio.cancel()
            await asyncio.gather(next_audio, return_exceptions=True)


async def race_response_and_capture(
    response_task: asyncio.Task,
    listen_task: asyncio.Task,
    turn: DuplexTurn,
    *,
    interrupt_timeout: float = 0.35,
    unspoken_reply_grace: float = 5.0,
    preempt_on_completed_capture: bool = False,
) -> tuple[str | None, np.ndarray | None]:
    """Buffer captured speech unless the selected mode explicitly preempts.

    Capture and playback intentionally overlap. Once the next utterance has
    reached an endpoint, preserve its audio for the next ASR turn. The safe
    default lets the current audible reply finish; aggressive barge-in modes
    may explicitly stop the speaker/provider and hand the audio back
    immediately.
    """

    done, _ = await asyncio.wait(
        (response_task, listen_task),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if listen_task in done:
        audio = listen_task.result()
        if not preempt_on_completed_capture:
            response = await response_task
            return response, audio
        # Continuous ASR may endpoint a continuation while the model is still
        # producing its first audio. Give an unspoken reply a short protected
        # window and preserve the captured audio for the following turn.
        # Once playback has begun, a completed utterance is a real barge-in.
        speaker = getattr(turn, "speaker", None)
        if speaker is not None and getattr(speaker, "first_play_at", None) is None:
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(response_task),
                    timeout=unspoken_reply_grace,
                )
            except TimeoutError:
                pass
            else:
                return response, audio
        try:
            await asyncio.wait_for(turn.interrupt(), timeout=interrupt_timeout)
        except (TimeoutError, OSError):
            pass
        if not response_task.done():
            response_task.cancel()
        await asyncio.gather(response_task, return_exceptions=True)
        return None, audio

    response = await response_task
    if not listen_task.done():
        listen_task.cancel()
        await asyncio.gather(listen_task, return_exceptions=True)
    return response, None


async def submit_authoritative(
    server: CodexAppServer,
    thread: str,
    text: str,
    effort: str,
    debug_path: Path | None,
) -> None:
    """Deliver a committed thought to the harness without awaiting its reply."""
    error: Exception | None = None
    for attempt in range(2):
        try:
            turn_id = await server.submit_turn(thread, text, effort=effort)
            if debug_path:
                await asyncio.to_thread(
                    append_debug,
                    debug_path,
                    "codex.authoritative.submitted",
                    turn=turn_id,
                    lane="medium",
                    effort=effort,
                )
            return
        except (AppServerError, OSError, TimeoutError) as caught:
            error = caught
            # A keyboard or voice turn can replace the active turn between
            # thread/read and turn/steer. Re-read once instead of treating that
            # normal shared-harness race as a lane outage.
            if attempt == 0 and "expected active turn id" in str(caught):
                await asyncio.sleep(0)
                continue
            break
    if debug_path:
        await asyncio.to_thread(
            append_debug,
            debug_path,
            "codex.authoritative.error",
            message=str(error) or type(error).__name__,
            lane="medium",
            effort=effort,
        )
    print(f"Authoritative lane unavailable: {error}", flush=True)


class ConsecutiveFailureBudget:
    """Restart a lane only after repeated failures, never after one bad turn."""

    def __init__(self, limit: int = 3) -> None:
        if limit < 1:
            raise ValueError("failure limit must be positive")
        self.limit = limit
        self.count = 0

    def failed(self) -> bool:
        self.count += 1
        return self.count >= self.limit

    def recovered(self) -> None:
        self.count = 0


async def transcribe_only(
    model,
    mic: Microphone,
    args,
    *,
    config: ListenConfig,
    input_name: str,
    control: VoiceControl,
    indicator,
    lexicon,
) -> None:
    debug_path = args.debug_events
    asr_lock = asyncio.Lock()
    preview_task: asyncio.Task | None = None
    last_preview = ""

    async def preview(audio: np.ndarray, generation: int | None) -> None:
        nonlocal last_preview
        async with asr_lock:
            text = await asyncio.to_thread(
                transcribe,
                model,
                audio,
                args.language,
                lexicon=lexicon,
            )
        if text and text != last_preview:
            if not mic.update_transcript(text, generation):
                return
            last_preview = text
            print(f"You[partial]: {text}", flush=True)
            if debug_path:
                await asyncio.to_thread(
                    append_debug, debug_path, "asr.partial", text=text
                )

    def observe(event: AudioEvent) -> None:
        nonlocal preview_task
        if debug_path and event.kind in (
            "speech.started",
            "speech.confirmed",
            "audio.warning",
        ):
            asyncio.create_task(
                asyncio.to_thread(
                    append_debug,
                    debug_path,
                    event.kind,
                    **({"detail": event.detail} if event.detail else {}),
                )
            )
        if event.kind == "speech.preview" and event.audio is not None:
            if preview_task is None or preview_task.done():
                preview_task = asyncio.create_task(
                    preview(event.audio.copy(), event.generation)
                )

    if debug_path:
        await asyncio.to_thread(
            append_debug,
            debug_path,
            "voice.listening" if control.state.capture_active else "voice.muted",
            mode="transcribe-only",
            mic_mode=control.state.modes.mic.value,
            input=input_name,
            threshold=round(config.threshold, 6),
        )
    print(
        f"Live transcription ready. input={input_name!r} "
        f"threshold={config.threshold:.4f} "
        f"mic={control.state.modes.mic.value} Ctrl-C exits.",
        flush=True,
    )
    try:
        while True:
            audio = await next_controlled_final(
                mic,
                control,
                indicator,
                on_event=observe,
            )
            async with asr_lock:
                text = await asyncio.to_thread(
                    transcribe,
                    model,
                    audio,
                    args.language,
                    lexicon=lexicon,
                    accurate=True,
                )
            if text:
                last_preview = text
                print(f"You[final]: {text}", flush=True)
                if debug_path:
                    await asyncio.to_thread(
                        append_debug, debug_path, "asr.final", text=text
                    )
    finally:
        mic.close()
        if preview_task and not preview_task.done():
            preview_task.cancel()


async def run(args) -> None:
    from faster_whisper import WhisperModel

    run_id = f"{time.time_ns():x}-{os.getpid():x}"
    health = RuntimeHealth(args.health, run_id) if args.health else None
    health_monitor = (
        asyncio.create_task(monitor_health(health), name="voice-health")
        if health
        else None
    )
    if health:
        health.transition("starting", reason="preflight")
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

    if health:
        health.transition("starting", reason="model-loading")
    model = WhisperModel(
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=True,
    )
    lexicon = load_lexicon(
        HERE.parent / "lexicon.json",
        Path.home() / ".config/zer0/voice-lexicon.json",
        *args.lexicon,
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
    mic = Microphone(config, device=args.input, health=health)
    control = VoiceControl(
        VoiceModes(
            mic=MicMode(args.mic_mode),
            notifications=NotificationMode(args.notification_mode),
        )
    )
    await control.apply(
        {
            "live_model": args.live_model,
            "live_effort": args.live_effort,
        }
    )
    indicator = make_indicator(
        args.keyboard_indicator,
        library=args.wooting_rgb_lib,
    )
    control_server = ControlServer(args.control_socket, control)
    debug_path = args.debug_events
    control_monitor = asyncio.create_task(
        monitor_control(control, debug_path, health)
    )
    if args.transcribe_only:
        try:
            await control_server.start()
            await transcribe_only(
                model,
                mic,
                args,
                config=config,
                input_name=input_name,
                control=control,
                indicator=indicator,
                lexicon=lexicon,
            )
        finally:
            control_monitor.cancel()
            indicator.close()
            await control_server.close()
        return
    if debug_path:
        await asyncio.to_thread(
            append_debug,
            debug_path,
            "voice.starting",
            session=args.session or "",
            run_id=run_id,
            bundle_sha256=args.release_bundle,
        )
    try:
        if health:
            health.transition("attaching", lane="live")
        server_context = CodexAppServer(
            cwd=args.cwd,
            shared=args.app_server == "shared",
            startup_timeout=args.session_timeout,
        )
        server = await server_context.__aenter__()
        if health_monitor:
            health_monitor.cancel()
            await asyncio.gather(health_monitor, return_exceptions=True)
            health_monitor = asyncio.create_task(
                monitor_health(health, server),
                name="voice-health",
            )
    except Exception as error:
        if debug_path:
            await asyncio.to_thread(
                append_debug,
                debug_path,
                "voice.error",
                message=str(error) or type(error).__name__,
            )
        indicator.close()
        control_monitor.cancel()
        await control_server.close()
        raise
    try:
        harness_thread = await existing_harness_thread(
            server,
            args.cwd,
            args.session,
            timeout=args.session_timeout,
        )
        live_model = control.state.live_model or args.live_model
        live_effort = control.state.live_effort or args.live_effort
        live_context = LiveContextMirror(
            server,
            cwd=args.cwd,
            model=live_model,
            developer_instructions=LIVE_LANE_INSTRUCTIONS,
            fixed_thread=args.live_session,
        )
        live_thread = await asyncio.wait_for(
            live_context.refresh(harness_thread),
            timeout=args.session_timeout,
        )
        # The private control socket is also the launcher readiness handshake.
        # Publish it only after both the authoritative and live threads attach.
        await control_server.start()
        print(f"Voice attached to existing Codex thread {harness_thread}.", flush=True)
        if debug_path:
            await asyncio.to_thread(
                append_debug,
                debug_path,
                "voice.attached",
                session=harness_thread,
                live_session=live_thread,
            )
            await asyncio.to_thread(
                append_debug,
                debug_path,
                "voice.lanes.configured",
                lanes=[
                    {
                        "name": "live",
                        "horizon": "realtime",
                        "effort": live_effort,
                        "model": live_model or "codex-default",
                        "state": "ready",
                    },
                    {
                        "name": "instant",
                        "horizon": "instant",
                        "effort": "minimal",
                        "model": "not-configured",
                        "state": "planned",
                    },
                    {
                        "name": "medium",
                        "horizon": "short",
                        "effort": args.effort,
                        "model": "codex-harness",
                        "state": "ready",
                    },
                    {
                        "name": "high",
                        "horizon": "mid",
                        "effort": "high",
                        "model": "not-configured",
                        "state": "planned",
                    },
                    {
                        "name": "pro",
                        "horizon": "meta",
                        "effort": "max",
                        "model": "not-configured",
                        "state": "planned",
                    },
                ],
            )
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
        if debug_path:
            await asyncio.to_thread(
                append_debug,
                debug_path,
                "voice.listening"
                if control.state.capture_active
                else "voice.muted",
                mic_mode=control.state.modes.mic.value,
                notification_mode=control.state.modes.notifications.value,
                input=input_name,
                threshold=round(config.threshold, 6),
            )
        print(
            f"Ready—just speak. input={input_name!r} "
            f"threshold={config.threshold:.4f} "
            f"mic={control.state.modes.mic.value} "
            f"notify={control.state.modes.notifications.value} "
            f"control={args.control_socket} Ctrl-C exits.",
            flush=True,
        )
        if health:
            health.transition("listening", lane="mic")
        pending_audio = None
        owned_capture: asyncio.Task | None = None
        preview_task: asyncio.Task | None = None
        last_preview = ""
        asr_lock = asyncio.Lock()
        turn_owner = TurnOwner(lexicon.terms)
        background: set[asyncio.Task] = set()
        pm_wirings: dict[str, VoicePMWiring] = {}
        pm_lifecycle = pm_lifecycle_sink(debug_path)
        speech_started_at: float | None = None
        first_partial_seconds: float | None = None
        live_failures = ConsecutiveFailureBudget()

        async def preview(audio: np.ndarray, generation: int | None) -> None:
            nonlocal last_preview, first_partial_seconds
            async with asr_lock:
                text = await asyncio.to_thread(
                    transcribe,
                    model,
                    audio,
                    args.language,
                    lexicon=lexicon,
                )
            if text and text != last_preview:
                if not mic.update_transcript(text, generation):
                    return
                if first_partial_seconds is None and speech_started_at is not None:
                    first_partial_seconds = time.monotonic() - speech_started_at
                last_preview = text
                print(f"You[partial]: {text}", flush=True)
                if debug_path:
                    await asyncio.to_thread(
                        append_debug, debug_path, "asr.partial", text=text
                        , onset_seconds=(
                            round(first_partial_seconds, 6)
                            if first_partial_seconds is not None
                            else None
                        )
                    )

        def observe(event: AudioEvent) -> None:
            nonlocal preview_task, speech_started_at, first_partial_seconds
            if event.kind == "speech.started":
                speech_started_at = time.monotonic()
                first_partial_seconds = None
            if debug_path and event.kind in (
                "speech.started",
                "speech.confirmed",
                "audio.warning",
            ):
                asyncio.create_task(
                    asyncio.to_thread(
                        append_debug,
                        debug_path,
                        event.kind,
                        **({"detail": event.detail} if event.detail else {}),
                    )
                )
            if event.kind == "speech.preview" and event.audio is not None:
                if preview_task is None or preview_task.done():
                    preview_task = asyncio.create_task(
                        preview(event.audio.copy(), event.generation)
                    )

        async def capture_owned(on_capture_event) -> np.ndarray:
            # A semantic continuation begins a fresh capture after the previous
            # fragment's ASR. Without this transition, the watchdog mistakes
            # long, actively streaming speech for a stuck transcriber.
            mark_capture_active(health)
            return await next_controlled_final(
                mic,
                control,
                indicator,
                on_event=on_capture_event,
            )

        async def transcribe_owned(audio: np.ndarray) -> str:
            if health:
                health.transition("transcribing", lane="asr")
            async with asr_lock:
                return await asyncio.to_thread(
                    transcribe,
                    model,
                    audio,
                    args.language,
                    lexicon=lexicon,
                    accurate=True,
                )

        async def show_owned_segment(
            segment: str,
            decision: SubmissionDecision,
        ) -> None:
            nonlocal last_preview
            if segment:
                last_preview = segment
            if decision.action == "reject":
                print(
                    f"You[rejected:{decision.reason}]: {segment}",
                    flush=True,
                )
                if debug_path:
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "asr.rejected",
                        reason=decision.reason,
                    )
                return
            print(f"You[continued]: {segment}", flush=True)
            if debug_path:
                await asyncio.to_thread(
                    append_debug,
                    debug_path,
                    "asr.owned",
                    action=decision.action,
                    reason=decision.reason,
                )

        try:
            while True:
                control_state = control.state
                live_model, live_effort = apply_live_reload(
                    live_context=live_context,
                    live_model=live_model,
                    live_effort=live_effort,
                    control_state=control_state,
                )
                if debug_path:
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "voice.live.config",
                        live_model=live_context.model,
                        effort=live_effort,
                    )
                if health:
                    health.transition("listening", lane="mic")
                owned = await collect_owned_transcript(
                    turn_owner,
                    capture_owned,
                    transcribe_owned,
                    initial_audio=pending_audio,
                    on_event=observe,
                    on_segment=show_owned_segment,
                )
                pending_audio = None
                owned_capture = owned.next_audio
                text = owned.decision.text
                asr_seconds = owned.asr_seconds
                started = time.monotonic() - asr_seconds
                indicator.set(VoiceState.THINKING)
                last_preview = text
                if debug_path:
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "asr.final",
                        text=text,
                        reason=owned.decision.reason,
                    )
                if voice_control(text) == "stop":
                    owned_capture.cancel()
                    await asyncio.gather(owned_capture, return_exceptions=True)
                    owned_capture = None
                    print(f"You[control]: {text}\nStopping voice mode.", flush=True)
                    break
                if health:
                    health.transition("syncing", lane="live")
                binding = await harness_router.resolve(text)
                publisher, published_lane = schedule_pm_decision(
                    pm_wirings,
                    endpoint=args.committed_voice_url,
                    lane=args.pm_publisher,
                    thread=binding.thread,
                    run_id=run_id,
                    lifecycle=pm_lifecycle,
                    decision=owned.decision,
                )
                if debug_path:
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "pm.publisher.scheduled",
                        lane=published_lane,
                        source_id=(
                            f"{publisher.run_id}-"
                            f"{publisher.commit_sequence:x}"
                        ),
                    )
                live_thread = await asyncio.wait_for(
                    live_context.refresh(binding.thread),
                    timeout=args.session_timeout,
                )
                if debug_path:
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "voice.routed",
                        route=binding.key,
                        session=binding.thread,
                    )
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "voice.context.synced",
                        mode=(
                            "fixed-live-thread"
                            if args.live_session
                            else "authoritative-thread-fork"
                        ),
                        route=binding.key,
                        source_session=binding.thread,
                        live_session=live_thread,
                    )
                print(
                    f"You[{binding.key} thread={binding.thread}]: {text}",
                    flush=True,
                )
                authoritative = asyncio.create_task(
                    submit_authoritative(
                        server,
                        binding.thread,
                        text,
                        args.effort,
                        debug_path,
                    )
                )
                background.add(authoritative)
                authoritative.add_done_callback(background.discard)
                if debug_path:
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "voice.lane.state",
                        lane="medium",
                        state="accepted",
                        effort=args.effort,
                    )
                speaker = Speaker(
                    args.kokoro_url,
                    args.voice,
                    backend=args.playback,
                    output=args.output,
                    latency=args.playback_latency,
                )
                turn = DuplexTurn(
                    CodexSubscription(server, live_thread),
                    speaker,
                    on_speaking=lambda: (
                        indicator.set(VoiceState.SPEAKING),
                        health.transition("speaking", lane="tts") if health else None,
                    ),
                    on_delta=(
                        (
                            lambda text: append_debug(
                                debug_path,
                                "voice.response.partial",
                                text=text,
                                lane="live",
                            )
                        )
                        if debug_path
                        else None
                    ),
                )
                if health:
                    health.transition("generating", lane="live")
                task = asyncio.create_task(
                    asyncio.wait_for(
                        turn.run(text, live_effort),
                        timeout=args.live_timeout,
                    )
                )
                if debug_path:
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "voice.lane.state",
                        lane="live",
                        state="running",
                        effort=live_effort,
                    )
                listen_task = owned_capture if args.barge_in != "off" else None
                if listen_task is None:
                    owned_capture.cancel()
                    await asyncio.gather(owned_capture, return_exceptions=True)
                    owned_capture = None
                try:
                    if listen_task is None:
                        response, pending_audio = await task, None
                    else:
                        response, pending_audio = await race_response_and_capture(
                            task,
                            listen_task,
                            turn,
                            preempt_on_completed_capture=(
                                args.barge_in in {"sustained", "immediate"}
                            ),
                        )
                        owned_capture = None
                    if response is not None:
                        require_audible_reply(
                            response,
                            speaker.first_play_at is not None,
                        )
                except (
                    AppServerError,
                    OSError,
                    RuntimeError,
                    TimeoutError,
                ) as error:
                    speaker.interrupt()
                    if listen_task is not None and not listen_task.done():
                        listen_task.cancel()
                        await asyncio.gather(
                            listen_task,
                            return_exceptions=True,
                        )
                    owned_capture = None
                    if control.state.capture_active:
                        indicator.set(VoiceState.LISTENING)
                    else:
                        indicator.clear()
                    if debug_path:
                        await asyncio.to_thread(
                            append_debug,
                            debug_path,
                            "voice.live.error",
                            message=str(error) or type(error).__name__,
                        )
                        await asyncio.to_thread(
                            append_debug,
                            debug_path,
                            "voice.lane.state",
                            lane="live",
                            state="error",
                            effort=live_effort,
                        )
                    print(f"Live lane unavailable: {error}", flush=True)
                    # A single provider race or failed turn must not surrender
                    # the microphone to the reduced simple fallback. Keep
                    # capturing and rebuild the disposable live fork next turn.
                    # Repeated failures still exit so the supervisor can
                    # reconstruct the shared transport and audio dependencies.
                    if not live_failures.failed():
                        if health:
                            health.transition(
                                "listening",
                                lane="mic",
                                reason="live-turn-retry",
                            )
                        continue
                    raise
                if response is None:
                    live_failures.recovered()
                    if debug_path:
                        await asyncio.to_thread(
                            append_debug,
                            debug_path,
                            "voice.interrupted",
                            handoff="next-utterance",
                        )
                        await asyncio.to_thread(
                            append_debug,
                            debug_path,
                            "voice.lane.state",
                            lane="live",
                            state="interrupted",
                            effort=live_effort,
                        )
                        await asyncio.to_thread(
                            append_debug,
                            debug_path,
                            "voice.lane.state",
                            lane="live",
                            state="ready",
                            effort=live_effort,
                            recovered_from="interrupted",
                        )
                    if health:
                        health.transition("listening", lane="mic")
                    continue
                live_failures.recovered()
                if control.state.capture_active:
                    indicator.set(VoiceState.LISTENING)
                else:
                    indicator.clear()
                if health:
                    health.transition("listening", lane="mic")
                if debug_path:
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "voice.response",
                        text=response,
                        lane="live",
                    )
                    await asyncio.to_thread(
                        append_debug,
                        debug_path,
                        "voice.lane.state",
                        lane="live",
                        state="ready",
                        effort=live_effort,
                    )
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
                    f"Live: {response}\n"
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
                            "pipeline": "codex-continuous-pcm-v5",
                            "run_id": run_id,
                            "ts_ns": time.time_ns(),
                            "recorded_at_ns": time.time_ns(),
                            "route": binding.key,
                            "routing_reason": binding.reason,
                            "asr_seconds": round(asr_seconds, 6),
                            "speech_to_partial_seconds": (
                                round(first_partial_seconds, 6)
                                if first_partial_seconds is not None
                                else None
                            ),
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
                                round(first_audio + mic.endpoint_silence_ms / 1000, 6)
                                if first_audio is not None
                                else None
                            ),
                            "endpoint_silence_seconds": round(
                                mic.endpoint_silence_ms / 1000,
                                6,
                            ),
                            "total_seconds": round(
                                time.monotonic() - started,
                                6,
                            ),
                            "interrupted": pending_audio is not None,
                        },
                    )
        finally:
            mic.close()
            if owned_capture and not owned_capture.done():
                owned_capture.cancel()
                await asyncio.gather(owned_capture, return_exceptions=True)
            if preview_task and not preview_task.done():
                preview_task.cancel()
            for task in background:
                task.cancel()
            if pm_wirings:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(wiring.drain() for wiring in pm_wirings.values())
                    ),
                    timeout=2.0,
                )
    finally:
        await server_context.__aexit__(None, None, None)
        control_monitor.cancel()
        if health_monitor:
            health_monitor.cancel()
        indicator.close()
        await control_server.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--whisper-model", default="small.en")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--lexicon",
        type=Path,
        action="append",
        default=[],
        help="additional JSON vocabulary/correction file; may be repeated",
    )
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--kokoro-url", default="http://127.0.0.1:8880")
    parser.add_argument(
        "--session",
        help="existing Codex harness thread; defaults to CODEX_THREAD_ID",
    )
    parser.add_argument(
        "--release-bundle",
        default=os.environ.get("ZERO_VOICE_BUNDLE_SHA256"),
        help="immutable release digest recorded in privacy-safe canary events",
    )
    parser.add_argument(
        "--app-server",
        choices=("shared", "isolated"),
        default="shared",
        help="shared uses the managed Codex daemon; isolated is diagnostic only",
    )
    parser.add_argument("--session-timeout", type=float, default=8.0)
    parser.add_argument(
        "--live-timeout",
        type=float,
        default=35.0,
        help="restart the voice worker if one realtime response exceeds this bound",
    )
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
        "--live-effort",
        default="low",
        help="reasoning effort for the independent realtime voice lane",
    )
    parser.add_argument(
        "--live-model",
        help="optional model override for the realtime voice lane",
    )
    parser.add_argument(
        "--live-session",
        help="reuse an existing realtime voice thread instead of an ephemeral one",
    )
    parser.add_argument(
        "--event-url",
        default="http://127.0.0.1:8787/v1/events",
        help="deprecated legacy event endpoint; retained for CLI compatibility",
    )
    parser.add_argument(
        "--committed-voice-url",
        default="http://127.0.0.1:8787/v1/voice/committed",
        help="typed best-effort PM intake for committed TurnOwner decisions",
    )
    parser.add_argument(
        "--pm-publisher",
        choices=("legacy", "candidate"),
        default="candidate",
        help="hot-select guarded committed-turn delivery; failures roll back",
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
        "--live-acknowledgments",
        nargs="*",
        default=["I hear you.", "Got it.", "I'm with you."],
        metavar="TEXT",
        help="immediate local phrases cycled while the reasoning lane responds; pass no values to disable",
    )
    parser.add_argument(
        "--barge-in",
        default="final",
        choices=("off", "final", "sustained", "immediate"),
        help=(
            "final buffers the next utterance while replies finish; sustained "
            "and immediate may preempt playback; off disables overlap"
        ),
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
        "--debug-events",
        type=Path,
        help="append live speech/ASR/routing lifecycle JSONL for dashboards",
    )
    parser.add_argument(
        "--health",
        type=Path,
        help="atomically update a privacy-safe stage heartbeat for supervision",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="keep live ASR/debug output running without attaching a Codex thread",
    )
    parser.add_argument(
        "--mic-mode",
        choices=("push-to-talk", "continuous", "muted"),
        default="continuous",
        help="initial capture policy; change live with voice/control",
    )
    parser.add_argument(
        "--notification-mode",
        choices=("conversational", "updates", "critical"),
        default="conversational",
        help="initial proactive spoken-notification density",
    )
    parser.add_argument(
        "--control-socket",
        type=Path,
        default=default_control_socket(),
        help="private Unix socket used by voice/control",
    )
    parser.add_argument(
        "--keyboard-indicator",
        choices=("none", "auto", "wooting"),
        default="none",
        help="opt-in Wooting Esc privacy/status light",
    )
    parser.add_argument(
        "--wooting-rgb-lib",
        type=Path,
        help="path to the official libwooting-rgb-sdk.so",
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
