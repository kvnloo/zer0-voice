#!/usr/bin/env python3
"""Full-duplex local voice runtime: mic -> Whisper -> Codex -> Kokoro."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
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
from orchestrator import OllamaLiveLane
from workspace_router import Route, WorkspaceRouter, load_routes


@dataclass(frozen=True)
class AudioEvent:
    kind: str
    audio: np.ndarray | None = None


def append_metric(path: Path, metric: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, separators=(",", ":"), sort_keys=True) + "\n")


class UtteranceDetector:
    """Stateful endpoint detector that reports speech start before final audio."""

    def __init__(self, config: ListenConfig):
        self.config = config
        self.before: deque[np.ndarray] = deque(maxlen=config.pre_roll_blocks)
        self.utterance: list[np.ndarray] = []
        self.voiced = 0
        self.silent = 0
        self.active = False

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
                self.silent = 0
                events.append(AudioEvent("speech.started"))
            return events

        self.utterance.append(block)
        self.silent = 0 if voice else self.silent + 1
        max_blocks = int(self.config.max_seconds * 1000 / self.config.block_ms)
        if self.silent >= self.config.silence_blocks or len(self.utterance) >= max_blocks:
            keep = max(0, len(self.utterance) - self.silent)
            audio = np.concatenate(self.utterance[:keep]) if keep else None
            self.reset()
            if audio is not None:
                events.append(AudioEvent("speech.final", audio))
        return events

    def reset(self) -> None:
        self.before.clear()
        self.utterance = []
        self.voiced = self.silent = 0
        self.active = False


def adaptive_threshold(levels: np.ndarray) -> float:
    """Choose a conservative speech threshold from a short ambient sample."""
    values = np.asarray(levels, dtype=np.float32).reshape(-1)
    if not values.size:
        return 0.018
    noise = float(np.percentile(values, 95))
    return min(0.03, max(0.004, noise * 3.0))


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

    async def say(self, text: str) -> None:
        if self.cancel.is_set() or not text:
            return
        synthesis_started = time.monotonic()
        audio = await asyncio.to_thread(
            kokoro_wav, text, base_url=self.base_url, voice=self.voice
        )
        if self.first_synthesis_seconds is None:
            self.first_synthesis_seconds = time.monotonic() - synthesis_started
        if self.cancel.is_set():
            return
        if self.first_play_at is None:
            self.first_play_at = time.monotonic()
        if self.backend == "pipewire":
            await asyncio.to_thread(self._pipewire_play, audio)
            return
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


class LiveTurn:
    """Immediate local response lane; deep Codex work never blocks its speech."""

    def __init__(
        self,
        lane: OllamaLiveLane,
        speaker: Speaker,
        context: tuple[str, ...],
    ):
        self.lane = lane
        self.speaker = speaker
        self.context = context

    async def run(self, text: str) -> str:
        chunks: asyncio.Queue[str | None] = asyncio.Queue()
        response: list[str] = []

        async def generate() -> None:
            chunker = SentenceChunker()
            async for delta in self.lane.stream(text, self.context):
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


class DeepWorker:
    """Sequential persistent Codex lane that enriches future live turns."""

    def __init__(self, server: CodexAppServer):
        self.server = server
        self.queue: asyncio.Queue[
            tuple[str, str, str, tuple[str, ...]] | None
        ] = asyncio.Queue()
        self.insights: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=8)
        )

    def submit(
        self,
        route_key: str,
        thread: str,
        text: str,
        context: tuple[str, ...],
    ) -> None:
        self.queue.put_nowait((route_key, thread, text, context))

    async def run(self) -> None:
        while item := await self.queue.get():
            route_key, thread, text, context = item
            prompt = (
                "Analyze the latest live conversation turn. Do useful tool work when "
                "appropriate. Return a compact verified result or correction that the "
                f"live lane should know next. This turn belongs to route {route_key!r}; "
                "keep all actions and assumptions scoped to that harness.\n\n"
                f"Context:\n{chr(10).join(context)}\n\nLatest user: {text}"
            )
            response = []
            async for event in self.server.stream_turn(
                thread, prompt, effort="medium"
            ):
                if event.kind == "assistant.delta":
                    response.append(str(event.payload["text"]))
            insight = "".join(response).strip()
            if insight:
                self.insights[route_key].append(insight)
                print(f"Deep[{route_key}]: {insight}", flush=True)


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
        instructions: str,
        timeout: float,
    ):
        self.server = server
        self.workspace = workspace
        self.fallback_thread = fallback_thread
        self.instructions = instructions
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
                return ThreadBinding(
                    binding.key,
                    binding.thread,
                    route,
                    "spoken_pin",
                )
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
                        developer_instructions=self.instructions,
                    ),
                    timeout=self.timeout,
                )
            else:
                thread = await self.server.start_thread(
                    cwd=route.cwd,
                    developer_instructions=self.instructions,
                )
            print(
                f"Voice route {route.project} pane={route.pane_id} thread={thread}",
                flush=True,
            )
            return ThreadBinding(route.project, thread, route, reason)
        except (TimeoutError, AppServerError, KeyError) as exc:
            print(
                f"Could not attach {route.project} harness ({exc or 'timed out'}); "
                "using an isolated project thread.",
                flush=True,
            )
            thread = await self.server.start_thread(
                cwd=route.cwd,
                developer_instructions=self.instructions,
            )
            return ThreadBinding(route.project, thread, route, "isolated_fallback")


async def next_final(
    mic: Microphone,
    turn: DuplexTurn | LiveTurn | None = None,
    *,
    interrupt_on_start: bool = False,
) -> np.ndarray:
    while True:
        event = await mic.next()
        if event.kind == "speech.started" and turn and interrupt_on_start:
            await turn.interrupt()
        if event.kind == "speech.final" and event.audio is not None:
            return event.audio


async def run(args) -> None:
    from faster_whisper import WhisperModel

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
    live = OllamaLiveLane(model=args.live_model)
    print(f"Warming live model {args.live_model}…", flush=True)
    async for _ in live.stream("Reply with only: ready.", ()):
        pass
    async with CodexAppServer(cwd=args.cwd) as server:
        instructions = (
            "You are the deep intelligence and tool lane behind a live local voice "
            "model. Work rigorously, then return concise verified steering."
        )
        deep_thread = None
        if args.session:
            try:
                deep_thread = await asyncio.wait_for(
                    server.resume_thread(
                        args.session,
                        developer_instructions=instructions,
                    ),
                    timeout=args.session_timeout,
                )
                print(f"Deep lane attached to Codex thread {deep_thread}.", flush=True)
            except (TimeoutError, AppServerError) as exc:
                print(
                    "Could not attach the requested Codex thread "
                    f"({exc or 'timed out'}); using a dedicated deep thread.",
                    flush=True,
                )
        if deep_thread is None:
            deep_thread = await server.start_thread(developer_instructions=instructions)
            print(f"Deep lane started Codex thread {deep_thread}.", flush=True)
        workspace = (
            WorkspaceRouter(load_routes(args.routes))
            if args.workspace_routing
            else None
        )
        harness_router = HarnessRouter(
            server,
            workspace,
            deep_thread,
            instructions=instructions,
            timeout=args.session_timeout,
        )
        deep = DeepWorker(server)
        deep_task = asyncio.create_task(deep.run())
        histories: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=16))
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
                binding = await harness_router.resolve(text)
                print(f"You[{binding.key}]: {text}", flush=True)
                history = histories[binding.key]
                route_note = (
                    f"system: This spoken turn belongs to the {binding.key!r} harness. "
                    f"Routing reason: {binding.reason}. Keep the response scoped to "
                    "that project and do not claim work in another harness."
                )
                context = (route_note,) + tuple(history) + tuple(
                    f"deep: {x}" for x in deep.insights[binding.key]
                )
                deep.submit(binding.key, binding.thread, text, context)
                speaker = Speaker(
                    args.kokoro_url,
                    args.voice,
                    backend=args.playback,
                    output=args.output,
                    latency=args.playback_latency,
                )
                turn = LiveTurn(live, speaker, context)
                task = asyncio.create_task(turn.run(text))
                listen_task = asyncio.create_task(
                    next_final(
                        mic,
                        turn,
                        interrupt_on_start=args.barge_in == "immediate",
                    )
                )
                done, _ = await asyncio.wait(
                    (task, listen_task), return_when=asyncio.FIRST_COMPLETED
                )
                if listen_task in done:
                    pending_audio = listen_task.result()
                    await turn.interrupt()
                else:
                    listen_task.cancel()
                response = await task
                history.extend((f"user: {text}", f"assistant: {response}"))
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
        finally:
            deep.queue.put_nowait(None)
            await deep_task
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
    parser.add_argument("--session", help="resume this Codex thread for the deep lane")
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
    parser.add_argument("--live-model", default="qwen2.5:3b")
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
        default="final",
        choices=("final", "immediate"),
        help="final avoids false echo interruptions; immediate yields on speech start",
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
    parser.add_argument("--silence-ms", type=int, default=520)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
