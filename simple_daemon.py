#!/usr/bin/env python3
"""Executable, deliberately sequential voice recovery daemon."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app_server import AppServerError, CodexAppServer
from conversation import KOKORO_URL, ListenConfig, kokoro_wav, listen, transcribe
from simple import SimpleVoiceSession


def acquire_singleton(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_CLOEXEC | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(f"another voice daemon owns {path}") from None
    return descriptor


class Capture(Protocol):
    async def one(self) -> np.ndarray | None: ...


class Recognizer(Protocol):
    async def text(self, audio: np.ndarray) -> str: ...


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Status:
    """Small transcript-free health and metrics sink."""

    def __init__(
        self,
        health_path: Path | None,
        metrics_path: Path | None,
        clock=time.monotonic,
        *,
        heartbeat_seconds: float = 2.0,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.health_path = health_path
        self.metrics_path = metrics_path
        self.clock = clock
        self.started = clock()
        self.turn = 0
        self.phase = "starting"
        self.error: str | None = None
        self.heartbeat_seconds = heartbeat_seconds

    def health(self, phase: str, *, error: str | None = None) -> None:
        self.phase, self.error = phase, error
        if self.health_path:
            _atomic_json(
                self.health_path,
                {
                    "schema": 1,
                    "runtime": "simple",
                    "phase": phase,
                    "turns": self.turn,
                    "uptime_seconds": round(self.clock() - self.started, 6),
                    "updated_ns": time.time_ns(),
                    "pid": os.getpid(),
                    "healthy": error is None and phase not in {"error", "stopped"},
                    **({"error": error} if error else {}),
                },
            )

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            self.health(self.phase, error=self.error)

    def metric(self, durations: dict[str, float]) -> None:
        if not self.metrics_path:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": 1,
            "runtime": "simple",
            "turn": self.turn,
            **{key: round(value, 6) for key, value in durations.items()},
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


class LocalCapture:
    def __init__(self, config: ListenConfig) -> None:
        self.config = config

    async def one(self) -> np.ndarray | None:
        return await asyncio.to_thread(listen, self.config)


class WhisperRecognizer:
    def __init__(self, model: Any, *, language: str, lexicon: Any) -> None:
        self.model, self.language, self.lexicon = model, language, lexicon

    async def text(self, audio: np.ndarray) -> str:
        return await asyncio.to_thread(
            transcribe,
            self.model,
            audio,
            self.language,
            lexicon=self.lexicon,
            accurate=True,
        )


class PipeWireKokoro:
    """Synthesize a complete reply, then play it on one PipeWire sink."""

    def __init__(
        self,
        base_url: str,
        voice: str,
        *,
        output: str | None,
        latency: str,
    ) -> None:
        self.base_url, self.voice = base_url, voice
        self.output, self.latency = output, latency

    async def say(self, text: str) -> None:
        wav = await asyncio.to_thread(
            kokoro_wav,
            text,
            base_url=self.base_url,
            voice=self.voice,
        )
        command = ["pw-play", "--latency", self.latency]
        if self.output:
            command.extend(("--target", self.output))
        command.append("-")
        player = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            _, stderr = await player.communicate(wav)
        except asyncio.CancelledError:
            if player.returncode is None:
                player.terminate()
                try:
                    await asyncio.wait_for(player.wait(), timeout=0.25)
                except TimeoutError:
                    player.kill()
                    await player.wait()
            raise
        if player.returncode:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"PipeWire playback failed: {detail or player.returncode}"
            )


class SimpleVoiceDaemon:
    def __init__(
        self,
        session: SimpleVoiceSession,
        capture: Capture,
        recognizer: Recognizer,
        *,
        status: Status | None = None,
        emit: Callable[[str], None] = print,
        response_timeout: float = 30.0,
    ) -> None:
        if response_timeout <= 0:
            raise ValueError("response_timeout must be positive")
        self.session, self.capture, self.recognizer = session, capture, recognizer
        self.status, self.emit = status or Status(None, None), emit
        self.response_timeout = response_timeout

    async def run(self, *, once: bool = False, max_turns: int | None = None) -> int:
        self.status.health("listening")
        heartbeat = asyncio.create_task(
            self.status.heartbeat(),
            name="simple-voice-health-heartbeat",
        )
        completed = 0
        try:
            while max_turns is None or completed < max_turns:
                started = time.monotonic()
                audio = await self.capture.one()
                captured = time.monotonic()
                if audio is None or not audio.size:
                    continue
                self.status.health("transcribing")
                prompt = (await self.recognizer.text(audio)).strip()
                recognized = time.monotonic()
                if not prompt:
                    self.status.health("listening")
                    continue
                self.emit(f"You: {prompt}")
                self.status.health("responding")
                try:
                    response = await asyncio.wait_for(
                        self.session.respond(prompt),
                        timeout=self.response_timeout,
                    )
                except TimeoutError:
                    self.emit("Codex: response timed out; listening again.")
                    self.status.health("listening")
                    continue
                finished = time.monotonic()
                completed += 1
                self.status.turn = completed
                self.emit(f"Codex: {response}")
                self.status.metric(
                    {
                        "capture_seconds": captured - started,
                        "asr_seconds": recognized - captured,
                        "response_and_audio_seconds": finished - recognized,
                        "total_seconds": finished - started,
                    }
                )
                self.status.health("listening")
                if once:
                    break
        except BaseException as error:
            self.status.health("error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            await self.session.close()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        self.status.health("stopped")
        return completed


async def resolve_thread(
    server: CodexAppServer, cwd: Path, requested: str | None, timeout: float
) -> str:
    thread_id = requested or os.environ.get("CODEX_THREAD_ID")
    if not thread_id:
        threads = await asyncio.wait_for(server.list_threads(cwd), timeout)
        if not threads:
            raise AppServerError(
                "no existing Codex harness thread; pass --session or set CODEX_THREAD_ID"
            )
        thread_id = str(threads[0]["id"])
    return await asyncio.wait_for(server.resume_thread(thread_id, cwd=cwd), timeout)


def smoke(args: argparse.Namespace) -> int:
    checks = {
        "cwd": args.cwd.is_dir(),
        "pw_play": shutil.which("pw-play") is not None,
        "whisper_python": Path(sys.executable).exists(),
    }
    if args.hardware:
        try:
            with urllib.request.urlopen(
                f"{args.kokoro_url.rstrip('/')}/health", timeout=2
            ) as response:
                checks["kokoro"] = json.load(response).get("status") == "healthy"
        except Exception:
            checks["kokoro"] = False
        try:
            import sounddevice as sd

            sd.check_input_settings(samplerate=16_000, channels=1, dtype="float32")
            checks["microphone"] = True
        except Exception:
            checks["microphone"] = False
    print(
        json.dumps(
            {
                "schema": 1,
                "mode": "hardware" if args.hardware else "dry-run",
                "checks": checks,
            },
            sort_keys=True,
        )
    )
    return 0 if all(checks.values()) else 2


async def run_real(args: argparse.Namespace) -> int:
    from faster_whisper import WhisperModel

    from lexicon import load_lexicon

    model = WhisperModel(
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=True,
    )
    context = CodexAppServer(
        cwd=args.cwd, shared=args.app_server == "shared", startup_timeout=args.timeout
    )
    async with context as server:
        thread = await resolve_thread(server, args.cwd, args.session, args.timeout)
        session = await SimpleVoiceSession.attach(
            server,
            thread,
            cwd=args.cwd,
            speaker=PipeWireKokoro(
                args.kokoro_url,
                args.voice,
                output=args.output,
                latency=args.playback_latency,
            ),
            model=args.model,
            developer_instructions=(
                "Reply naturally and concisely for a realtime spoken conversation."
            ),
            effort=args.effort,
        )
        daemon = SimpleVoiceDaemon(
            session,
            LocalCapture(
                ListenConfig(
                    threshold=args.threshold,
                    silence_ms=args.silence_ms,
                    max_seconds=args.max_seconds,
                )
            ),
            WhisperRecognizer(model, language=args.language, lexicon=load_lexicon()),
            status=Status(args.health, args.metrics),
            response_timeout=args.response_timeout,
        )
        return await daemon.run(once=args.once)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cwd", type=Path, default=Path.cwd())
    result.add_argument("--session")
    result.add_argument("--kokoro-url", default=KOKORO_URL)
    result.add_argument("--voice", default="af_heart")
    result.add_argument("--output")
    result.add_argument("--playback-latency", default="40ms")
    result.add_argument("--whisper-model", default="small.en")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--compute-type", default="float16")
    result.add_argument("--language", default="en")
    result.add_argument("--threshold", type=float, default=0.018)
    result.add_argument("--silence-ms", type=int, default=850)
    result.add_argument("--max-seconds", type=float, default=45.0)
    result.add_argument("--model")
    result.add_argument("--effort", default="low")
    result.add_argument("--app-server", choices=("private", "shared"), default="shared")
    result.add_argument("--timeout", type=float, default=8.0)
    result.add_argument("--response-timeout", type=float, default=30.0)
    result.add_argument("--health", type=Path)
    result.add_argument("--metrics", type=Path)
    result.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/zer0-{os.getuid()}"))
        / "zer0-voice/simple.lock",
    )
    result.add_argument("--once", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--hardware-smoke", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.smoke or args.hardware_smoke:
        args.hardware = args.hardware_smoke
        return smoke(args)
    lock_descriptor = -1
    try:
        lock_descriptor = acquire_singleton(args.lock)
        return asyncio.run(run_real(args))
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 0
    except Exception as error:
        print(f"simple voice fatal: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
