#!/usr/bin/env python3
"""Long-lived owner for hot-swappable continuous-voice generations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from control_plane import default_control_socket, request
from handoff import Candidate, HandoffError, handoff, wait_ready
from health import assess, read_snapshot
from release import read_pointer, resolve_production


@dataclass(slots=True)
class Generation:
    digest: str
    bundle: Path
    candidate: Candidate
    process: asyncio.subprocess.Process


class ControlProxy:
    """Keep keybinds and dashboards attached while workers are replaced."""

    def __init__(self, path: Path, active: Callable[[], Generation | None]):
        self.path = path
        self.active = active
        self.server: asyncio.Server | None = None

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.server = await asyncio.start_unix_server(self._handle, self.path)
        os.chmod(self.path, 0o600)

    async def close(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.path.unlink(missing_ok=True)

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=1)
            command = json.loads(raw or b"{}")
            if not isinstance(command, dict):
                raise ValueError("command must be an object")
            generation = self.active()
            if generation is None:
                raise RuntimeError("voice generation unavailable")
            response = await asyncio.wait_for(
                request(generation.candidate.control, command),
                timeout=1,
            )
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            response = {"ok": False, "error": type(error).__name__}
        writer.write(
            json.dumps(response, separators=(",", ":")).encode() + b"\n"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()


class RuntimeManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.active: Generation | None = None
        self.rollback: Generation | None = None
        self.rejected_digest: str | None = None
        self.generation = 0
        self.stopping = asyncio.Event()
        self.proxy = ControlProxy(args.control_socket, lambda: self.active)

    @staticmethod
    def url_ready(url: str, timeout: float = 0.5) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if response.status >= 300:
                    return False
                payload = json.loads(response.read())
                return payload.get("status") == "healthy"
        except (OSError, ValueError):
            return False

    async def ensure_kokoro(self) -> None:
        health = f"{self.args.kokoro_url.rstrip('/')}/health"
        if await asyncio.to_thread(self.url_ready, health):
            return
        launcher = self.args.kokoro_launcher
        if not launcher.is_file() or not os.access(launcher, os.X_OK):
            raise RuntimeError("Kokoro is unavailable and launcher is missing")
        repair = await asyncio.create_subprocess_exec(str(launcher), "start")
        if await repair.wait() != 0:
            raise RuntimeError("Kokoro launcher failed")
        deadline = time.monotonic() + self.args.dependency_timeout
        while time.monotonic() < deadline:
            if await asyncio.to_thread(self.url_ready, health):
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("Kokoro recovery timed out")

    def command(self, bundle: Path, state: Path, mic_mode: str) -> list[str]:
        return [
            str(bundle / "voice/duplex"),
            "--session",
            self.args.thread,
            "--cwd",
            str(self.args.root),
            "--app-server",
            "shared",
            "--no-workspace-routing",
            "--kokoro-url",
            self.args.kokoro_url,
            "--committed-voice-url",
            f"{self.args.relay_url}/v1/voice/committed",
            "--pm-publisher",
            "candidate",
            "--input",
            self.args.input,
            "--output",
            self.args.output,
            "--mic-mode",
            mic_mode,
            "--notification-mode",
            self.args.notification_mode,
            "--live-model",
            self.args.live_model,
            "--live-effort",
            self.args.live_effort,
            "--live-timeout",
            str(self.args.live_timeout),
            "--live-acknowledgments",
            "--barge-in",
            self.args.barge_in,
            "--startup-phrase",
            "",
            "--control-socket",
            str(state / "control.sock"),
            "--keyboard-indicator",
            self.args.keyboard_indicator,
            "--metrics",
            str(self.args.metrics),
            "--debug-events",
            str(self.args.debug_events),
            "--health",
            str(state / "health.json"),
        ]

    async def launch(
        self,
        bundle: Path,
        digest: str,
        mic_mode: str,
    ) -> Generation:
        await self.ensure_kokoro()
        self.generation += 1
        state = self.args.state / "generations" / f"{self.generation:08d}"
        state.mkdir(parents=True, exist_ok=False)
        process = await asyncio.create_subprocess_exec(
            *self.command(bundle, state, mic_mode),
        )
        return Generation(
            digest,
            bundle,
            Candidate(state / "control.sock", state / "health.json"),
            process,
        )

    async def activate_without_predecessor(self, generation: Generation) -> None:
        await wait_ready(
            generation.candidate,
            timeout=self.args.readiness_timeout,
        )
        status = await request(
            generation.candidate.control,
            {"mic": self.args.mic_mode},
        )
        if status.get("capture_active") is not True:
            raise HandoffError("replacement failed to acquire capture")
        self.active = generation

    async def hot_swap(self, digest: str, bundle: Path) -> bool:
        assert self.active is not None
        candidate = await self.launch(bundle, digest, "muted")
        try:
            await handoff(
                self.active.candidate,
                candidate.candidate,
                readiness_timeout=self.args.readiness_timeout,
            )
        except BaseException:
            await self.terminate(candidate)
            raise
        old = self.active
        self.active = candidate
        self.rollback = old
        healthy = await self.probation(candidate)
        if healthy:
            await self.terminate(old)
            self.rollback = None
            return True
        try:
            await request(candidate.candidate.control, {"mic": "muted"})
            restored = await request(
                old.candidate.control,
                {"mic": self.args.mic_mode},
            )
            if restored.get("capture_active") is not True:
                raise HandoffError("probation rollback failed")
            self.active = old
            self.rejected_digest = digest
        finally:
            await self.terminate(candidate)
            self.rollback = None
        return False

    async def probation(self, generation: Generation) -> bool:
        deadline = time.monotonic() + self.args.probation
        while time.monotonic() < deadline:
            if generation.process.returncode is not None:
                return False
            healthy, _ = assess(read_snapshot(generation.candidate.health))
            if not healthy:
                return False
            await asyncio.sleep(self.args.poll_interval)
        return True

    async def terminate(self, generation: Generation | None) -> None:
        if generation is None or generation.process.returncode is not None:
            return
        generation.process.terminate()
        try:
            await asyncio.wait_for(generation.process.wait(), timeout=2)
        except TimeoutError:
            generation.process.kill()
            await generation.process.wait()

    async def recover(self) -> None:
        assert self.active is not None
        failed = self.active
        replacement = await self.launch(failed.bundle, failed.digest, "muted")
        try:
            await self.activate_without_predecessor(replacement)
        except BaseException:
            await self.terminate(replacement)
            raise
        await self.terminate(failed)

    async def reconcile_release(self) -> bool | None:
        """Apply one verified pointer change without endangering the active lane."""
        assert self.active is not None
        pointer = read_pointer(self.args.release_state)
        digest = str(pointer["bundle_sha256"]) if pointer else ""
        if digest == self.active.digest:
            self.rejected_digest = None
            return None
        if not digest or digest == self.rejected_digest:
            return None
        try:
            return await self.hot_swap(
                digest,
                resolve_production(self.args.release_state),
            )
        except (HandoffError, OSError, RuntimeError):
            self.rejected_digest = digest
            raise

    async def run(self) -> None:
        pointer = read_pointer(self.args.release_state)
        if not pointer:
            raise RuntimeError("no production voice bundle is pinned")
        bundle = resolve_production(self.args.release_state)
        initial = await self.launch(
            bundle,
            str(pointer["bundle_sha256"]),
            "muted",
        )
        await self.activate_without_predecessor(initial)
        await self.proxy.start()
        failures = 0
        next_dependency_check = 0.0
        try:
            while not self.stopping.is_set():
                await asyncio.sleep(self.args.poll_interval)
                assert self.active is not None
                now = time.monotonic()
                if now >= next_dependency_check:
                    next_dependency_check = now + self.args.dependency_interval
                    try:
                        await self.ensure_kokoro()
                    except (OSError, RuntimeError) as error:
                        failures += 1
                        print(
                            f"voice dependency recovery deferred: "
                            f"{type(error).__name__}",
                            file=sys.stderr,
                            flush=True,
                        )
                        await asyncio.sleep(min(8, failures))
                        continue
                healthy, _ = assess(read_snapshot(self.active.candidate.health))
                if self.active.process.returncode is not None or not healthy:
                    try:
                        await self.recover()
                        failures = 0
                    except (HandoffError, OSError, RuntimeError) as error:
                        failures += 1
                        print(
                            f"voice recovery deferred: {type(error).__name__}",
                            file=sys.stderr,
                            flush=True,
                        )
                        await asyncio.sleep(min(8, failures))
                    continue
                try:
                    await self.reconcile_release()
                    failures = 0
                except (HandoffError, OSError, RuntimeError) as error:
                    failures += 1
                    print(
                        f"voice rollout deferred: {type(error).__name__}",
                        file=sys.stderr,
                        flush=True,
                    )
                    await asyncio.sleep(min(8, failures))
        finally:
            await self.proxy.close()
            await self.terminate(self.active)
            await self.terminate(self.rollback)


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    state = Path(
        os.environ.get(
            "ZERO_VOICE_STATE_DIR",
            Path.home() / ".local/state/zer0-voice",
        )
    )
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("thread")
    result.add_argument("--root", type=Path, default=root)
    result.add_argument("--state", type=Path, default=state)
    result.add_argument(
        "--release-state",
        type=Path,
        default=state / "releases",
    )
    result.add_argument(
        "--control-socket",
        type=Path,
        default=Path(
            os.environ.get(
                "ZERO_VOICE_CONTROL_SOCKET",
                default_control_socket(),
            )
        ),
    )
    result.add_argument("--kokoro-url", default="http://127.0.0.1:8880")
    result.add_argument(
        "--kokoro-launcher",
        type=Path,
        default=Path("/workspace/kokoro-tts/kokoro.sh"),
    )
    result.add_argument("--dependency-timeout", type=float, default=60)
    result.add_argument("--dependency-interval", type=float, default=1)
    result.add_argument("--relay-url", default="http://127.0.0.1:8787")
    result.add_argument("--input", default="default")
    result.add_argument("--output", default="effect_input.aural_evolution")
    result.add_argument("--mic-mode", default="continuous")
    result.add_argument("--notification-mode", default="conversational")
    result.add_argument("--live-model", default="gpt-5.6-luna")
    result.add_argument("--live-effort", default="low")
    result.add_argument("--live-timeout", type=float, default=35)
    result.add_argument("--barge-in", default="final")
    result.add_argument("--keyboard-indicator", default="none")
    result.add_argument("--readiness-timeout", type=float, default=180)
    result.add_argument("--probation", type=float, default=10)
    result.add_argument("--poll-interval", type=float, default=0.1)
    result.add_argument("--metrics", type=Path, default=root / "bench/voice-history.jsonl")
    result.add_argument("--debug-events", type=Path, default=state / "voice-debug.jsonl")
    return result


async def async_main(args: argparse.Namespace) -> None:
    manager = RuntimeManager(args)
    loop = asyncio.get_running_loop()
    for name in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(name, manager.stopping.set)
    await manager.run()


def main() -> int:
    asyncio.run(async_main(parser().parse_args()))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
