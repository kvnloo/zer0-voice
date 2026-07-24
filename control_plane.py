"""Small local control plane for live voice modes."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from modes import MicMode, NotificationMode, VoiceModes


def default_control_socket() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/zer0-voice-{os.getuid()}"
    return Path(runtime) / "zer0-voice" / "control.sock"


@dataclass(frozen=True, slots=True)
class RuntimeState:
    modes: VoiceModes = VoiceModes()
    push_held: bool = False
    live_model: str | None = None
    live_effort: str = "low"
    revision: int = 0

    @property
    def capture_active(self) -> bool:
        return self.modes.should_capture(push_held=self.push_held)

    def json(self) -> dict[str, object]:
        return {
            "schema": 1,
            "mic": self.modes.mic.value,
            "notifications": self.modes.notifications.value,
            "push_held": self.push_held,
            "live_model": self.live_model,
            "live_effort": self.live_effort,
            "capture_active": self.capture_active,
            "revision": self.revision,
        }


class VoiceControl:
    def __init__(self, modes: VoiceModes | None = None):
        self.state = RuntimeState(modes=modes or VoiceModes())
        self.changed = asyncio.Condition()

    async def apply(self, command: dict[str, object]) -> RuntimeState:
        state = self.state
        modes = state.modes
        if "mic" in command:
            modes = replace(modes, mic=MicMode(str(command["mic"])))
        if "notifications" in command:
            modes = replace(
                modes,
                notifications=NotificationMode(str(command["notifications"])),
            )
        live_model = state.live_model
        if "live_model" in command:
            value = command["live_model"]
            live_model = None if value is None else str(value)
        live_effort = state.live_effort
        if "live_effort" in command:
            live_effort = str(command["live_effort"])
        push_held = state.push_held
        if "push_held" in command:
            push_held = bool(command["push_held"])
        candidate = RuntimeState(
            modes=modes,
            push_held=push_held,
            live_model=live_model,
            live_effort=live_effort,
            revision=state.revision,
        )
        if candidate != state:
            candidate = replace(candidate, revision=state.revision + 1)
            async with self.changed:
                self.state = candidate
                self.changed.notify_all()
        return self.state

    async def wait_after(self, revision: int) -> RuntimeState:
        async with self.changed:
            await self.changed.wait_for(lambda: self.state.revision > revision)
            return self.state


class ControlServer:
    def __init__(self, path: Path, control: VoiceControl):
        self.path = path
        self.control = control
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
            self.server = None
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
            state = await self.control.apply(command)
            response = {"ok": True, **state.json()}
        except (TimeoutError, ValueError, json.JSONDecodeError) as error:
            response = {"ok": False, "error": str(error)}
        writer.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()


async def request(path: Path, command: dict[str, object]) -> dict[str, object]:
    reader, writer = await asyncio.open_unix_connection(path)
    writer.write(json.dumps(command, separators=(",", ":")).encode() + b"\n")
    await writer.drain()
    raw = await reader.readline()
    writer.close()
    await writer.wait_closed()
    response = json.loads(raw)
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "voice control failed")))
    return response
