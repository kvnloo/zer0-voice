"""Parallel live-conversation and deep-reasoning coordinator."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import urllib.request
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class VoiceEvent:
    """Provider-neutral event shared by the harness, canvas, and PM engine."""

    type: str
    turn_id: str
    source: str
    timestamp: float = field(default_factory=time.time)
    text: str | None = None
    data: dict[str, object] = field(default_factory=dict)
    version: int = 1

    def json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class LiveLane(Protocol):
    def stream(self, text: str, context: tuple[str, ...]) -> AsyncIterator[str]: ...


class ReasoningLane(Protocol):
    async def reason(self, text: str, context: tuple[str, ...]) -> str: ...


EventSink = Callable[[VoiceEvent], Awaitable[None]]


class TurnCoordinator:
    """Fan one utterance out without letting deep reasoning block live speech."""

    def __init__(
        self,
        live: LiveLane,
        reasoning: ReasoningLane,
        sink: EventSink,
        *,
        context_turns: int = 12,
    ):
        self.live = live
        self.reasoning = reasoning
        self.sink = sink
        self.context_turns = context_turns
        self.history: list[str] = []

    async def handle(self, text: str) -> tuple[str, str]:
        turn_id = str(uuid.uuid4())
        context = tuple(self.history[-self.context_turns :])
        await self.sink(VoiceEvent("user.final", turn_id, "asr", text=text))

        deep_task = asyncio.create_task(self.reasoning.reason(text, context))
        spoken: list[str] = []
        async for delta in self.live.stream(text, context):
            spoken.append(delta)
            await self.sink(
                VoiceEvent("assistant.live.delta", turn_id, "live", text=delta)
            )
        live_text = "".join(spoken).strip()
        await self.sink(
            VoiceEvent("assistant.live.final", turn_id, "live", text=live_text)
        )

        deep_text = await deep_task
        await self.sink(
            VoiceEvent(
                "assistant.reasoning.final",
                turn_id,
                "codex",
                text=deep_text,
                data={"disposition": "context_or_intervention"},
            )
        )
        self.history.extend((f"user: {text}", f"assistant: {live_text}"))
        return live_text, deep_text


class OllamaLiveLane:
    """Streaming local model adapter; no API key or per-token charge."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        url: str = "http://127.0.0.1:11434/api/chat",
    ):
        self.model = model
        self.url = url

    def _request(self, text: str, context: tuple[str, ...]):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Zer0's live conversational layer. Respond naturally "
                    "and briefly. Never pretend deep tool work is finished; say "
                    "you are checking when needed. Prefer one or two sentences."
                ),
            }
        ]
        for item in context:
            role, _, content = item.partition(": ")
            if role not in {"system", "assistant", "user"}:
                role = "user"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": text})
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "keep_alive": "30m",
                "options": {"num_predict": 96},
            }
        ).encode()
        return urllib.request.urlopen(
            urllib.request.Request(
                self.url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=120,
        )

    async def stream(
        self, text: str, context: tuple[str, ...]
    ) -> AsyncIterator[str]:
        response = await asyncio.to_thread(self._request, text, context)
        try:
            while True:
                line = await asyncio.to_thread(response.readline)
                if not line:
                    break
                payload = json.loads(line)
                delta = payload.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if payload.get("done"):
                    break
        finally:
            response.close()


class CodexReasoningLane:
    """Use the authenticated Codex subscription as the deep/tool lane."""

    def __init__(self, cwd: Path, session: str | None = None):
        self.cwd = cwd
        self.session = session

    def _run(self, text: str, context: tuple[str, ...]) -> str:
        prompt = (
            "You are the deep reasoning lane behind a live voice conversation. "
            "Analyze the user's latest statement, perform any useful project work, "
            "and return a concise intervention or result for the live lane. Do not "
            "repeat conversational filler.\n\n"
            f"Recent live context:\n{chr(10).join(context)}\n\nLatest user: {text}"
        )
        command = ["codex", "exec"]
        if self.session:
            command.extend(["resume", self.session])
        command.extend(
            [
                "--skip-git-repo-check",
                "--color",
                "never",
                "-C",
                str(self.cwd),
                prompt,
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Codex reasoning failed")
        return result.stdout.strip()

    async def reason(self, text: str, context: tuple[str, ...]) -> str:
        return await asyncio.to_thread(self._run, text, context)
