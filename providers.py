"""Streaming BYO-LLM adapters for subscriptions and OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: str
    turn_id: str
    text: str = ""


class EmptyCompletionError(RuntimeError):
    """The transport completed without producing assistant speech text."""


class Provider(Protocol):
    async def stream(
        self,
        text: str,
        *,
        context: tuple[str, ...] = (),
        effort: str | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def interrupt(self, turn_id: str) -> None: ...


class CodexSubscription:
    """Adapt an authenticated Codex app-server thread to the provider contract."""

    def __init__(self, server, thread: str):
        self.server, self.thread = server, thread

    async def stream(
        self,
        text: str,
        *,
        context: tuple[str, ...] = (),
        effort: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        prompt = "\n".join((*context, text)) if context else text
        output_chars = 0
        async for event in self.server.stream_turn(self.thread, prompt, effort=effort):
            turn = event.subject.removeprefix("turn:")
            if event.kind == "assistant.delta":
                delta = str(event.payload["text"])
                if not delta:
                    continue
                output_chars += len(delta)
                yield StreamEvent("delta", turn, delta)
            elif event.kind == "assistant.completed":
                if output_chars == 0:
                    raise EmptyCompletionError(
                        f"turn {turn} completed without assistant text"
                    )
                yield StreamEvent("completed", turn)

    async def interrupt(self, turn_id: str) -> None:
        await self.server.interrupt(self.thread, turn_id)


class OpenAICompatible:
    """Stream any `/v1/chat/completions` provider using only the standard library."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        timeout: float = 120,
    ):
        self.url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.responses: dict[str, object] = {}
        self.lock = threading.Lock()

    async def stream(
        self,
        text: str,
        *,
        context: tuple[str, ...] = (),
        effort: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del effort
        turn = uuid.uuid4().hex
        queue: asyncio.Queue[StreamEvent | BaseException | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(item: StreamEvent | BaseException | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            messages = []
            if context:
                messages.append({"role": "system", "content": "\n".join(context)})
            messages.append({"role": "user", "content": text})
            body = json.dumps(
                {"model": self.model, "messages": messages, "stream": True}
            ).encode()
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(
                self.url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    with self.lock:
                        self.responses[turn] = response
                    for raw in response:
                        line = raw.decode(errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            break
                        payload = json.loads(data)
                        delta = payload["choices"][0].get("delta", {}).get("content")
                        if delta:
                            emit(StreamEvent("delta", turn, str(delta)))
                emit(StreamEvent("completed", turn))
            except BaseException as error:
                emit(error)
            finally:
                with self.lock:
                    self.responses.pop(turn, None)
                emit(None)

        threading.Thread(target=worker, daemon=True).start()
        while (item := await queue.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def interrupt(self, turn_id: str) -> None:
        with self.lock:
            response = self.responses.get(turn_id)
        if response is not None:
            await asyncio.to_thread(response.close)
