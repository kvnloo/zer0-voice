"""Small, fail-closed continuous voice conversation core.

This module is intentionally independent of the production daemon.  It owns one
conversation thread for its whole lifetime, serializes model turns and speech,
and exposes a single TurnOwner boundary for an upstream mic/ASR producer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from floor import SubmissionDecision, TurnOwner
from providers import CodexSubscription, EmptyCompletionError, Provider


class SpeechBackend(Protocol):
    async def say(self, text: str) -> None: ...


class SimpleVoiceError(RuntimeError):
    """A user-visible failure in the minimal conversation path."""


class SerialSpeaker:
    """One bounded FIFO around any Kokoro/Speaker backend."""

    def __init__(self, backend: SpeechBackend, *, capacity: int = 8) -> None:
        self.backend = backend
        self.queue: asyncio.Queue[tuple[str, asyncio.Future[None]] | None] = (
            asyncio.Queue(maxsize=capacity)
        )
        self.worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.worker is None:
            self.worker = asyncio.create_task(self._run(), name="simple-voice-speaker")

    async def say(self, text: str) -> None:
        if not text.strip():
            raise SimpleVoiceError("refusing to enqueue empty speech")
        self.start()
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((text, future))
        try:
            await future
        except asyncio.CancelledError:
            # A turn deadline owns the active speech job. Propagate
            # cancellation into the backend so stale audio cannot play over
            # the next listening window.
            if self.worker is not None:
                self.worker.cancel()
                await asyncio.gather(self.worker, return_exceptions=True)
                self.worker = None
            raise

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            text, future = item
            try:
                await self.backend.say(text)
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            else:
                if not future.done():
                    future.set_result(None)
            finally:
                self.queue.task_done()

    async def close(self) -> None:
        if self.worker is None:
            return
        await self.queue.put(None)
        await self.worker
        self.worker = None


@dataclass(frozen=True, slots=True)
class SimpleThread:
    authoritative: str
    conversation: str
    inherited_through_turn: str | None


class SimpleVoiceSession:
    """One capture owner, one inherited model thread, and one speech queue."""

    def __init__(
        self,
        binding: SimpleThread,
        provider: Provider,
        speaker: SerialSpeaker,
        *,
        owner: TurnOwner | None = None,
        effort: str = "low",
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.binding = binding
        self.provider = provider
        self.speaker = speaker
        self.owner = owner or TurnOwner()
        self.effort = effort
        self.on_error = on_error
        self._turn_lock = asyncio.Lock()
        self.turns = 0

    @classmethod
    async def attach(
        cls,
        server,
        authoritative_thread: str,
        *,
        cwd: Path,
        speaker: SpeechBackend,
        model: str | None = None,
        developer_instructions: str | None = None,
        provider_factory=CodexSubscription,
        owner: TurnOwner | None = None,
        effort: str = "low",
        on_error: Callable[[str], None] | None = None,
    ) -> "SimpleVoiceSession":
        """Snapshot stable harness history exactly once, then keep that fork."""
        source = await server.read_thread(authoritative_thread)
        last_completed = next(
            (
                str(turn["id"])
                for turn in reversed(source.get("turns", []))
                if turn.get("status") == "completed" and turn.get("id")
            ),
            None,
        )
        conversation = await server.fork_thread(
            authoritative_thread,
            cwd=cwd,
            model=model,
            developer_instructions=developer_instructions,
            ephemeral=True,
            last_turn_id=last_completed,
        )
        binding = SimpleThread(
            authoritative_thread,
            conversation,
            last_completed,
        )
        return cls(
            binding,
            provider_factory(server, conversation),
            SerialSpeaker(speaker),
            owner=owner,
            effort=effort,
            on_error=on_error,
        )

    def observe(self, text: str, *, now: float) -> SubmissionDecision:
        """Feed one final ASR fragment to the session's sole TurnOwner."""
        return self.owner.observe(text, now=now)

    async def submit_due(self, *, now: float) -> str | None:
        decision = self.owner.due(now=now)
        if decision.action != "submit":
            return None
        return await self.respond(decision.text)

    async def respond(self, text: str) -> str:
        """Run and speak one whole turn; concurrent calls remain ordered."""
        if not text.strip():
            return self._fail("refusing to submit empty user text")
        async with self._turn_lock:
            chunks: list[str] = []
            try:
                async for event in self.provider.stream(text, effort=self.effort):
                    if event.kind == "delta":
                        chunks.append(event.text)
                response = "".join(chunks).strip()
                if not response:
                    raise EmptyCompletionError(
                        "conversation completed without assistant text"
                    )
                await self.speaker.say(response)
            except Exception as error:
                return self._fail(f"simple voice turn failed: {error}", cause=error)
            self.turns += 1
            return response

    def _fail(self, message: str, *, cause: Exception | None = None) -> str:
        if self.on_error:
            self.on_error(message)
        if cause is None:
            raise SimpleVoiceError(message)
        raise SimpleVoiceError(message) from cause

    async def close(self) -> None:
        await self.speaker.close()
