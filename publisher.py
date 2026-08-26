"""Shadow publisher from one TurnOwner commit to canonical PM intake."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from collections.abc import Callable
from typing import Awaitable, Protocol

try:
    from contracts.events import CommittedVoiceTurn, text_digest
except ModuleNotFoundError:  # standalone runtime bundle
    from events import CommittedVoiceTurn, text_digest

LifecycleSink = Callable[[dict[str, object]], None]
Publisher = Callable[[CommittedVoiceTurn], Awaitable[bool]]


class OwnerDecision(Protocol):
    action: str
    text: str


def build_committed_turn(
    decision: OwnerDecision,
    *,
    thread: str,
    conversation: str,
    run_id: str,
    commit_sequence: int,
    ts_ns: int,
    issue: int | str | None = None,
    boundary: dict[str, object] | None = None,
) -> CommittedVoiceTurn:
    """Build a stable identity from the owner decision and optional PM metadata."""
    if decision.action != "submit" or not decision.text:
        raise ValueError("only a committed TurnOwner submission may be published")
    if not run_id or commit_sequence < 1:
        raise ValueError("run_id and positive commit_sequence are required")
    return CommittedVoiceTurn(
        thread=thread,
        conversation=conversation,
        source_id=f"{run_id}-{commit_sequence:x}",
        text=decision.text,
        digest=text_digest(decision.text),
        ts_ns=ts_ns,
        issue=issue,
        boundary=boundary,
    )


def send_committed_turn(
    url: str,
    turn: CommittedVoiceTurn,
    *,
    timeout: float = 0.5,
) -> None:
    """Send the typed body once; caller owns retry policy and identity."""
    request = urllib.request.Request(
        url,
        data=json.dumps(
            turn.request(),
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status >= 300:
            raise RuntimeError(f"PM intake returned HTTP {response.status}")


def _lifecycle(
    sink: LifecycleSink | None,
    kind: str,
    turn: CommittedVoiceTurn,
    attempt: int,
    error: BaseException | None = None,
) -> None:
    if sink is None:
        return
    row: dict[str, object] = {
        "schema": 1,
        "kind": kind,
        "source_id": turn.source_id,
        "attempt": attempt,
    }
    if error is not None:
        row["error"] = type(error).__name__
    sink(row)


async def publish_best_effort(
    url: str,
    turn: CommittedVoiceTurn,
    lifecycle: LifecycleSink | None = None,
    *,
    attempts: int = 2,
    retry_delay: float = 0.05,
    timeout: float = 0.5,
) -> bool:
    """Publish off-loop; retries reuse the exact immutable turn identity."""
    total = max(1, attempts)
    for attempt in range(1, total + 1):
        try:
            await asyncio.to_thread(
                send_committed_turn,
                url,
                turn,
                timeout=timeout,
            )
        except (OSError, RuntimeError) as error:
            _lifecycle(lifecycle, "pm.intake.error", turn, attempt, error)
            close = getattr(error, "close", None)
            if callable(close):
                close()
            if attempt < total:
                await asyncio.sleep(max(0.0, retry_delay))
                continue
            return False
        _lifecycle(lifecycle, "pm.intake.accepted", turn, attempt)
        return True
    return False


class PublisherSwitch:
    """Hot-switch one non-blocking publisher without owning conversation state."""

    def __init__(
        self,
        legacy: Publisher,
        candidate: Publisher,
        lifecycle: LifecycleSink | None = None,
        *,
        active: str = "legacy",
        capacity: int = 128,
        rollback_after: int = 3,
    ) -> None:
        if active not in {"legacy", "candidate"}:
            raise ValueError("publisher lane must be legacy or candidate")
        if capacity < 1 or rollback_after < 1:
            raise ValueError("capacity and rollback_after must be positive")
        self.publishers = {"legacy": legacy, "candidate": candidate}
        self.lifecycle = lifecycle
        self.active = active
        self.capacity = capacity
        self.rollback_after = rollback_after
        self.candidate_failures = 0
        self.tasks: set[asyncio.Task[None]] = set()

    def select(self, lane: str) -> str:
        """Atomically change future deliveries; in-flight turns keep their lane."""
        if lane not in self.publishers:
            raise ValueError("publisher lane must be legacy or candidate")
        previous, self.active = self.active, lane
        if lane == "candidate":
            self.candidate_failures = 0
        return previous

    def submit(self, turn: CommittedVoiceTurn) -> str:
        """Schedule a committed turn and return immediately to the caller."""
        lane = self.active
        if len(self.tasks) >= self.capacity:
            self._event("pm.publisher.saturated", turn, lane)
            if lane == "candidate":
                self.active = "legacy"
                self._event("pm.publisher.rollback", turn, lane)
            return "dropped"
        task = asyncio.create_task(
            self._deliver(lane, self.publishers[lane], turn),
            name=f"voice-pm-{lane}-{turn.source_id}",
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return lane

    async def _deliver(
        self,
        lane: str,
        publisher: Publisher,
        turn: CommittedVoiceTurn,
    ) -> None:
        try:
            accepted = await publisher(turn)
        except Exception as error:
            accepted = False
            self._event(
                "pm.publisher.error",
                turn,
                lane,
                error=type(error).__name__,
            )
        if accepted:
            if lane == "candidate":
                self.candidate_failures = 0
            self._event("pm.publisher.accepted", turn, lane)
            return
        self._event("pm.publisher.rejected", turn, lane)
        if lane != "candidate":
            return
        self.candidate_failures += 1
        if self.candidate_failures >= self.rollback_after:
            self.active = "legacy"
            self._event("pm.publisher.rollback", turn, lane)

    def _event(
        self,
        kind: str,
        turn: CommittedVoiceTurn,
        lane: str,
        *,
        error: str | None = None,
    ) -> None:
        if self.lifecycle is None:
            return
        row: dict[str, object] = {
            "schema": 1,
            "kind": kind,
            "source_id": turn.source_id,
            "lane": lane,
        }
        if error:
            row["error"] = error
        self.lifecycle(row)

    async def drain(self) -> None:
        """Wait for the deliveries already accepted by ``submit``."""
        while self.tasks:
            await asyncio.gather(*tuple(self.tasks))
