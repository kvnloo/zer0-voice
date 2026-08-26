"""Independent, hot-switchable boundary from committed voice turns to PM."""

from __future__ import annotations

import time
from collections.abc import Callable

try:
    from .publisher import (
        LifecycleSink,
        OwnerDecision,
        Publisher,
        PublisherSwitch,
        build_committed_turn,
        publish_best_effort,
    )
except ImportError:  # standalone runtime bundle
    from publisher import (
        LifecycleSink,
        OwnerDecision,
        Publisher,
        PublisherSwitch,
        build_committed_turn,
        publish_best_effort,
    )


async def _legacy_noop(_turn) -> bool:
    """The pre-promotion lane intentionally performs no PM delivery."""
    return True


def relay_publisher(
    endpoint: str,
    lifecycle: LifecycleSink | None = None,
    *,
    attempts: int = 2,
    retry_delay: float = 0.05,
    timeout: float = 0.5,
) -> Publisher:
    """Build the candidate publisher without putting I/O on the caller's path."""

    async def publish(turn) -> bool:
        return await publish_best_effort(
            endpoint,
            turn,
            lifecycle,
            attempts=attempts,
            retry_delay=retry_delay,
            timeout=timeout,
        )

    return publish


class VoicePMWiring:
    """Schedule only committed owner decisions through one blue/green switch.

    This object deliberately owns no microphone, TurnOwner, model, or TTS state.
    ``schedule`` is synchronous: relay work begins in a detached asyncio task.
    """

    def __init__(
        self,
        *,
        thread: str,
        conversation: str,
        run_id: str,
        candidate: Publisher,
        legacy: Publisher = _legacy_noop,
        lifecycle: LifecycleSink | None = None,
        active: str = "legacy",
        capacity: int = 128,
        rollback_after: int = 3,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not thread or not conversation or not run_id:
            raise ValueError("thread, conversation, and run_id are required")
        self.thread = thread
        self.conversation = conversation
        self.run_id = run_id
        self.clock_ns = clock_ns
        self.commit_sequence = 0
        self.switch = PublisherSwitch(
            legacy,
            candidate,
            lifecycle,
            active=active,
            capacity=capacity,
            rollback_after=rollback_after,
        )

    @classmethod
    def for_relay(
        cls,
        endpoint: str,
        *,
        thread: str,
        conversation: str,
        run_id: str,
        lifecycle: LifecycleSink | None = None,
        attempts: int = 2,
        retry_delay: float = 0.05,
        timeout: float = 0.5,
        **kwargs,
    ) -> "VoicePMWiring":
        """Create a legacy-default boundary with one best-effort relay candidate."""
        return cls(
            thread=thread,
            conversation=conversation,
            run_id=run_id,
            candidate=relay_publisher(
                endpoint,
                lifecycle,
                attempts=attempts,
                retry_delay=retry_delay,
                timeout=timeout,
            ),
            lifecycle=lifecycle,
            **kwargs,
        )

    @property
    def active(self) -> str:
        return self.switch.active

    def select(self, lane: str) -> str:
        """Hot-select future delivery without replacing conversation state."""
        return self.switch.select(lane)

    def schedule(self, decision: OwnerDecision) -> str:
        """Validate, identify, and enqueue one committed TurnOwner decision."""
        return self.schedule_with_boundary(decision, boundary=None)

    def schedule_with_boundary(
        self,
        decision: OwnerDecision,
        boundary: dict[str, object] | None = None,
    ) -> str:
        """Validate, identify, and enqueue one committed decision."""
        if decision.action != "submit" or not decision.text:
            raise ValueError("only a committed TurnOwner submission may be scheduled")
        sequence = self.commit_sequence + 1
        turn = build_committed_turn(
            decision,
            thread=self.thread,
            conversation=self.conversation,
            run_id=self.run_id,
            commit_sequence=sequence,
            ts_ns=self.clock_ns(),
            boundary=boundary,
        )
        lane = self.switch.submit(turn)
        self.commit_sequence = sequence
        return lane

    async def drain(self) -> None:
        """Wait only during controlled shutdown or tests, never in response flow."""
        await self.switch.drain()
