"""Shared causal blackboard for mutual steering between intelligence lanes."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from fleet import Horizon, Severity


@dataclass(frozen=True)
class Proposal:
    lane: str
    topic: str
    text: str
    confidence: float
    horizon: Horizon
    severity: Severity = Severity.CONTEXT
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    supersedes: str | None = None
    caused_by: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")


class PeerLane(Protocol):
    name: str

    async def react(
        self,
        user_text: str,
        context: tuple[str, ...],
        board: tuple[Proposal, ...],
    ) -> Proposal | None: ...


class TurnBoard:
    def __init__(self, on_steer=None):
        self._proposals: dict[str, Proposal] = {}
        self._superseded: set[str] = set()
        self.on_steer = on_steer

    @property
    def version(self) -> int:
        return len(self._proposals)

    def publish(self, proposal: Proposal) -> bool:
        if proposal.id in self._proposals:
            return False
        if proposal.supersedes and proposal.supersedes not in self._proposals:
            raise ValueError("a proposal can only supersede a known proposal")
        # Do not let the same causal result circulate forever under fresh IDs.
        causal_signature = (proposal.lane, proposal.topic, proposal.text, proposal.caused_by)
        if any(
            (item.lane, item.topic, item.text, item.caused_by) == causal_signature
            for item in self._proposals.values()
        ):
            return False
        self._proposals[proposal.id] = proposal
        if proposal.supersedes:
            self._superseded.add(proposal.supersedes)
            superseded = self._proposals[proposal.supersedes]
            if self.on_steer is not None and superseded.lane != proposal.lane:
                # One lane explicitly steering another is dashboard-visible
                # hierarchy, not an implementation detail.
                self.on_steer(proposal.lane, superseded.lane, proposal.topic)
        return True

    def active(self, topic: str | None = None) -> tuple[Proposal, ...]:
        proposals = (
            item
            for item in self._proposals.values()
            if item.id not in self._superseded
            and (topic is None or item.topic == topic)
        )
        return tuple(
            sorted(
                proposals,
                key=lambda item: (
                    int(item.severity),
                    item.confidence,
                    item.created_at,
                ),
                reverse=True,
            )
        )

    def best(self, topic: str) -> Proposal | None:
        active = self.active(topic)
        return active[0] if active else None


class DeliberationMesh:
    """Let every lane steer every other lane with a strict convergence bound."""

    def __init__(self, lanes: list[PeerLane], *, max_rounds: int = 3, on_steer=None):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.lanes = tuple(lanes)
        self.max_rounds = max_rounds
        self.on_steer = on_steer

    async def run(
        self, user_text: str, context: tuple[str, ...] = ()
    ) -> TurnBoard:
        board = TurnBoard(on_steer=self.on_steer)
        seen_versions = {lane.name: -1 for lane in self.lanes}
        for _round in range(self.max_rounds):
            snapshot = board.active()
            version = board.version

            async def react(lane: PeerLane):
                if seen_versions[lane.name] == version:
                    return None
                seen_versions[lane.name] = version
                return await lane.react(user_text, context, snapshot)

            results = await asyncio.gather(*(react(lane) for lane in self.lanes))
            changed = False
            for result in results:
                if result is not None:
                    changed = board.publish(result) or changed
            if not changed:
                break
        return board
