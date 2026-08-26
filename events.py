"""Minimal provider-neutral event envelope used at adapter boundaries."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from urllib.parse import quote


@dataclass(frozen=True, slots=True)
class Event:
    source: str
    kind: str
    subject: str = ""
    payload: dict[str, object] = field(default_factory=dict)
    causation: str | None = None
    seq: int = 0
    ts_ns: int = field(default_factory=time.time_ns)
    v: int = 1

    def json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)


def text_digest(text: str) -> str:
    """Canonical digest for committed text crossing an adapter boundary."""
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CommittedVoiceTurn:
    """Stable final-turn contract; streaming partials cannot inhabit this type."""

    thread: str
    conversation: str
    source_id: str
    text: str
    digest: str
    issue: int | str | None = None
    boundary: dict[str, object] | None = None
    source: str = "codex.voice"
    seq: int = 0
    ts_ns: int = 0
    v: int = 1

    def request(self) -> dict[str, object]:
        """Return the typed body accepted by ``/v1/voice/committed``."""
        self.event()  # Reuse the strict digest/type validation.
        body: dict[str, object] = {
            "v": self.v,
            "committed": True,
            "thread": self.thread,
            "conversation": self.conversation,
            "source_id": self.source_id,
            "source": self.source,
            "text": self.text,
            "digest": self.digest,
            "seq": self.seq,
            "ts_ns": self.ts_ns,
        }
        if self.issue is not None:
            body["issue"] = self.issue
        if self.boundary is not None:
            body["boundary"] = dict(self.boundary)
        return body

    def event(self) -> Event:
        fields = {
            "thread": self.thread,
            "conversation": self.conversation,
            "source_id": self.source_id,
            "source": self.source,
            "text": self.text,
            "digest": self.digest,
        }
        for name, value in fields.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.v != 1:
            raise ValueError("unsupported committed voice turn version")
        if (
            not isinstance(self.seq, int)
            or isinstance(self.seq, bool)
            or not isinstance(self.ts_ns, int)
            or isinstance(self.ts_ns, bool)
            or self.seq < 0
            or self.ts_ns < 0
        ):
            raise ValueError("seq and ts_ns must be non-negative")
        if self.digest != text_digest(self.text):
            raise ValueError("committed voice text digest mismatch")
        if self.boundary is not None and not isinstance(self.boundary, dict):
            raise ValueError("boundary must be an object")

        pm: dict[str, object] = {}
        if self.boundary is not None:
            boundary = dict(self.boundary)
            if self.issue is not None:
                boundary.setdefault("parent", self.issue)
            pm["boundary"] = boundary
        elif self.issue is not None:
            pm["issue"] = self.issue
        subject = "voice:" + ":".join(
            quote(value, safe="")
            for value in (self.source, self.thread, self.source_id)
        )
        return Event(
            source=self.source,
            kind="voice.transcript.final",
            subject=subject,
            payload={
                "committed": True,
                "thread": self.thread,
                "conversation": self.conversation,
                "source_id": self.source_id,
                "text": self.text,
                "digest": self.digest,
                "pm": pm,
            },
            seq=self.seq,
            ts_ns=self.ts_ns,
            v=self.v,
        )


def committed_voice_turn_from(data: object) -> CommittedVoiceTurn:
    """Validate the explicit HTTP/adapter payload and reject partial delivery."""
    if not isinstance(data, dict):
        raise ValueError("committed voice turn must be an object")
    if data.get("committed") is not True:
        raise ValueError("committed must be true; partials use the debug stream")
    allowed = {
        "v",
        "committed",
        "thread",
        "conversation",
        "source_id",
        "text",
        "digest",
        "issue",
        "boundary",
        "source",
        "seq",
        "ts_ns",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown committed voice field: {min(unknown)}")
    required = ("thread", "conversation", "source_id", "text", "digest")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"missing committed voice field: {missing[0]}")
    turn = CommittedVoiceTurn(
        thread=data["thread"],
        conversation=data["conversation"],
        source_id=data["source_id"],
        text=data["text"],
        digest=data["digest"],
        issue=data.get("issue"),
        boundary=data.get("boundary"),
        source=data.get("source", "codex.voice"),
        seq=data.get("seq", 0),
        ts_ns=data.get("ts_ns", 0),
        v=data.get("v", 1),
    )
    turn.event()
    return turn


def _canvas_artifact(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("canvas artifact must be an object")
    if value.get("v") != 1:
        raise ValueError("unsupported canvas artifact version")
    for field in ("id", "title"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"canvas artifact {field} must be a non-empty string")
    nodes, edges = value.get("nodes"), value.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("canvas artifact nodes and edges must be arrays")
    for index, node in enumerate(nodes):
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("id"), str)
            or not isinstance(node.get("position"), dict)
            or not isinstance(node.get("data"), dict)
        ):
            raise ValueError(f"invalid canvas artifact node at index {index}")
    for index, edge in enumerate(edges):
        if (
            not isinstance(edge, dict)
            or not isinstance(edge.get("id"), str)
            or not isinstance(edge.get("source"), str)
            or not isinstance(edge.get("target"), str)
        ):
            raise ValueError(f"invalid canvas artifact edge at index {index}")
    allowed = {"v", "id", "title", "nodes", "edges"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown canvas artifact field: {min(unknown)}")
    return {
        "v": 1,
        "id": value["id"],
        "title": value["title"],
        "nodes": [dict(node) for node in nodes],
        "edges": [dict(edge) for edge in edges],
    }


@dataclass(frozen=True, slots=True)
class CanvasMutationProposal:
    """One validated, non-executing canvas mutation proposed by an LLM."""

    proposal_id: str
    artifact: dict[str, object]
    source: str
    causation: str
    seq: int = 0
    ts_ns: int = field(default_factory=time.time_ns)
    v: int = 1

    def event(self) -> Event:
        for name, value in {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "causation": self.causation,
        }.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.v != 1:
            raise ValueError("unsupported canvas proposal version")
        if (
            not isinstance(self.seq, int)
            or isinstance(self.seq, bool)
            or not isinstance(self.ts_ns, int)
            or isinstance(self.ts_ns, bool)
            or self.seq < 0
            or self.ts_ns < 0
        ):
            raise ValueError("seq and ts_ns must be non-negative")
        artifact = _canvas_artifact(self.artifact)
        return Event(
            source=self.source,
            kind="canvas.mutation.proposed",
            subject=f"canvas:proposal:{quote(self.proposal_id, safe='')}",
            causation=self.causation,
            payload={
                "v": 1,
                "proposal_id": self.proposal_id,
                "operation": "artifact.upsert",
                "artifact": artifact,
            },
            seq=self.seq,
            ts_ns=self.ts_ns,
            v=self.v,
        )


def canvas_mutation_proposal_from(data: object) -> CanvasMutationProposal:
    """Validate a provider-neutral proposed canvas mutation."""
    if not isinstance(data, dict):
        raise ValueError("canvas proposal must be an object")
    allowed = {
        "v",
        "proposal_id",
        "artifact",
        "source",
        "causation",
        "seq",
        "ts_ns",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown canvas proposal field: {min(unknown)}")
    required = ("proposal_id", "artifact", "source", "causation")
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"missing canvas proposal field: {missing[0]}")
    proposal = CanvasMutationProposal(
        proposal_id=data["proposal_id"],
        artifact=data["artifact"],
        source=data["source"],
        causation=data["causation"],
        seq=data.get("seq", 0),
        ts_ns=data.get("ts_ns", time.time_ns()),
        v=data.get("v", 1),
    )
    proposal.event()
    return proposal
