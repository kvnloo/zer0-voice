"""Deterministic conversational floor control for a full-duplex voice agent."""

from __future__ import annotations

import enum
import re
from collections import Counter
from dataclasses import dataclass


class FloorDecision(str, enum.Enum):
    YIELD = "yield"
    DUCK = "duck"
    CONTINUE = "continue"
    ACKNOWLEDGE_AND_HOLD = "acknowledge_and_hold"


class AgentPriority(str, enum.Enum):
    NORMAL = "normal"
    IMPORTANT = "important"
    CRITICAL = "critical"


BACKCHANNELS = {
    "yeah",
    "yep",
    "yes",
    "right",
    "okay",
    "ok",
    "uh huh",
    "mhm",
    "got it",
    "sure",
    "exactly",
    "wow",
    "nice",
}
YIELD_PHRASES = {
    "stop",
    "wait",
    "hold on",
    "pause",
    "no",
    "actually",
    "that's wrong",
    "let me",
    "hang on",
    "one second",
}


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()


@dataclass(frozen=True)
class Interruption:
    text: str = ""
    duration_ms: int = 0
    confidence: float = 0.0
    speech_probability: float = 0.0

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(normalized(self.text).split())

    @property
    def is_backchannel(self) -> bool:
        phrase = normalized(self.text)
        return bool(phrase) and (
            phrase in BACKCHANNELS
            or (len(self.words) <= 2 and phrase.rstrip("s") in BACKCHANNELS)
        )

    @property
    def explicitly_requests_floor(self) -> bool:
        phrase = normalized(self.text)
        return (
            any(phrase == item or phrase.startswith(f"{item} ") for item in YIELD_PHRASES)
            or "?" in self.text
        )

    @property
    def is_substantial(self) -> bool:
        return len(self.words) >= 4 or self.duration_ms >= 900


@dataclass(frozen=True)
class FloorPolicy:
    """Policy tuned for natural overlap rather than universal interruption."""

    onset_duck_ms: int = 180
    semantic_commit_ms: int = 420
    min_speech_probability: float = 0.55
    min_transcript_confidence: float = 0.45

    def decide(
        self,
        interruption: Interruption,
        *,
        agent_priority: AgentPriority = AgentPriority.NORMAL,
    ) -> FloorDecision:
        if interruption.speech_probability < self.min_speech_probability:
            return FloorDecision.CONTINUE

        # Duck quickly while gathering enough audio to know whether this is an
        # interruption, a backchannel, or speaker echo.
        if interruption.duration_ms < self.semantic_commit_ms:
            return (
                FloorDecision.DUCK
                if interruption.duration_ms >= self.onset_duck_ms
                else FloorDecision.CONTINUE
            )

        if (
            interruption.confidence >= self.min_transcript_confidence
            and interruption.is_backchannel
        ):
            return FloorDecision.CONTINUE

        wants_floor = (
            interruption.explicitly_requests_floor or interruption.is_substantial
        )
        if not wants_floor:
            return FloorDecision.DUCK
        if agent_priority is AgentPriority.CRITICAL:
            return FloorDecision.ACKNOWLEDGE_AND_HOLD
        if agent_priority is AgentPriority.IMPORTANT and not interruption.explicitly_requests_floor:
            return FloorDecision.ACKNOWLEDGE_AND_HOLD
        return FloorDecision.YIELD


@dataclass
class AdaptiveEndpoint:
    """Turn endpointing that tolerates reflective pauses and adapts to cadence."""

    baseline_ms: int = 900
    minimum_ms: int = 700
    maximum_ms: int = 2_200
    cadence_ms: float = 760.0
    alpha: float = 0.2
    complete_factor: float = 0.60
    incomplete_factor: float = 1.65
    thinking_factor: float = 1.35

    def observe_within_turn_pause(self, pause_ms: int) -> None:
        if self.minimum_ms <= pause_ms <= self.maximum_ms:
            self.cadence_ms = self.alpha * pause_ms + (1 - self.alpha) * self.cadence_ms

    def silence_needed_ms(self, *, syntactically_complete: bool, thinking_words: bool) -> int:
        threshold = self.cadence_ms * (
            self.complete_factor if syntactically_complete else self.incomplete_factor
        )
        if thinking_words:
            threshold *= self.thinking_factor
        return round(max(self.minimum_ms, min(self.maximum_ms, threshold)))


@dataclass(frozen=True)
class EndpointHint:
    complete: bool
    thinking: bool = False
    force: bool = False
    defer: bool = False


INCOMPLETE_TAILS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "are",
    "because",
    "can",
    "could",
    "but",
    "by",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "if",
    "i",
    "in",
    "into",
    "is",
    "it",
    "like",
    "need",
    "of",
    "or",
    "should",
    "so",
    "than",
    "that",
    "the",
    "they",
    "then",
    "to",
    "unless",
    "until",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
    "we",
    "you",
}
THINKING_TAILS = {
    "er",
    "hmm",
    "like",
    "so",
    "uh",
    "um",
    "you know",
}
FORCE_COMMIT = re.compile(r"(?:^|\s)(?:send it|send that|over)\s*[.!?]*$", re.I)
SHORT_COMPLETE = {
    "go ahead",
    "keep going",
    "make it",
    "send it",
    "stop",
    "thank you",
    "thanks",
    "yes",
    "no",
}
NON_LEXICAL = {"ah", "cough", "er", "hmm", "uh", "um"}
INCOMPLETE_PHRASE_TAILS = {
    "going to",
    "have to",
    "how we",
    "how you",
    "if we",
    "if you",
    "need to",
    "so that",
    "that we",
    "that you",
    "want to",
    "what we",
    "what you",
    "when we",
    "when you",
}


@dataclass(frozen=True)
class TranscriptQuality:
    accepted: bool
    reason: str = "accepted"


def transcript_quality(
    text: str,
    *,
    hotwords: tuple[str, ...] = (),
) -> TranscriptQuality:
    """Fail closed on deterministic ASR junk while preserving real short turns."""
    words = normalized(text).split()
    if not words:
        return TranscriptQuality(False, "empty")
    if all(word in NON_LEXICAL for word in words):
        return TranscriptQuality(False, "non-lexical")

    counts = Counter(words)
    most_common = counts.most_common(1)[0][1]
    longest_run = 1
    run = 1
    for previous, current in zip(words, words[1:]):
        run = run + 1 if current == previous else 1
        longest_run = max(longest_run, run)

    # Whisper hallucinations commonly repeat a hotword ("JAX, JAX, JAX") or a
    # tiny vocabulary for an entire no-speech segment. Those partials remain
    # useful on the debug display, but must never become harness commands.
    if longest_run >= 3:
        return TranscriptQuality(False, "repeated-token-run")
    if len(words) >= 5 and most_common >= 4:
        return TranscriptQuality(False, "dominant-repeated-token")
    if len(words) >= 6 and len(counts) / len(words) <= 0.40:
        return TranscriptQuality(False, "low-lexical-diversity")
    if (
        len(words) >= 2
        and len(counts) == 1
        and words[0] in {normalized(term) for term in hotwords}
    ):
        return TranscriptQuality(False, "repeated-hotword")
    return TranscriptQuality(True)


def endpoint_hint(text: str) -> EndpointHint:
    """Classify an unstable partial without ever publishing it as a turn."""
    value = text.strip()
    normalized_text = normalized(value)
    if not normalized_text:
        return EndpointHint(False)
    if FORCE_COMMIT.search(value):
        return EndpointHint(True, force=True)
    words = normalized_text.split()
    tail = words[-1]
    two_word_tail = " ".join(words[-2:])
    thinking = tail in THINKING_TAILS or two_word_tail in THINKING_TAILS
    question = bool(re.search(r"\?[\"')\]]*\s*$", value))
    ellipsis = bool(re.search(r"(?:\.{2,}|…+|[-–—])\s*$", value))
    explicitly_closed = bool(re.search(r"[.!?][\"')\]]*\s*$", value)) and not ellipsis
    starts_as_continuation = words[0] in {
        "and",
        "as",
        "because",
        "but",
        "for",
        "or",
        "so",
    }
    structurally_incomplete = (
        normalized_text not in SHORT_COMPLETE
        and (
            tail in INCOMPLETE_TAILS
            or two_word_tail in INCOMPLETE_PHRASE_TAILS
            or starts_as_continuation
            or ellipsis
        )
    )
    defer = structurally_incomplete and not question
    incomplete = (
        defer
        or (
            len(words) < 4
            and normalized_text not in SHORT_COMPLETE
            and not explicitly_closed
        )
    )
    # Whisper punctuation is useful when present, but absence of punctuation is
    # not enough to split a fluent speaker. A content-word tail is tentatively
    # complete; resuming speech before the deadline cancels the commit.
    complete = question or (
        not incomplete
        and not thinking
        and (explicitly_closed or len(words) >= 2)
    )
    return EndpointHint(complete, thinking, defer=defer)


@dataclass(frozen=True)
class SubmissionDecision:
    action: str
    text: str = ""
    reason: str = ""


@dataclass
class TurnOwner:
    """Own one pending thought across ASR endpoint fragments.

    Segment transcripts may update the UI immediately, but only this state
    machine publishes a harness turn. The caller supplies monotonic timestamps
    so its pacing is deterministic and directly benchmarkable.
    """

    hotwords: tuple[str, ...] = ()
    settle_seconds: float = 0.42
    incomplete_seconds: float = 1.60
    question_seconds: float = 0.22
    pending: str = ""
    deadline: float | None = None

    def observe(self, text: str, *, now: float) -> SubmissionDecision:
        quality = transcript_quality(text, hotwords=self.hotwords)
        if not quality.accepted:
            return SubmissionDecision("reject", self.pending, quality.reason)

        value = text.strip()
        self.pending = " ".join(part for part in (self.pending, value) if part)
        hint = endpoint_hint(value)
        if hint.force:
            return self.commit(reason="forced")

        wait = self.incomplete_seconds if hint.defer else self.settle_seconds
        if value.rstrip().endswith("?"):
            wait = min(wait, self.question_seconds)
        self.deadline = now + max(0.0, wait)
        return SubmissionDecision(
            "hold",
            self.pending,
            "incomplete" if hint.defer else "settling",
        )

    def remaining(self, *, now: float) -> float | None:
        if not self.pending or self.deadline is None:
            return None
        return max(0.0, self.deadline - now)

    def due(self, *, now: float) -> SubmissionDecision:
        if not self.pending:
            return SubmissionDecision("idle")
        if self.deadline is None or now < self.deadline:
            return SubmissionDecision("hold", self.pending, "settling")
        return self.commit(reason="deadline")

    def commit(self, *, reason: str = "commit") -> SubmissionDecision:
        text = self.pending
        self.pending = ""
        self.deadline = None
        return SubmissionDecision("submit", text, reason)

    def cancel(self) -> SubmissionDecision:
        """Discard an uncommitted thought when its owning coroutine is cancelled."""
        text = self.pending
        self.pending = ""
        self.deadline = None
        return SubmissionDecision("cancel", text, "cancelled")
