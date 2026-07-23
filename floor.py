"""Deterministic conversational floor control for a full-duplex voice agent."""

from __future__ import annotations

import enum
import re
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

    baseline_ms: int = 620
    minimum_ms: int = 380
    maximum_ms: int = 1_400
    cadence_ms: float = 620.0
    alpha: float = 0.2

    def observe_within_turn_pause(self, pause_ms: int) -> None:
        if self.minimum_ms <= pause_ms <= self.maximum_ms:
            self.cadence_ms = self.alpha * pause_ms + (1 - self.alpha) * self.cadence_ms

    def silence_needed_ms(self, *, syntactically_complete: bool, thinking_words: bool) -> int:
        threshold = self.cadence_ms * (0.85 if syntactically_complete else 1.2)
        if thinking_words:
            threshold *= 1.35
        return round(max(self.minimum_ms, min(self.maximum_ms, threshold)))
