"""Deterministic input and spoken-notification policy for voice runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class MicMode(StrEnum):
    PUSH_TO_TALK = "push-to-talk"
    CONTINUOUS = "continuous"
    MUTED = "muted"


class NotificationMode(StrEnum):
    CONVERSATIONAL = "conversational"
    UPDATES = "updates"
    CRITICAL = "critical"


class Severity(IntEnum):
    DETAIL = 0
    UPDATE = 1
    CRITICAL = 2


@dataclass(frozen=True, slots=True)
class VoiceModes:
    mic: MicMode = MicMode.CONTINUOUS
    notifications: NotificationMode = NotificationMode.CONVERSATIONAL

    @property
    def microphone_open(self) -> bool:
        return self.mic is MicMode.CONTINUOUS

    def should_capture(self, *, push_held: bool = False) -> bool:
        if self.mic is MicMode.MUTED:
            return False
        if self.mic is MicMode.PUSH_TO_TALK:
            return push_held
        return True

    def should_speak(
        self,
        severity: Severity,
        *,
        direct_reply: bool = False,
    ) -> bool:
        # Notification density never suppresses an answer explicitly requested
        # by the user. It governs proactive/background speech only.
        if direct_reply:
            return True
        threshold = {
            NotificationMode.CONVERSATIONAL: Severity.DETAIL,
            NotificationMode.UPDATES: Severity.UPDATE,
            NotificationMode.CRITICAL: Severity.CRITICAL,
        }[self.notifications]
        return severity >= threshold
