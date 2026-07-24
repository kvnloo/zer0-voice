"""Pure completion contract for one direct conversational voice turn."""

from __future__ import annotations


def require_audible_reply(response: str, playback_started: bool) -> None:
    """Accept a turn only when model output exists and playback began."""
    if not response.strip():
        raise RuntimeError("live-response-empty")
    if not playback_started:
        raise RuntimeError("tts-playback-missing")
