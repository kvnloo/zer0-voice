"""xAI Grok realtime voice transport for Hermes."""

import os

from .provider import XAIRealtimeVoiceProvider


def register(ctx):
    """Register only the transport; the Hermes coordinator retains tool authority."""
    ctx.register_realtime_voice_provider(XAIRealtimeVoiceProvider(
        speed=float(os.environ.get("HERMES_XAI_VOICE_SPEED", "1.0")),
        reasoning=os.environ.get("HERMES_XAI_VOICE_REASONING", "high"),
    ))


__all__ = ["XAIRealtimeVoiceProvider", "register"]
