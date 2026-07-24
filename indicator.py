"""Hardware privacy/status indicators for continuous voice."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class VoiceState(StrEnum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


COLORS = {
    VoiceState.LISTENING: (0, 210, 255),
    VoiceState.THINKING: (255, 145, 0),
    VoiceState.SPEAKING: (155, 80, 255),
    VoiceState.ERROR: (255, 0, 30),
}


class Indicator(Protocol):
    def set(self, state: VoiceState) -> None: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...


class NullIndicator:
    def set(self, state: VoiceState) -> None:
        del state

    def close(self) -> None:
        pass

    clear = close


class WootingIndicator:
    """Drive only Esc through Wooting's official transient RGB API."""

    def __init__(
        self,
        library: str | Path | None = None,
        *,
        row: int = 0,
        column: int = 0,
    ):
        resolved = (
            str(library)
            if library
            else os.environ.get("WOOTING_RGB_LIB")
            or ctypes.util.find_library("wooting-rgb-sdk")
        )
        if not resolved:
            raise RuntimeError(
                "Wooting RGB SDK not found; set WOOTING_RGB_LIB or install "
                "libwooting-rgb-sdk.so"
            )
        self.library = ctypes.CDLL(resolved)
        self.row = row
        self.column = column
        self.closed = False
        self.library.wooting_rgb_kbd_connected.argtypes = []
        self.library.wooting_rgb_kbd_connected.restype = ctypes.c_bool
        self.library.wooting_rgb_direct_set_key.argtypes = [ctypes.c_uint8] * 5
        self.library.wooting_rgb_direct_set_key.restype = ctypes.c_bool
        self.library.wooting_rgb_direct_reset_key.argtypes = [ctypes.c_uint8] * 2
        self.library.wooting_rgb_direct_reset_key.restype = ctypes.c_bool
        if not self.library.wooting_rgb_kbd_connected():
            raise RuntimeError(
                "Wooting keyboard is unavailable (check USB/hidraw permissions)"
            )

    def set(self, state: VoiceState) -> None:
        if self.closed:
            return
        red, green, blue = COLORS[state]
        if not self.library.wooting_rgb_direct_set_key(
            self.row, self.column, red, green, blue
        ):
            raise RuntimeError(f"failed to set Wooting indicator to {state.value}")

    def close(self) -> None:
        if self.closed:
            return
        self.clear()
        self.closed = True

    def clear(self) -> None:
        if self.closed:
            return
        # Reset only Esc. Never use full-array or profile-writing APIs.
        self.library.wooting_rgb_direct_reset_key(self.row, self.column)


def make_indicator(
    kind: str,
    *,
    library: str | Path | None = None,
    warning=print,
) -> Indicator:
    if kind == "none":
        return NullIndicator()
    try:
        return WootingIndicator(library)
    except (OSError, RuntimeError) as error:
        if kind == "wooting":
            raise
        warning(f"Wooting voice indicator unavailable: {error}")
        return NullIndicator()
