"""Persistent xAI Speech-to-Speech transport; Hermes retains all authority."""

from __future__ import annotations

import base64
import json
import os
from collections import deque
from typing import Any

from agent.realtime_voice import (
    HeardAudioBoundary,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSession,
    RealtimeVoiceProvider,
)

_ENDPOINT = "wss://api.x.ai/v1/realtime?model=grok-voice-latest"


class XAIRealtimeSession(RealtimeSession):
    def __init__(self, socket: Any, reconnect: Any) -> None:
        self._socket = socket
        self._reconnect = reconnect
        self._closed = False
        self._reconnected = False
        self._conversation_id: str | None = None
        self._buffer: deque[dict[str, Any]] = deque()
        self._tool_group: set[str] = set()
        self._settled_calls: set[str] = set()
        self._audio_input_bytes = 0
        self._audio_output_bytes = 0

    async def send_audio(self, pcm: bytes) -> None:
        self._audio_input_bytes += len(pcm)
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        })

    async def events(self):
        while not self._closed:
            try:
                event = self._buffer.popleft() if self._buffer else json.loads(await self._socket.recv())
            except Exception:
                if self._reconnected or not self._conversation_id:
                    yield RealtimeEvent(type=RealtimeEventType.ERROR, text="xAI realtime connection closed")
                    return
                self._reconnected = True
                self._socket = await self._reconnect(self._conversation_id)
                continue
            kind = event.get("type")
            if kind == "conversation.created":
                self._conversation_id = event.get("conversation", {}).get("id")
            elif kind == "input_audio_buffer.speech_started":
                yield RealtimeEvent(type=RealtimeEventType.TURN_STARTED)
            elif kind == "conversation.item.input_audio_transcription.updated":
                yield RealtimeEvent.transcript(event.get("transcript", ""))
            elif kind in {"response.output_audio.delta", "response.audio.delta"}:
                audio = base64.b64decode(event["delta"], validate=True)
                self._audio_output_bytes += len(audio)
                yield RealtimeEvent.audio(audio, item_id=event.get("item_id"))
            elif kind == "response.function_call_arguments.done":
                group = [event]
                while True:
                    following = json.loads(await self._socket.recv())
                    if following.get("type") != kind:
                        self._buffer.append(following)
                        break
                    group.append(following)
                for call in group:
                    call_id = call.get("call_id")
                    if not call_id or call_id in self._settled_calls or call_id in self._tool_group:
                        continue
                    try:
                        arguments = json.loads(call.get("arguments") or "{}")
                    except (TypeError, json.JSONDecodeError):
                        arguments = {}
                    self._tool_group.add(call_id)
                    yield RealtimeEvent.tool_call(call_id, call.get("name", ""), arguments)
            elif kind == "response.done":
                yield RealtimeEvent(type=RealtimeEventType.TURN_ENDED)
            elif kind == "error":
                yield RealtimeEvent(type=RealtimeEventType.ERROR, text="xAI realtime error")

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        if call_id not in self._tool_group or call_id in self._settled_calls:
            return
        self._settled_calls.add(call_id)
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        })
        if self._tool_group <= self._settled_calls:
            await self._send({"type": "response.create"})
            self._tool_group.clear()

    async def truncate_response(self, boundary: HeardAudioBoundary) -> None:
        """Keep provider history aligned with audio the host actually rendered."""
        await self._send({
            "type": "conversation.item.truncate",
            "item_id": boundary.item_id,
            "content_index": 0,
            "audio_end_ms": boundary.audio_end_ms,
        })

    async def cancel_response(self) -> None:
        await self._send({"type": "response.cancel"})

    def metrics(self) -> dict[str, int]:
        """Return bounded counters only: never credentials, transcripts, or provider errors."""
        return {
            "audio_input_bytes": self._audio_input_bytes,
            "audio_output_bytes": self._audio_output_bytes,
            "reconnects": int(self._reconnected),
            "tool_calls": len(self._settled_calls),
        }

    async def _send(self, event: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("realtime session is closed")
        await self._socket.send(json.dumps(event))

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._socket.close()


class XAIRealtimeVoiceProvider(RealtimeVoiceProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        connector: Any = None,
        speed: float = 1.0,
        reasoning: str = "high",
        timeout: float = 10.0,
    ) -> None:
        if not 0.7 <= speed <= 1.5:
            raise ValueError("speed must be between 0.7 and 1.5")
        if reasoning not in {"high", "none"}:
            raise ValueError("reasoning must be 'high' or 'none'")
        self._api_key = api_key or os.environ.get("XAI_API_KEY")
        self._connector = connector
        self._speed = speed
        self._reasoning = reasoning
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "xai-grok-voice"

    @property
    def display_name(self) -> str:
        return "xAI Grok Voice"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "xAI",
            "tag": "realtime",
            "env_vars": ["XAI_API_KEY"],
        }

    async def open_session(self, *, instructions, tools, voice=None):
        if not self._api_key:
            raise RuntimeError("XAI_API_KEY is required")
        connector = self._connector
        if connector is None:
            import websockets
            connector = websockets.connect
        function_tools = [
            {"type": "function", **tool}
            for tool in tools
            if tool.get("type", "function") == "function"
        ]
        config = {
            "type": "session.update",
            "session": {
                "instructions": instructions,
                "voice": voice,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "grok-transcribe"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "speed": self._speed,
                    },
                },
                "turn_detection": {"type": "server_vad"},
                "reasoning": {"effort": self._reasoning},
                "tools": function_tools,
                "resumption": {"enabled": True},
            },
        }

        async def connect(conversation_id: str | None = None):
            url = _ENDPOINT
            if conversation_id:
                url += f"&conversation_id={conversation_id}"
            try:
                socket = await connector(
                    url,
                    additional_headers={"Authorization": f"Bearer {self._api_key}"},
                    open_timeout=self._timeout,
                )
            except Exception:
                raise RuntimeError("xAI realtime connection failed") from None
            await socket.send(json.dumps(config))
            return socket

        return XAIRealtimeSession(await connect(), connect)
