import asyncio
import json
import sys
import types
import unittest
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    TOOL_CALL = "tool_call"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    type: EventType
    audio_bytes: bytes | None = None
    text: str | None = None
    final: bool = False
    call_id: str | None = None
    tool_name: str | None = None
    item_id: str | None = None
    arguments: dict = field(default_factory=dict)

    @classmethod
    def audio(cls, value, *, item_id=None):
        return cls(type=EventType.AUDIO, audio_bytes=value, item_id=item_id)

    @classmethod
    def transcript(cls, value, *, final=False):
        return cls(type=EventType.TRANSCRIPT, text=value, final=final)

    @classmethod
    def tool_call(cls, call_id, name, arguments):
        return cls(type=EventType.TOOL_CALL, call_id=call_id, tool_name=name, arguments=arguments)


@dataclass(frozen=True)
class HeardAudioBoundary:
    item_id: str
    audio_end_ms: int


class Session:
    async def truncate_response(self, boundary):
        pass

class Provider: pass


class ExactContractCoordinator:
    """Focused copy of Hermes PR #95147's public heard-boundary orchestration."""

    def __init__(self, session):
        self._session = session
        self._current_item_id = None
        self._current_audio_events = {}
        self._heard_boundary = None

    async def events(self):
        async for event in self._session.events():
            if event.type is EventType.AUDIO and event.item_id:
                self._current_item_id = event.item_id
                self._current_audio_events[id(event)] = event
            yield event

    def report_audio_heard(self, event, *, audio_end_ms):
        if (
            event.item_id != self._current_item_id
            or self._current_audio_events.get(id(event)) is not event
        ):
            return False
        self._heard_boundary = HeardAudioBoundary(event.item_id, audio_end_ms)
        return True

    async def cancel_response(self):
        if self._heard_boundary is not None:
            await self._session.truncate_response(self._heard_boundary)
        await self._session.cancel_response()


agent = types.ModuleType("agent")
contract = types.ModuleType("agent.realtime_voice")
contract.RealtimeEvent = Event
contract.RealtimeEventType = EventType
contract.HeardAudioBoundary = HeardAudioBoundary
contract.RealtimeSession = Session
contract.RealtimeVoiceProvider = Provider
agent.realtime_voice = contract
sys.modules.setdefault("agent", agent)
sys.modules.setdefault("agent.realtime_voice", contract)

from xai_realtime_voice.provider import XAIRealtimeVoiceProvider
from xai_realtime_voice import register


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.closed = False

    async def send(self, value):
        self.sent.append(json.loads(value))

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)

    async def close(self):
        self.closed = True


class XAIRealtimeVoiceTests(unittest.IsolatedAsyncioTestCase):
    def test_plugin_registers_through_supported_realtime_provider_seam(self):
        class Context:
            provider = None

            def register_realtime_voice_provider(self, provider):
                self.provider = provider

        context = Context()
        register(context)
        self.assertEqual(context.provider.name, "xai-grok-voice")

    async def make_session(self):
        socket = FakeSocket()

        async def connect(url, *, additional_headers, open_timeout):
            return socket

        provider = XAIRealtimeVoiceProvider(api_key="secret", connector=connect)
        return socket, await provider.open_session(instructions="host", tools=[])

    async def test_open_configures_exact_server_side_session_without_builtin_tools(self):
        socket = FakeSocket()
        seen = {}

        async def connect(url, *, additional_headers, open_timeout):
            seen.update(url=url, headers=additional_headers, timeout=open_timeout)
            return socket

        provider = XAIRealtimeVoiceProvider(
            api_key="top-secret", connector=connect, speed=1.2, reasoning="none"
        )
        session = await provider.open_session(
            instructions="Hermes owns tools", tools=[{"name": "lookup"}], voice="eve"
        )

        self.assertEqual(
            seen,
            {
                "url": "wss://api.x.ai/v1/realtime?model=grok-voice-latest",
                "headers": {"Authorization": "Bearer top-secret"},
                "timeout": 10.0,
            },
        )
        self.assertEqual(socket.sent, [{
            "type": "session.update",
            "session": {
                "instructions": "Hermes owns tools",
                "voice": "eve",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "grok-transcribe"},
                    },
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}, "speed": 1.2},
                },
                "turn_detection": {"type": "server_vad"},
                "reasoning": {"effort": "none"},
                "tools": [{"type": "function", "name": "lookup"}],
                "resumption": {"enabled": True},
            },
        }])
        await session.close()

    async def test_events_stream_cumulative_corrected_transcript_audio_and_turn_marks(self):
        socket, session = await self.make_session()
        for event in [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "conversation.item.input_audio_transcription.updated", "transcript": "turn on"},
            {"type": "conversation.item.input_audio_transcription.updated", "transcript": "turn off"},
            {"type": "conversation.item.input_audio_transcription.updated", "transcript": "turn off lights"},
            {"type": "response.output_audio.delta", "delta": "YWJj", "item_id": "answer-1"},
            {"type": "response.done"},
        ]:
            await socket.incoming.put(event)

        stream = session.events()
        received = [await anext(stream) for _ in range(6)]

        self.assertEqual([event.type for event in received], [
            EventType.TURN_STARTED, EventType.TRANSCRIPT, EventType.TRANSCRIPT,
            EventType.TRANSCRIPT, EventType.AUDIO, EventType.TURN_ENDED,
        ])
        self.assertEqual([event.text for event in received[1:4]], ["turn on", "turn off", "turn off lights"])
        self.assertEqual([event.final for event in received[1:4]], [False, False, False])
        self.assertEqual(received[4].audio_bytes, b"abc")
        self.assertEqual(received[4].item_id, "answer-1")

    async def test_exact_coordinator_contract_truncates_heard_audio_before_cancel(self):
        socket, session = await self.make_session()
        coordinator = ExactContractCoordinator(session)
        await session.send_audio(b"pcm")

        await socket.incoming.put(
            {"type": "response.output_audio.delta", "delta": "YWJj", "item_id": "answer-1"}
        )
        output = await anext(coordinator.events())
        self.assertTrue(coordinator.report_audio_heard(output, audio_end_ms=640))
        await coordinator.cancel_response()

        self.assertEqual(socket.sent[1:], [
            {"type": "input_audio_buffer.append", "audio": "cGNt"},
            {"type": "conversation.item.truncate", "item_id": "answer-1", "content_index": 0, "audio_end_ms": 640},
            {"type": "response.cancel"},
        ])

    async def test_corrected_transcript_replaces_prior_cumulative_value(self):
        socket, session = await self.make_session()
        for transcript in ("book a flight to Boston", "book a flight to Austin"):
            await socket.incoming.put({
                "type": "conversation.item.input_audio_transcription.updated",
                "transcript": transcript,
            })

        stream = session.events()
        rendered = ""
        rendered = (await anext(stream)).text
        rendered = (await anext(stream)).text

        self.assertEqual(rendered, "book a flight to Austin")
        self.assertNotIn("Bostonbook", rendered)

    async def test_grouped_tool_results_continue_once_and_duplicate_call_ids_are_ignored(self):
        socket, session = await self.make_session()
        for event in [
            {"type": "response.function_call_arguments.done", "call_id": "a", "name": "safe", "arguments": "{\"x\": 1}"},
            {"type": "response.function_call_arguments.done", "call_id": "b", "name": "approval", "arguments": "{}"},
            {"type": "response.function_call_arguments.done", "call_id": "a", "name": "safe", "arguments": "{\"x\": 1}"},
            {"type": "response.done"},
        ]:
            await socket.incoming.put(event)

        stream = session.events()
        calls = [await anext(stream), await anext(stream)]
        self.assertEqual([(c.call_id, c.tool_name, c.arguments) for c in calls], [
            ("a", "safe", {"x": 1}), ("b", "approval", {}),
        ])

        await session.submit_tool_result("b", "Denied by user")
        await session.submit_tool_result("a", "approved result")
        await session.submit_tool_result("a", "duplicate")
        self.assertEqual(socket.sent[1:], [
            {"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": "b", "output": "Denied by user"}},
            {"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": "a", "output": "approved result"}},
            {"type": "response.create"},
        ])

    async def test_reconnect_resumes_once_and_deduplicates_replayed_call(self):
        first, second = FakeSocket(), FakeSocket()
        sockets = [first, second]
        urls = []

        async def connect(url, *, additional_headers, open_timeout):
            urls.append(url)
            return sockets.pop(0)

        provider = XAIRealtimeVoiceProvider(api_key="secret", connector=connect)
        session = await provider.open_session(instructions="host", tools=[])
        await first.incoming.put({"type": "conversation.created", "conversation": {"id": "resume-7"}})
        await first.incoming.put({"type": "response.function_call_arguments.done", "call_id": "once", "name": "lookup", "arguments": "{}"})
        await first.incoming.put({"type": "response.done"})
        stream = session.events()
        call = await anext(stream)
        await session.submit_tool_result(call.call_id, "ok")
        await anext(stream)
        await first.incoming.put(ConnectionError("wire closed with secret"))
        await second.incoming.put({"type": "conversation.item.created", "item": {"type": "function_call", "call_id": "once"}})
        await second.incoming.put({"type": "response.output_audio.delta", "delta": "eg==", "item_id": "new"})

        replay_safe = await anext(stream)
        self.assertEqual(replay_safe.audio_bytes, b"z")
        self.assertEqual(urls[1], "wss://api.x.ai/v1/realtime?model=grok-voice-latest&conversation_id=resume-7")
        self.assertEqual(second.sent[0]["type"], "session.update")

    async def test_close_is_idempotent_and_metrics_are_secret_and_transcript_free(self):
        socket, session = await self.make_session()
        await session.send_audio(b"private speech bytes")
        await session.close()
        await session.close()
        metrics = json.dumps(session.metrics())
        self.assertEqual(metrics, '{"audio_input_bytes": 20, "audio_output_bytes": 0, "reconnects": 0, "tool_calls": 0}')
        self.assertNotIn("private", metrics)

    async def test_invalid_configuration_and_connection_failure_fail_closed_for_legacy_fallback(self):
        with self.assertRaises(ValueError):
            XAIRealtimeVoiceProvider(api_key="x", speed=1.6)
        with self.assertRaises(ValueError):
            XAIRealtimeVoiceProvider(api_key="x", reasoning="medium")

        async def unavailable(*args, **kwargs):
            raise TimeoutError("Bearer must never be logged")

        provider = XAIRealtimeVoiceProvider(api_key="secret", connector=unavailable)
        with self.assertRaisesRegex(RuntimeError, "xAI realtime connection failed") as caught:
            await provider.open_session(instructions="host", tools=[])
        self.assertNotIn("Bearer", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
