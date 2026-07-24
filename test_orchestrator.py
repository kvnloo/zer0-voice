import asyncio
import json
import time
import unittest
from unittest.mock import patch

from orchestrator import OllamaLiveLane, TurnCoordinator, VoiceEvent


class FakeLive:
    async def stream(self, _text, _context):
        await asyncio.sleep(0.01)
        yield "I’m "
        await asyncio.sleep(0.01)
        yield "checking."


class FakeReasoner:
    async def reason(self, _text, _context):
        await asyncio.sleep(0.06)
        return "The verified result."


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_lane_streams_while_reasoning_runs(self):
        events: list[VoiceEvent] = []

        async def sink(event):
            events.append(event)

        started = time.monotonic()
        live, deep = await TurnCoordinator(FakeLive(), FakeReasoner(), sink).handle(
            "research this"
        )
        elapsed = time.monotonic() - started

        self.assertEqual(live, "I’m checking.")
        self.assertEqual(deep, "The verified result.")
        self.assertLess(elapsed, 0.08)
        types = [event.type for event in events]
        self.assertLess(
            types.index("assistant.live.delta"),
            types.index("assistant.reasoning.final"),
        )
        self.assertEqual(events[0].version, 1)
        self.assertNotIn("transcript_path", events[0].json())


class OllamaLiveLaneTests(unittest.TestCase):
    @patch("orchestrator.urllib.request.urlopen")
    def test_route_context_is_sent_as_system_not_user_history(self, urlopen):
        lane = OllamaLiveLane()
        lane._request(
            "hello",
            (
                "system: This turn belongs to pm.",
                "user: prior question",
                "assistant: prior answer",
            ),
        )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["system", "system", "user", "assistant", "user"],
        )
        self.assertEqual(
            payload["messages"][1]["content"],
            "This turn belongs to pm.",
        )

    def test_live_lane_retries_connection_before_emitting(self):
        class Response:
            def __init__(self):
                self.lines = [
                    b'{"message":{"content":"ready"},"done":false}\n',
                    b'{"done":true}\n',
                ]

            def readline(self):
                return self.lines.pop(0)

            def close(self):
                pass

        lane = OllamaLiveLane()
        calls = 0

        def request(_text, _context):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionRefusedError
            return Response()

        lane._request = request

        async def collect():
            return [delta async for delta in lane.stream("hi", ())]

        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch("orchestrator.asyncio.to_thread", new=inline):
            self.assertEqual(asyncio.run(collect()), ["ready"])
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
