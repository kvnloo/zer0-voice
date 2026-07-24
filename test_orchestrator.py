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


if __name__ == "__main__":
    unittest.main()
