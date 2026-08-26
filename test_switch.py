import asyncio
import json
import unittest

from events import CommittedVoiceTurn, text_digest
from publisher import PublisherSwitch


def turn(sequence: int, text: str | None = None) -> CommittedVoiceTurn:
    content = text or f"private committed turn {sequence}"
    return CommittedVoiceTurn(
        thread="thread-promotion",
        conversation="conversation-promotion",
        source_id=f"promotion-run-{sequence:x}",
        text=content,
        digest=text_digest(content),
        seq=sequence,
        ts_ns=sequence,
    )


class PublisherSwitchTests(unittest.IsolatedAsyncioTestCase):
    async def test_blue_green_switch_does_not_replace_conversation_owner(self):
        legacy_calls, candidate_calls = [], []

        async def legacy(item):
            legacy_calls.append(item.source_id)
            return True

        async def candidate(item):
            candidate_calls.append(item.source_id)
            return True

        switch = PublisherSwitch(legacy, candidate)
        identity = id(switch)
        self.assertEqual(switch.submit(turn(1)), "legacy")
        self.assertEqual(switch.select("candidate"), "legacy")
        self.assertEqual(switch.submit(turn(2)), "candidate")
        self.assertEqual(switch.select("legacy"), "candidate")
        self.assertEqual(switch.submit(turn(3)), "legacy")
        await switch.drain()

        self.assertEqual(id(switch), identity)
        self.assertEqual(legacy_calls, ["promotion-run-1", "promotion-run-3"])
        self.assertEqual(candidate_calls, ["promotion-run-2"])

    async def test_candidate_rolls_back_without_raising_into_conversation(self):
        legacy_calls = []
        lifecycle = []

        async def legacy(item):
            legacy_calls.append(item.source_id)
            return True

        async def unavailable(_item):
            raise ConnectionError("transcript content must not escape")

        switch = PublisherSwitch(
            legacy,
            unavailable,
            lifecycle.append,
            active="candidate",
            rollback_after=2,
        )
        switch.submit(turn(1))
        await switch.drain()
        self.assertEqual(switch.active, "candidate")
        switch.submit(turn(2))
        await switch.drain()
        self.assertEqual(switch.active, "legacy")
        self.assertEqual(switch.submit(turn(3)), "legacy")
        await switch.drain()

        self.assertEqual(legacy_calls, ["promotion-run-3"])
        serialized = json.dumps(lifecycle)
        self.assertNotIn("private committed turn", serialized)
        self.assertNotIn(text_digest("private committed turn 1"), serialized)
        self.assertEqual(lifecycle[-2]["kind"], "pm.publisher.rollback")
        self.assertEqual(lifecycle[-1]["kind"], "pm.publisher.accepted")

    async def test_saturation_fails_open_and_selects_legacy(self):
        release = asyncio.Event()
        lifecycle = []

        async def legacy(_item):
            return True

        async def blocked(_item):
            await release.wait()
            return True

        switch = PublisherSwitch(
            legacy,
            blocked,
            lifecycle.append,
            active="candidate",
            capacity=1,
        )
        self.assertEqual(switch.submit(turn(1)), "candidate")
        self.assertEqual(switch.submit(turn(2)), "dropped")
        self.assertEqual(switch.active, "legacy")
        release.set()
        await switch.drain()
        self.assertEqual(
            [row["kind"] for row in lifecycle[:2]],
            ["pm.publisher.saturated", "pm.publisher.rollback"],
        )


if __name__ == "__main__":
    unittest.main()
