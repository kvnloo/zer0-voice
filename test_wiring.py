import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from floor import SubmissionDecision, TurnOwner

from wiring import VoicePMWiring


class VoicePMWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_relay_candidate_is_dormant_until_hot_selected(self):
        publish = AsyncMock(return_value=True)
        with patch("wiring.publish_best_effort", publish):
            wiring = VoicePMWiring.for_relay(
                "http://relay/v1/voice/committed",
                thread="thread-1",
                conversation="conversation-1",
                run_id="run-relay",
                attempts=1,
                timeout=0.01,
            )
            wiring.schedule(SubmissionDecision("submit", "legacy", "forced"))
            await wiring.drain()
            publish.assert_not_awaited()

            self.assertEqual(wiring.select("candidate"), "legacy")
            wiring.schedule(SubmissionDecision("submit", "candidate", "forced"))
            await wiring.drain()

        publish.assert_awaited_once()
        self.assertEqual(
            publish.await_args.args[0],
            "http://relay/v1/voice/committed",
        )
        self.assertEqual(publish.await_args.args[1].source_id, "run-relay-2")
        self.assertEqual(publish.await_args.kwargs["attempts"], 1)
        self.assertEqual(publish.await_args.kwargs["timeout"], 0.01)

    async def test_only_committed_owner_decision_is_scheduled(self):
        delivered = []

        async def candidate(turn):
            delivered.append(turn)
            return True

        wiring = VoicePMWiring(
            thread="thread-1",
            conversation="conversation-1",
            run_id="run-1",
            candidate=candidate,
            active="candidate",
            clock_ns=lambda: 99,
        )
        owner = TurnOwner(settle_seconds=0.01)
        partial = owner.observe("private partial", now=0.0)
        with self.assertRaisesRegex(ValueError, "committed"):
            wiring.schedule(partial)
        committed = owner.due(now=0.02)
        self.assertEqual(wiring.schedule(committed), "candidate")
        await wiring.drain()

        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].source_id, "run-1-1")
        self.assertEqual(delivered[0].text, "private partial")
        self.assertEqual(delivered[0].ts_ns, 99)
        self.assertEqual(wiring.commit_sequence, 1)

    async def test_legacy_candidate_selection_is_hot_and_preserves_boundary(self):
        legacy, candidate = [], []

        async def publish_legacy(turn):
            legacy.append(turn.source_id)
            return True

        async def publish_candidate(turn):
            candidate.append(turn.source_id)
            return True

        wiring = VoicePMWiring(
            thread="thread-1",
            conversation="conversation-1",
            run_id="run-hot",
            legacy=publish_legacy,
            candidate=publish_candidate,
        )
        identity = id(wiring)
        self.assertEqual(
            wiring.schedule(SubmissionDecision("submit", "one", "forced")),
            "legacy",
        )
        self.assertEqual(wiring.select("candidate"), "legacy")
        self.assertEqual(
            wiring.schedule(SubmissionDecision("submit", "two", "forced")),
            "candidate",
        )
        self.assertEqual(wiring.select("legacy"), "candidate")
        self.assertEqual(
            wiring.schedule(SubmissionDecision("submit", "three", "forced")),
            "legacy",
        )
        await wiring.drain()

        self.assertEqual(id(wiring), identity)
        self.assertEqual(legacy, ["run-hot-1", "run-hot-3"])
        self.assertEqual(candidate, ["run-hot-2"])

    async def test_scheduling_never_awaits_relay_or_response_work(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked(_turn):
            started.set()
            await release.wait()
            return True

        wiring = VoicePMWiring(
            thread="thread-1",
            conversation="conversation-1",
            run_id="run-latency",
            candidate=blocked,
            active="candidate",
        )
        lane = wiring.schedule(
            SubmissionDecision("submit", "do not delay the reply", "deadline")
        )

        self.assertEqual(lane, "candidate")
        self.assertEqual(len(wiring.switch.tasks), 1)
        await asyncio.wait_for(started.wait(), timeout=0.1)
        self.assertFalse(next(iter(wiring.switch.tasks)).done())
        release.set()
        await wiring.drain()

    async def test_lifecycle_contains_identity_and_state_but_no_content(self):
        secret = "private transcript prompt response tool content"
        lifecycle = []

        async def rejected(_turn):
            raise RuntimeError(secret)

        wiring = VoicePMWiring(
            thread="thread-secret",
            conversation="conversation-secret",
            run_id="run-private",
            candidate=rejected,
            lifecycle=lifecycle.append,
            active="candidate",
            rollback_after=1,
        )
        wiring.schedule(SubmissionDecision("submit", secret, "deadline"))
        await wiring.drain()

        serialized = json.dumps(lifecycle, sort_keys=True)
        self.assertNotIn(secret, serialized)
        self.assertNotIn("digest", serialized)
        self.assertNotIn("prompt", serialized)
        self.assertNotIn("response", serialized)
        self.assertNotIn("tool", serialized)
        self.assertEqual(wiring.active, "legacy")
        self.assertEqual(
            [row["kind"] for row in lifecycle],
            [
                "pm.publisher.error",
                "pm.publisher.rejected",
                "pm.publisher.rollback",
            ],
        )


if __name__ == "__main__":
    unittest.main()
