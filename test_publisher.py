import asyncio
import json
import unittest
from unittest.mock import patch

from floor import SubmissionDecision, TurnOwner
from publisher import (
    build_committed_turn,
    publish_best_effort,
    send_committed_turn,
)


class PublisherTests(unittest.TestCase):
    @patch("publisher.urllib.request.urlopen")
    def test_one_turn_owner_commit_produces_one_typed_canonical_call(
        self,
        urlopen,
    ):
        urlopen.return_value.__enter__.return_value.status = 202
        owner = TurnOwner(settle_seconds=0.01)
        partial = owner.observe("Ship the typed publisher.", now=0.0)
        committed = owner.due(now=0.02)
        self.assertEqual((partial.action, committed.action), ("hold", "submit"))
        self.assertEqual(owner.due(now=0.03).action, "idle")

        turn = build_committed_turn(
            committed,
            thread="thread-7",
            conversation="thread-7",
            run_id="voice-run-a",
            commit_sequence=4,
            ts_ns=99,
        )
        send_committed_turn("http://relay/v1/voice/committed", turn)

        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://relay/v1/voice/committed")
        self.assertTrue(body["committed"])
        self.assertEqual(body["source_id"], "voice-run-a-4")
        self.assertEqual(body["thread"], "thread-7")
        self.assertEqual(body["conversation"], "thread-7")
        self.assertEqual(body["text"], "Ship the typed publisher.")
        self.assertEqual(len(body["digest"]), 64)
        self.assertNotIn("kind", body)
        self.assertNotIn("boundary", body)
        urlopen.assert_called_once()

    @patch("publisher.urllib.request.urlopen")
    def test_partial_rejected_or_cancelled_decisions_cannot_publish(
        self,
        urlopen,
    ):
        for action in ("hold", "reject", "cancel", "idle"):
            with self.assertRaisesRegex(ValueError, "TurnOwner submission"):
                build_committed_turn(
                    SubmissionDecision(action, "unstable partial", action),
                    thread="thread-7",
                    conversation="thread-7",
                    run_id="voice-run-a",
                    commit_sequence=1,
                    ts_ns=99,
                )
        urlopen.assert_not_called()


class AsyncPublisherTests(unittest.IsolatedAsyncioTestCase):
    @patch("publisher.urllib.request.urlopen")
    async def test_retry_reuses_identical_identity_and_logs_no_text(
        self,
        urlopen,
    ):
        class Response:
            status = 202

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

        secret = "private committed transcript"
        turn = build_committed_turn(
            SubmissionDecision("submit", secret, "deadline"),
            thread="thread-7",
            conversation="thread-7",
            run_id="voice-run-a",
            commit_sequence=9,
            ts_ns=99,
        )
        lifecycle = []
        urlopen.side_effect = [OSError("temporary"), Response()]

        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch("publisher.asyncio.to_thread", side_effect=inline):
            accepted = await publish_best_effort(
                "http://relay/v1/voice/committed",
                turn,
                lifecycle.append,
                retry_delay=0,
            )
        self.assertTrue(accepted)
        requests = [call.args[0] for call in urlopen.call_args_list]
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].data, requests[1].data)
        self.assertEqual(
            [row["kind"] for row in lifecycle],
            ["pm.intake.error", "pm.intake.accepted"],
        )
        self.assertEqual(
            {row["source_id"] for row in lifecycle},
            {"voice-run-a-9"},
        )
        self.assertNotIn(secret, json.dumps(lifecycle))
        self.assertNotIn(turn.digest, json.dumps(lifecycle))

    async def test_final_failure_is_best_effort_and_privacy_safe(self):
        turn = build_committed_turn(
            SubmissionDecision("submit", "private text", "deadline"),
            thread="thread-7",
            conversation="thread-7",
            run_id="voice-run-a",
            commit_sequence=2,
            ts_ns=99,
        )
        lifecycle = []

        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch(
            "publisher.urllib.request.urlopen",
            side_effect=TimeoutError("private text must not be logged"),
        ), patch("publisher.asyncio.to_thread", side_effect=inline):
            accepted = await publish_best_effort(
                "http://relay/v1/voice/committed",
                turn,
                lifecycle.append,
                attempts=1,
            )
        self.assertFalse(accepted)
        self.assertEqual(lifecycle[0]["error"], "TimeoutError")
        self.assertNotIn("private text", json.dumps(lifecycle))

    async def test_http_failure_resource_is_closed_before_returning(self):
        class ClosableFailure(OSError):
            closed = False

            def close(self):
                self.closed = True

        failure = ClosableFailure("rejected")
        turn = build_committed_turn(
            SubmissionDecision("submit", "private text", "deadline"),
            thread="thread-7",
            conversation="thread-7",
            run_id="voice-run-a",
            commit_sequence=3,
            ts_ns=99,
        )

        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        with patch(
            "publisher.urllib.request.urlopen",
            side_effect=failure,
        ), patch("publisher.asyncio.to_thread", side_effect=inline):
            accepted = await publish_best_effort(
                "http://relay/v1/voice/committed",
                turn,
                attempts=1,
            )

        self.assertFalse(accepted)
        self.assertTrue(failure.closed)


if __name__ == "__main__":
    unittest.main()
