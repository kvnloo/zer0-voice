import asyncio
import unittest
from pathlib import Path

from floor import TurnOwner
from providers import EmptyCompletionError, StreamEvent
from simple import SerialSpeaker, SimpleVoiceError, SimpleVoiceSession


class ContextServer:
    def __init__(self):
        self.forks = []
        self.prompts = []

    async def read_thread(self, thread):
        self.read = thread
        return {
            "turns": [
                {"id": "old", "status": "completed"},
                {"id": "typing", "status": "inProgress"},
            ]
        }

    async def fork_thread(self, thread, **kwargs):
        self.forks.append((thread, kwargs))
        return "voice-stable"

    async def stream_turn(self, thread, text, effort=None):
        self.prompts.append((thread, text, effort))
        prior = "first" if len(self.prompts) == 1 else "remembered first"
        yield type(
            "Event",
            (),
            {
                "subject": f"turn:{len(self.prompts)}",
                "kind": "assistant.delta",
                "payload": {"text": prior},
            },
        )
        yield type(
            "Event",
            (),
            {
                "subject": f"turn:{len(self.prompts)}",
                "kind": "assistant.completed",
                "payload": {},
            },
        )

    async def interrupt(self, thread, turn):
        raise AssertionError("simple mode never interrupts")


class RecordingSpeaker:
    def __init__(self):
        self.spoken = []
        self.active = 0
        self.maximum_active = 0

    async def say(self, text):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0)
        self.spoken.append(text)
        self.active -= 1


class SimpleVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_attaches_once_and_reuses_context_thread_across_turns(self):
        server = ContextServer()
        speaker = RecordingSpeaker()
        session = await SimpleVoiceSession.attach(
            server,
            "harness-thread",
            cwd=Path("/workspace/project"),
            speaker=speaker,
        )
        self.assertEqual(session.binding.inherited_through_turn, "old")
        self.assertEqual(len(server.forks), 1)
        self.assertEqual(await session.respond("one"), "first")
        self.assertEqual(await session.respond("what did I say?"), "remembered first")
        self.assertEqual(len(server.forks), 1)
        self.assertEqual(
            [prompt[0] for prompt in server.prompts],
            ["voice-stable", "voice-stable"],
        )
        self.assertEqual(speaker.spoken, ["first", "remembered first"])
        await session.close()

    async def test_model_and_speaker_are_serialized_under_concurrent_input(self):
        server = ContextServer()
        speaker = RecordingSpeaker()
        session = await SimpleVoiceSession.attach(
            server, "harness", cwd=Path("."), speaker=speaker
        )
        replies = await asyncio.gather(
            session.respond("one"),
            session.respond("two"),
            session.respond("three"),
        )
        self.assertEqual(replies, ["first", "remembered first", "remembered first"])
        self.assertEqual(speaker.maximum_active, 1)
        self.assertEqual(session.turns, 3)
        await session.close()

    async def test_bounded_ten_turn_cohort_keeps_one_thread_and_full_history(self):
        server = ContextServer()
        speaker = RecordingSpeaker()
        session = await SimpleVoiceSession.attach(
            server, "harness", cwd=Path("."), speaker=speaker
        )
        for index in range(10):
            await asyncio.wait_for(
                session.respond(f"cohort turn {index}"),
                timeout=1.0,
            )
        self.assertEqual(session.turns, 10)
        self.assertEqual(len(server.forks), 1)
        self.assertEqual(len(server.prompts), 10)
        self.assertEqual({thread for thread, _, _ in server.prompts}, {"voice-stable"})
        self.assertEqual(len(speaker.spoken), 10)
        await session.close()

    async def test_single_turn_owner_combines_fragments_before_one_submission(self):
        server = ContextServer()
        speaker = RecordingSpeaker()
        session = await SimpleVoiceSession.attach(
            server,
            "harness",
            cwd=Path("."),
            speaker=speaker,
            owner=TurnOwner(settle_seconds=0.4, incomplete_seconds=1.0),
        )
        first = session.observe("we need to", now=0.0)
        second = session.observe("keep context.", now=0.5)
        self.assertEqual((first.action, second.action), ("hold", "hold"))
        self.assertIsNone(await session.submit_due(now=0.8))
        response = await session.submit_due(now=0.91)
        self.assertEqual(response, "first")
        self.assertEqual(server.prompts[0][1], "we need to keep context.")
        self.assertEqual(len(server.prompts), 1)
        await session.close()

    async def test_empty_completion_fails_visibly_and_never_speaks(self):
        errors = []
        speaker = RecordingSpeaker()

        class EmptyProvider:
            async def stream(self, text, *, context=(), effort=None):
                del text, context, effort
                if False:
                    yield StreamEvent("delta", "never")
                raise EmptyCompletionError("empty")

            async def interrupt(self, turn_id):
                raise AssertionError(turn_id)

        session = SimpleVoiceSession(
            binding=None,  # type: ignore[arg-type]
            provider=EmptyProvider(),
            speaker=SerialSpeaker(speaker),
            on_error=errors.append,
        )
        with self.assertRaisesRegex(SimpleVoiceError, "empty"):
            await session.respond("hello")
        self.assertEqual(speaker.spoken, [])
        self.assertEqual(len(errors), 1)
        await session.close()

    async def test_cancelling_speech_aborts_backend_and_leaves_speaker_reusable(self):
        """A turn deadline must not leave a stale TTS job ahead of later replies."""

        class CancellableSpeaker:
            def __init__(self):
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()
                self.release = asyncio.Event()
                self.calls = []

            async def say(self, text):
                self.calls.append(text)
                if len(self.calls) == 1:
                    self.started.set()
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        self.cancelled.set()
                        raise

        backend = CancellableSpeaker()
        speaker = SerialSpeaker(backend)
        first = asyncio.create_task(speaker.say("stale reply"))
        await asyncio.wait_for(backend.started.wait(), timeout=0.1)

        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        try:
            await asyncio.wait_for(backend.cancelled.wait(), timeout=0.1)
            await asyncio.wait_for(speaker.say("fresh reply"), timeout=0.1)
            self.assertEqual(backend.calls, ["stale reply", "fresh reply"])
            await asyncio.wait_for(speaker.close(), timeout=0.1)
        finally:
            backend.release.set()
            if speaker.worker is not None and not speaker.worker.done():
                await speaker.close()


if __name__ == "__main__":
    unittest.main()
