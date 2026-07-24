import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

from simple_daemon import (
    PipeWireKokoro,
    SimpleVoiceDaemon,
    Status,
    acquire_singleton,
)


class FakeCapture:
    def __init__(self, count):
        self.items = [np.ones(16, dtype=np.float32) for _ in range(count)]
        self.active = self.maximum_active = 0

    async def one(self):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return self.items.pop(0)


class FakeRecognizer:
    def __init__(self):
        self.turn = 0

    async def text(self, audio):
        self.turn += 1
        return f"thought {self.turn}"


class FakeSession:
    def __init__(self):
        self.prompts = []
        self.closed = self.active = self.maximum_active = 0

    async def respond(self, prompt):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        self.active -= 1
        return f"reply {len(self.prompts)}"

    async def close(self):
        self.closed += 1


class SimpleDaemonTests(unittest.IsolatedAsyncioTestCase):
    async def test_singleton_rejects_a_second_voice_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.lock"
            first = acquire_singleton(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "another voice daemon"):
                    acquire_singleton(path)
            finally:
                os.close(first)

    async def test_cancelled_reply_terminates_pipewire_player(self):
        class Player:
            def __init__(self):
                self.returncode = None
                self.started = asyncio.Event()
                self.terminated = False

            async def communicate(self, _wav):
                self.started.set()
                await asyncio.Event().wait()

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            async def wait(self):
                return self.returncode

        player = Player()
        speaker = PipeWireKokoro(
            "http://kokoro",
            "voice",
            output="effect_input.test",
            latency="40ms",
        )
        with patch(
            "simple_daemon.asyncio.to_thread",
            new=AsyncMock(return_value=b"wav"),
        ), patch(
            "simple_daemon.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=player),
        ):
            task = asyncio.create_task(speaker.say("stale reply"))
            await asyncio.wait_for(player.started.wait(), timeout=0.1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(player.terminated)

    async def test_ten_turns_are_strictly_sequential_and_observable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, session, output = FakeCapture(10), FakeSession(), []
            daemon = SimpleVoiceDaemon(
                session,  # type: ignore[arg-type]
                capture,
                FakeRecognizer(),
                status=Status(root / "health.json", root / "metrics.jsonl"),
                emit=output.append,
            )
            self.assertEqual(await daemon.run(max_turns=10), 10)
            self.assertEqual(session.prompts, [f"thought {i}" for i in range(1, 11)])
            self.assertEqual((session.maximum_active, capture.maximum_active), (1, 1))
            self.assertEqual(session.closed, 1)
            self.assertEqual(len(output), 20)
            metrics = (root / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(metrics), 10)
            self.assertEqual(
                [json.loads(line)["turn"] for line in metrics], list(range(1, 11))
            )
            health = json.loads((root / "health.json").read_text())
            self.assertEqual((health["phase"], health["turns"]), ("stopped", 10))

    async def test_failure_is_fatal_visible_and_closes_once(self):
        class Broken(FakeSession):
            async def respond(self, prompt):
                raise RuntimeError("model unavailable")

        with tempfile.TemporaryDirectory() as directory:
            health = Path(directory) / "health.json"
            session = Broken()
            daemon = SimpleVoiceDaemon(
                session,  # type: ignore[arg-type]
                FakeCapture(1),
                FakeRecognizer(),
                status=Status(health, None),
                emit=lambda _: None,
            )
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                await daemon.run(max_turns=1)
            self.assertEqual(session.closed, 1)
            state = json.loads(health.read_text())
            self.assertEqual(state["phase"], "error")
            self.assertFalse(state["healthy"])

    async def test_response_deadline_cancels_stale_turn_and_resumes_listening(self):
        class TwoCaptures(FakeCapture):
            def __init__(self):
                super().__init__(2)
                self.second_capture_started = asyncio.Event()

            async def one(self):
                if len(self.items) == 1:
                    self.second_capture_started.set()
                return await super().one()

        class OneStaleTurn(FakeSession):
            def __init__(self):
                super().__init__()
                self.cancelled = asyncio.Event()

            async def respond(self, prompt):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled.set()
                        raise
                return "fresh reply"

        capture, session, output = TwoCaptures(), OneStaleTurn(), []
        daemon = SimpleVoiceDaemon(
            session,  # type: ignore[arg-type]
            capture,
            FakeRecognizer(),
            emit=output.append,
            response_timeout=0.01,
        )

        self.assertEqual(
            await asyncio.wait_for(daemon.run(max_turns=1), timeout=0.2),
            1,
        )
        self.assertTrue(session.cancelled.is_set())
        self.assertTrue(capture.second_capture_started.is_set())
        self.assertEqual(session.prompts, ["thought 1", "thought 2"])
        self.assertEqual(session.closed, 1)
        self.assertTrue(any("timed out" in line.lower() for line in output))

    async def test_response_timeout_is_a_hard_deadline_not_a_fatal_daemon_error(self):
        class HangingThenListening(FakeCapture):
            def __init__(self):
                super().__init__(1)
                self.calls = 0
                self.listening_again = asyncio.Event()

            async def one(self):
                self.calls += 1
                if self.calls == 1:
                    return await super().one()
                self.listening_again.set()
                await asyncio.Event().wait()

        class HangingSession(FakeSession):
            async def respond(self, prompt):
                await asyncio.Event().wait()

        session, capture = HangingSession(), HangingThenListening()
        daemon = SimpleVoiceDaemon(
            session,  # type: ignore[arg-type]
            capture,
            FakeRecognizer(),
            emit=lambda _: None,
            response_timeout=0.01,
        )
        task = asyncio.create_task(daemon.run())
        await asyncio.wait_for(capture.listening_again.wait(), timeout=0.1)
        self.assertFalse(task.done(), "daemon must still be listening after the deadline")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(session.closed, 1)

    async def test_health_heartbeat_advances_while_microphone_is_idle(self):
        class IdleCapture:
            entered = asyncio.Event()

            async def one(self):
                self.entered.set()
                await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as directory:
            health = Path(directory) / "health.json"
            capture = IdleCapture()
            daemon = SimpleVoiceDaemon(
                FakeSession(),  # type: ignore[arg-type]
                capture,
                FakeRecognizer(),
                status=Status(
                    health,
                    None,
                    heartbeat_seconds=0.005,
                ),
                emit=lambda _: None,
            )
            task = asyncio.create_task(daemon.run())
            await capture.entered.wait()
            first = json.loads(health.read_text())
            await asyncio.sleep(0.015)
            second = json.loads(health.read_text())
            self.assertEqual(second["phase"], "listening")
            self.assertGreater(second["updated_ns"], first["updated_ns"])
            self.assertEqual(second["pid"], os.getpid())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    unittest.main()
