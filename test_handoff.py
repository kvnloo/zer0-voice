import tempfile
import time
import unittest
from pathlib import Path

from handoff import Candidate, HandoffError, handoff, ready


def health(pid: int, phase: str = "listening"):
    now = time.time_ns()
    return {
        "pid": pid,
        "phase": phase,
        "updated_ns": now,
        "phase_since_ns": now,
    }


class Controls:
    def __init__(self, active: Path, candidate: Path, *, candidate_fails=False):
        self.active = active
        self.candidate = candidate
        self.candidate_fails = candidate_fails
        self.states = {
            active: {
                "ok": True,
                "mic": "continuous",
                "capture_active": True,
                "live_model": "gpt-live",
            },
            candidate: {
                "ok": True,
                "mic": "muted",
                "capture_active": False,
                "live_model": "gpt-live",
            },
        }
        self.calls = []

    async def __call__(self, path, command):
        self.calls.append((path, command))
        state = self.states[path]
        if command.get("mic") == "muted":
            state = {**state, "mic": "muted", "capture_active": False}
        elif "mic" in command:
            if path == self.candidate and self.candidate_fails:
                raise OSError("candidate control died")
            state = {**state, "mic": command["mic"], "capture_active": True}
        self.states[path] = state
        return state


class HandoffTests(unittest.IsolatedAsyncioTestCase):
    def test_candidate_must_be_fresh_attached_and_muted(self):
        status = {
            "mic": "muted",
            "capture_active": False,
            "live_model": "gpt-live",
        }
        self.assertEqual(ready(health(7), status), (True, "ready"))
        self.assertEqual(
            ready(health(7, "generating"), status),
            (False, "candidate-phase:generating"),
        )
        self.assertEqual(
            ready(health(7), {**status, "capture_active": True})[1],
            "candidate-already-capturing",
        )

    async def test_switch_mutes_old_then_activates_warm_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = Candidate(root / "old.sock", root / "old.json")
            new = Candidate(root / "new.sock", root / "new.json")
            old.health.write_text(__import__("json").dumps(health(1)))
            new.health.write_text(__import__("json").dumps(health(2)))
            controls = Controls(old.control, new.control)
            result = await handoff(
                old,
                new,
                control_request=controls,
                readiness_timeout=0.01,
            )
        mutations = [call for call in controls.calls if call[1]]
        self.assertEqual(
            mutations,
            [
                (old.control, {"mic": "muted"}),
                (new.control, {"mic": "continuous"}),
            ],
        )
        self.assertEqual(result["status"], "candidate-active")
        self.assertTrue(result["old_retained_for_rollback"])

    async def test_failed_candidate_restores_old_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = Candidate(root / "old.sock", root / "old.json")
            new = Candidate(root / "new.sock", root / "new.json")
            old.health.write_text(__import__("json").dumps(health(1)))
            new.health.write_text(__import__("json").dumps(health(2)))
            controls = Controls(
                old.control,
                new.control,
                candidate_fails=True,
            )
            with self.assertRaisesRegex(OSError, "candidate control died"):
                await handoff(
                    old,
                    new,
                    control_request=controls,
                    readiness_timeout=0.01,
                )
        self.assertTrue(controls.states[old.control]["capture_active"])
        self.assertEqual(controls.states[old.control]["mic"], "continuous")

    async def test_never_mutes_active_when_candidate_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = Candidate(root / "old.sock", root / "old.json")
            new = Candidate(root / "new.sock", root / "new.json")
            old.health.write_text(__import__("json").dumps(health(1)))
            new.health.write_text(__import__("json").dumps(health(2, "attaching")))
            controls = Controls(old.control, new.control)
            with self.assertRaises(HandoffError):
                await handoff(
                    old,
                    new,
                    control_request=controls,
                    readiness_timeout=0.01,
                )
        self.assertFalse(any(command for _, command in controls.calls))

    async def test_never_switches_during_an_active_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = Candidate(root / "old.sock", root / "old.json")
            new = Candidate(root / "new.sock", root / "new.json")
            old.health.write_text(__import__("json").dumps(health(1, "generating")))
            new.health.write_text(__import__("json").dumps(health(2)))
            controls = Controls(old.control, new.control)
            with self.assertRaisesRegex(HandoffError, "not idle"):
                await handoff(
                    old,
                    new,
                    control_request=controls,
                    readiness_timeout=0.01,
                )
        self.assertFalse(any(command for _, command in controls.calls))

    async def test_hung_candidate_control_is_bounded_and_keeps_active(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = Candidate(root / "old.sock", root / "old.json")
            new = Candidate(root / "new.sock", root / "new.json")
            old.health.write_text(__import__("json").dumps(health(1)))
            new.health.write_text(__import__("json").dumps(health(2)))
            controls = Controls(old.control, new.control)

            async def hanging(path, command):
                if path == new.control:
                    await __import__("asyncio").sleep(60)
                return await controls(path, command)

            with self.assertRaises(HandoffError):
                await handoff(
                    old,
                    new,
                    control_request=hanging,
                    readiness_timeout=0.01,
                    request_timeout=0.005,
                )
        self.assertTrue(controls.states[old.control]["capture_active"])
        self.assertFalse(any(command for _, command in controls.calls))


if __name__ == "__main__":
    unittest.main()
