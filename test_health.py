import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from health import RuntimeHealth, assess, read_snapshot


class HealthTests(unittest.TestCase):
    def test_runtime_heartbeat_contains_no_conversation_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            health = RuntimeHealth(path, "run-1")
            health.transition("generating", lane="live", reason="turn-active")
            snapshot = read_snapshot(path)
        self.assertEqual(snapshot["phase"], "generating")
        self.assertEqual(snapshot["lane"], "live")
        self.assertNotIn("text", json.dumps(snapshot))

    def test_concurrent_heartbeats_are_atomic_and_never_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            health = RuntimeHealth(path, "run-concurrent")
            with ThreadPoolExecutor(max_workers=8) as workers:
                futures = [
                    workers.submit(
                        health.transition,
                        "listening" if index % 2 else "generating",
                        lane="mic" if index % 2 else "live",
                    )
                    for index in range(200)
                ]
                for future in futures:
                    future.result()
            snapshot = read_snapshot(path)
            leftovers = tuple(path.parent.glob(f".{path.name}.*.tmp"))
        self.assertEqual(snapshot["run_id"], "run-concurrent")
        self.assertEqual(snapshot["revision"], 200)
        self.assertEqual(leftovers, ())

    def test_stale_heartbeat_is_unhealthy(self):
        ok, reason = assess(
            {
                "updated_ns": 1_000_000_000,
                "phase_since_ns": 1_000_000_000,
                "phase": "listening",
            },
            now_ns=7_000_000_000,
            heartbeat_timeout_seconds=5,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "heartbeat-stale:6.0s")

    def test_stuck_lane_is_unhealthy_despite_fresh_heartbeat(self):
        ok, reason = assess(
            {
                "updated_ns": 30_000_000_000,
                "phase_since_ns": 1_000_000_000,
                "phase": "generating",
            },
            now_ns=30_000_000_000,
            phase_deadlines_seconds={"generating": 25},
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "phase-stale:generating:29.0s")

    def test_runtime_phase_names_have_bounded_deadlines(self):
        for phase in (
            "starting",
            "attaching",
            "transcribing",
            "syncing",
            "generating",
            "speaking",
            "recovering",
        ):
            ok, reason = assess(
                {
                    "updated_ns": 200_000_000_000,
                    "phase_since_ns": 1,
                    "phase": phase,
                },
                now_ns=200_000_000_000,
            )
            self.assertFalse(ok, phase)
            self.assertTrue(reason.startswith(f"phase-stale:{phase}:"), phase)

    def test_idle_listener_can_remain_healthy_indefinitely(self):
        ok, reason = assess(
            {
                "updated_ns": 100_000_000_000,
                "phase_since_ns": 1,
                "phase": "listening",
            },
            now_ns=100_000_000_000,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "healthy")


if __name__ == "__main__":
    unittest.main()
