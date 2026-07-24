import unittest

from canary import audit
from verify_metrics import PRODUCTION_PIPELINE


def metric(scale=1.0, run_id=None, ts_ns=None):
    return {
        "pipeline": PRODUCTION_PIPELINE,
        **({"run_id": run_id} if run_id else {}),
        **({"ts_ns": ts_ns} if ts_ns else {}),
        "interrupted": False,
        "speech_to_partial_seconds": 0.1 * scale,
        "asr_seconds": 0.1 * scale,
        "tts_first_seconds": 0.1 * scale,
        "audio_onset_after_endpoint_seconds": 0.5 * scale,
        "estimated_audio_onset_after_user_stop_seconds": 1.0 * scale,
    }


def healthy(now_ns, run_id=None):
    return {
        "updated_ns": now_ns - 100_000_000,
        "phase_since_ns": now_ns - 100_000_000,
        "phase": "listening",
        **({"run_id": run_id} if run_id else {}),
    }


def cycle(start_ns, text):
    return [
        {"kind": "asr.final", "text": text, "ts_ns": start_ns},
        {
            "kind": "voice.lane.state",
            "lane": "live",
            "state": "running",
            "ts_ns": start_ns + 100_000_000,
        },
        {
            "kind": "voice.response",
            "lane": "live",
            "text": f"response to {text}",
            "ts_ns": start_ns + 400_000_000,
        },
    ]


class CanaryTests(unittest.TestCase):
    def test_collects_without_leaking_transcript_fields(self):
        now_ns = 1_000_000_000
        events = [
            {"kind": "voice.starting", "ts_ns": 1},
            {"kind": "asr.final", "text": "private words", "ts_ns": 2},
            {
                "kind": "voice.lane.state",
                "lane": "live",
                "state": "running",
                "ts_ns": 3,
            },
            {
                "kind": "voice.response",
                "lane": "live",
                "text": "private response",
                "ts_ns": 4,
            },
        ]
        result = audit([], events, now_ns=now_ns, health=healthy(now_ns))
        self.assertEqual(result["status"], "collecting")
        self.assertNotIn("private words", str(result))
        self.assertNotIn("private response", str(result))

    def test_passes_only_after_full_green_window(self):
        now_ns = 20_000_000_000
        events = [{"kind": "voice.starting", "ts_ns": 1}]
        for index in range(10):
            events.extend(cycle(1_000_000_000 + index * 1_000_000_000, f"turn {index}"))
        result = audit(
            [
                metric(ts_ns=1_000_000_000 + index * 1_000_000_000)
                for index in range(10)
            ],
            events,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["promotion"]["verdict"], "promote")
        self.assertTrue(result["promotion"]["eligible"])
        self.assertEqual(result["counts"]["canonical_completions"], 10)
        self.assertEqual(
            result["latencies"]["final_to_response_seconds"]["p95"],
            0.4,
        )

    def test_runtime_error_fails_even_with_fast_samples(self):
        now_ns = 3_000_000_000
        result = audit(
            [metric(ts_ns=1_000_000_000)],
            [
                {"kind": "voice.starting", "ts_ns": 1},
                *cycle(1_000_000_000, "one turn"),
                {"kind": "codex.authoritative.error", "ts_ns": 2_000_000_000},
            ],
            minimum_samples=1,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["counts"]["errors"], 1)
        self.assertFalse(result["promotion"]["eligible"])
        self.assertEqual(result["promotion"]["verdict"], "reject")

    def test_restart_id_excludes_previous_run_metrics(self):
        now_ns = 2_000_000_000
        result = audit(
            [
                metric(scale=100, run_id="old", ts_ns=100),
                metric(run_id="new", ts_ns=1_000_000_000),
            ],
            [
                {"kind": "voice.starting", "run_id": "new", "ts_ns": 1},
                *cycle(1_000_000_000, "new run"),
            ],
            minimum_samples=1,
            now_ns=now_ns,
            health=healthy(now_ns, "new"),
        )
        self.assertEqual(result["status"], "collecting")
        self.assertEqual(result["gate"]["rows"], 1)
        self.assertFalse(result["promotion"]["eligible"])

    def test_supplied_start_timestamp_excludes_old_failures_and_metrics(self):
        now_ns = 10_000_000_000
        metrics = [metric(scale=100, ts_ns=1_000_000_000)]
        events = [
            {"kind": "voice.starting", "ts_ns": 1},
            {"kind": "voice.live.error", "ts_ns": 1_000_000_000},
            *cycle(2_000_000_000, "duplicate"),
            *cycle(2_500_000_000, "duplicate"),
        ]
        for index in range(10):
            timestamp = 6_000_000_000 + index * 100_000_000
            metrics.append(metric(ts_ns=timestamp))
            events.extend(cycle(timestamp, f"fresh turn {index}"))
        result = audit(
            metrics,
            events,
            minimum_samples=1,
            start_ns=5_000_000_000,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["gate"]["rows"], 10)
        self.assertEqual(result["counts"]["errors"], 0)
        self.assertEqual(result["counts"]["rapid_duplicate_finals"], 0)

    def test_rapid_duplicate_final_fails_without_echoing_text(self):
        now_ns = 4_000_000_000
        private = "private duplicated command"
        events = [{"kind": "voice.starting", "ts_ns": 1}]
        events.extend(cycle(1_000_000_000, private))
        events.extend(cycle(2_000_000_000, private.upper() + "!"))
        result = audit(
            [metric(ts_ns=1_000_000_000), metric(ts_ns=2_000_000_000)],
            events,
            minimum_samples=2,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["counts"]["rapid_duplicate_finals"], 1)
        self.assertNotIn(private, str(result).lower())

    def test_out_of_order_live_response_fails_sequence(self):
        now_ns = 3_000_000_000
        result = audit(
            [metric(ts_ns=1_000_000_000)],
            [
                {"kind": "voice.starting", "ts_ns": 1},
                {"kind": "asr.final", "text": "turn", "ts_ns": 1_000_000_000},
                {
                    "kind": "voice.response",
                    "lane": "live",
                    "text": "response",
                    "ts_ns": 1_500_000_000,
                },
            ],
            minimum_samples=1,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["counts"]["sequence_violations"], 1)

    def test_empty_live_response_is_counted_and_rejected(self):
        now_ns = 3_000_000_000
        result = audit(
            [metric(ts_ns=1_000_000_000)],
            [
                {"kind": "voice.starting", "ts_ns": 1},
                {"kind": "asr.final", "text": "private turn", "ts_ns": 1_000_000_000},
                {
                    "kind": "voice.lane.state",
                    "lane": "live",
                    "state": "running",
                    "ts_ns": 1_100_000_000,
                },
                {
                    "kind": "voice.response",
                    "lane": "live",
                    "text": "",
                    "ts_ns": 1_500_000_000,
                },
            ],
            minimum_samples=1,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["counts"]["empty_model_outputs"], 1)
        self.assertEqual(result["counts"]["canonical_completions"], 0)
        self.assertIn("empty model outputs 1 > allowed 0", result["invariants"])
        self.assertEqual(result["promotion"]["verdict"], "reject")

    def test_interruption_budget_is_explicit_and_bounded(self):
        now_ns = 4_000_000_000
        events = [
            {"kind": "voice.starting", "ts_ns": 1},
            *cycle(1_000_000_000, "completed"),
            {"kind": "voice.interrupted", "ts_ns": 2_000_000_000},
            {"kind": "voice.interrupted", "ts_ns": 2_500_000_000},
        ]
        result = audit(
            [metric(ts_ns=1_000_000_000)],
            events,
            minimum_samples=1,
            max_interruptions=1,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("interruptions 2 > allowed 1", result["invariants"])

    def test_one_interruption_within_budget_does_not_fail_small_cohort(self):
        now_ns = 4_000_000_000
        result = audit(
            [metric(ts_ns=1_000_000_000)],
            [
                {"kind": "voice.starting", "ts_ns": 1},
                *cycle(1_000_000_000, "completed"),
                {"kind": "voice.interrupted", "ts_ns": 2_000_000_000},
            ],
            minimum_samples=1,
            max_interruptions=1,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "collecting")
        self.assertEqual(result["invariants"], [])
        self.assertFalse(result["promotion"]["eligible"])

    def test_one_successful_turn_is_explicitly_insufficient(self):
        now_ns = 3_000_000_000
        result = audit(
            [metric(ts_ns=1_000_000_000)],
            [
                {"kind": "voice.starting", "ts_ns": 1},
                *cycle(1_000_000_000, "single good turn"),
            ],
            minimum_samples=1,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "collecting")
        self.assertEqual(result["promotion"]["verdict"], "hold")
        self.assertEqual(result["promotion"]["required_completed_turns"], 10)
        self.assertEqual(result["promotion"]["observed_completed_turns"], 1)
        self.assertFalse(result["promotion"]["eligible"])

    def test_failed_latency_gate_never_promotes(self):
        now_ns = 20_000_000_000
        events = [{"kind": "voice.starting", "ts_ns": 1}]
        metrics = []
        for index in range(10):
            timestamp = 1_000_000_000 + index * 1_000_000_000
            events.extend(cycle(timestamp, f"slow turn {index}"))
            metrics.append(metric(scale=10, ts_ns=timestamp))
        result = audit(
            metrics,
            events,
            now_ns=now_ns,
            health=healthy(now_ns),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["promotion"]["verdict"], "reject")
        self.assertFalse(result["promotion"]["eligible"])

    def test_stale_health_fails_green_log_cohort(self):
        now_ns = 20_000_000_000
        result = audit(
            [metric(ts_ns=1_000_000_000)],
            [{"kind": "voice.starting", "ts_ns": 1}, *cycle(1_000_000_000, "turn")],
            minimum_samples=1,
            now_ns=now_ns,
            health={
                "updated_ns": 1,
                "phase_since_ns": 1,
                "phase": "listening",
            },
        )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["health"]["ok"])

    def test_fresh_heartbeat_from_another_run_is_rejected(self):
        now_ns = 3_000_000_000
        result = audit(
            [metric(run_id="new", ts_ns=1_000_000_000)],
            [
                {"kind": "voice.starting", "run_id": "new", "ts_ns": 1},
                *cycle(1_000_000_000, "turn"),
            ],
            minimum_samples=1,
            now_ns=now_ns,
            health=healthy(now_ns, "old"),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["health"]["reason"], "heartbeat-run-mismatch")


if __name__ == "__main__":
    unittest.main()
