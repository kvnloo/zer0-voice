import unittest

from verify_metrics import DEFAULT_BUDGETS, percentile, verify


def row(*, interrupted=False, scale=1.0):
    return {
        "interrupted": interrupted,
        "asr_seconds": 0.2 * scale,
        "tts_first_seconds": 0.2 * scale,
        "audio_onset_after_endpoint_seconds": 1.5 * scale,
        "estimated_audio_onset_after_user_stop_seconds": 2.0 * scale,
    }


class VoiceMetricVerifierTests(unittest.TestCase):
    def test_nearest_rank_percentile_is_deterministic(self):
        self.assertEqual(percentile([4, 1, 3, 2], 0.50), 2)
        self.assertEqual(percentile([4, 1, 3, 2], 0.95), 4)

    def test_interrupted_turns_do_not_poison_latency_gate(self):
        result = verify([row(interrupted=True, scale=100), row()])
        self.assertTrue(result["ok"])
        self.assertEqual(result["completed_samples"], 1)
        self.assertEqual(result["interrupted_samples"], 1)

    def test_missing_sample_floor_fails(self):
        result = verify([row()], minimum_samples=2)
        self.assertFalse(result["ok"])
        self.assertIn("completed samples 1 < required 2", result["violations"])

    def test_median_regression_fails_named_budget(self):
        slow = [row(scale=2), row(scale=2), row()]
        result = verify(slow, budgets=DEFAULT_BUDGETS)
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                violation.startswith("audio_onset_after_endpoint_seconds")
                for violation in result["violations"]
            )
        )

    def test_pipeline_cohorts_preserve_history_without_poisoning_current_gate(self):
        legacy = row(scale=100)
        current = {**row(), "pipeline": "codex-harness-pcm-v1"}
        result = verify(
            [legacy, current],
            pipeline="codex-harness-pcm-v1",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["ledger_rows"], 2)
        self.assertEqual(result["rows"], 1)


if __name__ == "__main__":
    unittest.main()
