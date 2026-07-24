import unittest

from floor import (
    AdaptiveEndpoint,
    AgentPriority,
    FloorDecision,
    FloorPolicy,
    Interruption,
    TurnOwner,
    endpoint_hint,
    transcript_quality,
)


class FloorPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = FloorPolicy()

    def event(self, text="", duration=700, confidence=0.9, probability=0.9):
        return Interruption(text, duration, confidence, probability)

    def test_brief_overlap_ducks_before_semantic_commit(self):
        self.assertEqual(
            self.policy.decide(self.event(duration=250)),
            FloorDecision.DUCK,
        )

    def test_backchannel_does_not_steal_floor(self):
        self.assertEqual(
            self.policy.decide(self.event("yeah")),
            FloorDecision.CONTINUE,
        )

    def test_question_yields_normal_floor(self):
        self.assertEqual(
            self.policy.decide(self.event("wait, what does that mean?")),
            FloorDecision.YIELD,
        )

    def test_important_thought_holds_on_non_explicit_overlap(self):
        self.assertEqual(
            self.policy.decide(
                self.event("I was also thinking we could add caching"),
                agent_priority=AgentPriority.IMPORTANT,
            ),
            FloorDecision.ACKNOWLEDGE_AND_HOLD,
        )

    def test_explicit_stop_always_yields_except_critical(self):
        interruption = self.event("stop")
        self.assertEqual(self.policy.decide(interruption), FloorDecision.YIELD)
        self.assertEqual(
            self.policy.decide(interruption, agent_priority=AgentPriority.CRITICAL),
            FloorDecision.ACKNOWLEDGE_AND_HOLD,
        )

    def test_probable_echo_is_ignored(self):
        self.assertEqual(
            self.policy.decide(self.event("stop", probability=0.2)),
            FloorDecision.CONTINUE,
        )

    def test_endpoint_waits_longer_for_thinking_pause(self):
        endpoint = AdaptiveEndpoint()
        complete = endpoint.silence_needed_ms(
            syntactically_complete=True, thinking_words=False
        )
        thinking = endpoint.silence_needed_ms(
            syntactically_complete=False, thinking_words=True
        )
        self.assertGreater(thinking, complete)

    def test_partial_endpoint_keeps_incomplete_thought_open(self):
        for text in (
            "we need to",
            "we need to.",
            "the reason is because",
            "I want to",
            "and",
            "so that",
            "the system is",
            "the system is...",
            "we really need",
            "for both the transcription",
            "as well as",
            "it needs to have",
            "with a...",
            "the medium ver-",
            "it should integrate with what we",
        ):
            with self.subTest(text=text):
                self.assertFalse(endpoint_hint(text).complete)
                self.assertTrue(endpoint_hint(text).defer)

    def test_partial_endpoint_can_close_complete_thought(self):
        self.assertTrue(endpoint_hint("make the main pane larger").complete)
        self.assertTrue(endpoint_hint("does that make sense?").complete)

    def test_explicit_send_forces_commit(self):
        hint = endpoint_hint("okay send it")
        self.assertTrue(hint.complete)
        self.assertTrue(hint.force)

    def test_repetitive_asr_hallucinations_never_commit(self):
        for text in (
            "JAX, JAX, JAX, JAX, JAX",
            "WebSocket JAX JAX JAX JAX JAX",
            "cough cough",
        ):
            with self.subTest(text=text):
                self.assertFalse(transcript_quality(text).accepted)

    def test_short_real_turns_still_commit(self):
        for text in ("Hello?", "keep going", "JAX", "See right there."):
            with self.subTest(text=text):
                self.assertTrue(transcript_quality(text).accepted)

    def test_two_repeated_technical_hotwords_fail_closed(self):
        quality = transcript_quality(
            "JAX, JAX",
            hotwords=("JAX", "WebSocket"),
        )
        self.assertFalse(quality.accepted)
        self.assertEqual(quality.reason, "repeated-hotword")
        self.assertTrue(
            transcript_quality("no no", hotwords=("JAX", "WebSocket")).accepted
        )

    def test_turn_owner_aggregates_fragments_and_commits_exactly_once(self):
        owner = TurnOwner(settle_seconds=0.4, incomplete_seconds=1.5)
        self.assertEqual(
            owner.observe("I'm looking at", now=1.0).action,
            "hold",
        )
        self.assertEqual(
            owner.observe("pane one with a...", now=1.3).action,
            "hold",
        )
        self.assertEqual(
            owner.observe("tmux widget called voice.", now=1.6).action,
            "hold",
        )
        self.assertEqual(owner.due(now=1.99).action, "hold")
        committed = owner.due(now=2.0)
        self.assertEqual(committed.action, "submit")
        self.assertEqual(
            committed.text,
            "I'm looking at pane one with a... tmux widget called voice.",
        )
        self.assertEqual(owner.due(now=3.0).action, "idle")

    def test_turn_owner_force_commits_pending_fragments(self):
        owner = TurnOwner()
        owner.observe("we should update the dashboard", now=2.0)
        committed = owner.observe("send it", now=2.1)
        self.assertEqual(committed.action, "submit")
        self.assertEqual(
            committed.text,
            "we should update the dashboard send it",
        )
        self.assertEqual(committed.reason, "forced")

    def test_turn_owner_questions_use_short_but_nonzero_deadline(self):
        owner = TurnOwner(question_seconds=0.2)
        held = owner.observe("Can you hear me?", now=5.0)
        self.assertEqual(held.action, "hold")
        self.assertEqual(owner.due(now=5.19).action, "hold")
        self.assertEqual(owner.due(now=5.20).text, "Can you hear me?")

    def test_turn_owner_rejects_cough_without_resetting_pending_deadline(self):
        owner = TurnOwner(settle_seconds=0.4)
        owner.observe("make the pane larger.", now=7.0)
        rejected = owner.observe("cough cough", now=7.2)
        self.assertEqual(rejected.action, "reject")
        self.assertEqual(rejected.reason, "non-lexical")
        self.assertEqual(owner.due(now=7.39).action, "hold")
        self.assertEqual(owner.due(now=7.40).text, "make the pane larger.")

    def test_turn_owner_cancel_discards_uncommitted_text(self):
        owner = TurnOwner()
        owner.observe("do not leak this", now=9.0)
        cancelled = owner.cancel()
        self.assertEqual(cancelled.action, "cancel")
        self.assertEqual(cancelled.text, "do not leak this")
        self.assertEqual(owner.due(now=99.0).action, "idle")


if __name__ == "__main__":
    unittest.main()
