import unittest

from floor import (
    AdaptiveEndpoint,
    AgentPriority,
    FloorDecision,
    FloorPolicy,
    Interruption,
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


if __name__ == "__main__":
    unittest.main()
