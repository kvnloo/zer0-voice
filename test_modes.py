import unittest

from modes import MicMode, NotificationMode, Severity, VoiceModes


class VoiceModesTests(unittest.TestCase):
    def test_continuous_captures_without_a_key(self):
        modes = VoiceModes(mic=MicMode.CONTINUOUS)
        self.assertTrue(modes.microphone_open)
        self.assertTrue(modes.should_capture())

    def test_push_to_talk_captures_only_while_held(self):
        modes = VoiceModes(mic=MicMode.PUSH_TO_TALK)
        self.assertFalse(modes.microphone_open)
        self.assertFalse(modes.should_capture(push_held=False))
        self.assertTrue(modes.should_capture(push_held=True))

    def test_muted_never_captures(self):
        modes = VoiceModes(mic=MicMode.MUTED)
        self.assertFalse(modes.should_capture())
        self.assertFalse(modes.should_capture(push_held=True))

    def test_notification_modes_filter_only_proactive_speech(self):
        conversational = VoiceModes(
            notifications=NotificationMode.CONVERSATIONAL
        )
        updates = VoiceModes(notifications=NotificationMode.UPDATES)
        critical = VoiceModes(notifications=NotificationMode.CRITICAL)

        self.assertTrue(conversational.should_speak(Severity.DETAIL))
        self.assertFalse(updates.should_speak(Severity.DETAIL))
        self.assertTrue(updates.should_speak(Severity.UPDATE))
        self.assertFalse(critical.should_speak(Severity.UPDATE))
        self.assertTrue(critical.should_speak(Severity.CRITICAL))

    def test_direct_replies_are_never_suppressed(self):
        modes = VoiceModes(notifications=NotificationMode.CRITICAL)
        self.assertTrue(
            modes.should_speak(Severity.DETAIL, direct_reply=True)
        )


if __name__ == "__main__":
    unittest.main()
