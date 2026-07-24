import unittest
from unittest.mock import Mock, call, patch

from indicator import COLORS, NullIndicator, VoiceState, WootingIndicator, make_indicator


class IndicatorTests(unittest.TestCase):
    @patch("indicator.ctypes.CDLL")
    def test_wooting_indicator_sets_only_esc_and_resets_only_esc(self, load):
        sdk = load.return_value
        sdk.wooting_rgb_kbd_connected.return_value = True
        sdk.wooting_rgb_direct_set_key.return_value = True

        indicator = WootingIndicator("/tmp/libwooting-rgb-sdk.so")
        indicator.set(VoiceState.LISTENING)
        indicator.set(VoiceState.THINKING)
        indicator.set(VoiceState.SPEAKING)
        indicator.clear()
        indicator.close()
        indicator.close()

        self.assertEqual(
            sdk.wooting_rgb_direct_set_key.call_args_list,
            [
                call(0, 0, *COLORS[VoiceState.LISTENING]),
                call(0, 0, *COLORS[VoiceState.THINKING]),
                call(0, 0, *COLORS[VoiceState.SPEAKING]),
            ],
        )
        self.assertEqual(
            sdk.wooting_rgb_direct_reset_key.call_args_list,
            [call(0, 0), call(0, 0)],
        )
        sdk.wooting_rgb_array_set_full.assert_not_called()

    @patch("indicator.ctypes.CDLL")
    def test_explicit_wooting_fails_when_keyboard_is_unavailable(self, load):
        load.return_value.wooting_rgb_kbd_connected.return_value = False
        with self.assertRaisesRegex(RuntimeError, "hidraw permissions"):
            WootingIndicator("/tmp/libwooting-rgb-sdk.so")

    @patch("indicator.WootingIndicator", side_effect=RuntimeError("no access"))
    def test_auto_mode_warns_and_falls_back_without_blocking_voice(self, _indicator):
        warning = Mock()
        result = make_indicator("auto", warning=warning)
        self.assertIsInstance(result, NullIndicator)
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
