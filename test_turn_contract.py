import unittest

from turn_contract import require_audible_reply


class TurnContractTests(unittest.TestCase):
    def test_nonempty_reply_with_playback_is_complete(self):
        require_audible_reply("Actual answer.", True)

    def test_empty_model_output_is_never_complete(self):
        with self.assertRaisesRegex(RuntimeError, "live-response-empty"):
            require_audible_reply(" \n", True)

    def test_unplayed_model_output_is_never_complete(self):
        with self.assertRaisesRegex(RuntimeError, "tts-playback-missing"):
            require_audible_reply("Unheard answer.", False)


if __name__ == "__main__":
    unittest.main()
