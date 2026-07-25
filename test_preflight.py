import unittest
from pathlib import Path
from unittest.mock import patch

from preflight import matching_device, parse_pulse_sinks, preflight


DEVICES = [
    {"name": "Blue Snowball Mono", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Topping DX5", "max_input_channels": 0, "max_output_channels": 2},
]


class PreflightTests(unittest.TestCase):
    @patch("preflight.tts_health", return_value={"ok": True})
    @patch(
        "preflight.asr_health",
        return_value={"ok": True},
    )
    @patch("preflight.subprocess.run")
    def test_muted_candidate_defers_input_discovery(
        self,
        run,
        _asr,
        _tts,
    ):
        run.return_value.returncode = 0
        run.return_value.stdout = "1\n"
        report = preflight(
            whisper_python="python",
            kokoro_url="http://kokoro",
            ollama_url=None,
            live_model=None,
            input_device="owned production microphone",
            output_device=None,
            workspace_routing=False,
            routes=Path("/unused"),
            require_input=False,
        )
        self.assertTrue(report["checks"]["input"]["ok"])
        self.assertTrue(report["checks"]["input"]["deferred"])

    def test_device_matching_is_directional_and_case_insensitive(self):
        self.assertEqual(
            matching_device(DEVICES, "snowBALL", "input")["name"],
            "Blue Snowball Mono",
        )
        self.assertIsNone(matching_device(DEVICES, "Topping", "input"))
        self.assertEqual(
            matching_device(DEVICES, "topping", "output")["name"],
            "Topping DX5",
        )

    def test_missing_device_fails_without_guessing(self):
        self.assertIsNone(matching_device(DEVICES, "nonexistent", "input"))

    def test_pipewire_sink_parser_uses_stable_node_name(self):
        output = (
            "81\talsa_output.usb-Topping_DX5-00.HiFi__Headphones__sink"
            "\tPipeWire\ts32le 2ch 48000Hz\tIDLE\n"
        )
        self.assertEqual(
            parse_pulse_sinks(output),
            ["alsa_output.usb-Topping_DX5-00.HiFi__Headphones__sink"],
        )


if __name__ == "__main__":
    unittest.main()
