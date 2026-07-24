import json
import unittest
from unittest.mock import AsyncMock, patch

from control import main


class VoiceControlCLITests(unittest.TestCase):
    def test_status_has_no_reload_only_attribute_dependency(self):
        with patch(
            "control.request",
            new=AsyncMock(return_value={"ok": True}),
        ) as request_mock, patch(
            "control.sys.argv",
            ["voice/control", "status"],
        ), patch("builtins.print"):
            self.assertEqual(main(), 0)
            self.assertEqual(request_mock.await_args.args[1], {})

    def test_every_non_reload_command_maps_to_one_typed_payload(self):
        cases = (
            (["mic", "muted"], {"mic": "muted"}),
            (["notify", "critical"], {"notifications": "critical"}),
            (["press"], {"push_held": True}),
            (["release"], {"push_held": False}),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv), patch(
                "control.request",
                new=AsyncMock(return_value={"ok": True}),
            ) as request_mock, patch(
                "control.sys.argv",
                ["voice/control", *argv],
            ), patch("builtins.print"):
                self.assertEqual(main(), 0)
                self.assertEqual(request_mock.await_args.args[1], expected)

    def test_reload_requires_arguments(self):
        with patch("control.request", new=AsyncMock()) as _request, patch(
            "control.sys.argv", ["voice/control", "reload"]
        ), self.assertRaises(SystemExit):
            main()

    def test_reload_invokes_request_with_model_and_effort(self):
        observed = {}

        async def fake_request(_path, payload):
            observed.update(payload)
            return {"ok": True, **payload}

        with patch("control.request", new=AsyncMock(side_effect=fake_request)), patch(
            "control.sys.argv",
            [
                "voice/control",
                "reload",
                "--live-model",
                "gpt-5.6-luna",
                "--live-effort",
                "critical",
            ],
        ), patch("builtins.print") as printer:
            self.assertEqual(main(), 0)
            self.assertEqual(
                observed,
                {"live_model": "gpt-5.6-luna", "live_effort": "critical"},
            )
            self.assertTrue(printer.called)


if __name__ == "__main__":
    unittest.main()
