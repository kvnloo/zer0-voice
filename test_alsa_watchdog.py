import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alsa_watchdog import (
    Cursor,
    assess_progress,
    card_number,
    main,
    parse_cursor,
    pcm_path,
)


STATUS = """\
state: RUNNING
owner_pid   : 42
trigger_time: 1.0
-----
hw_ptr      : 200
appl_ptr    : 190
"""


class AlsaWatchdogTests(unittest.TestCase):
    def test_status_parser_ignores_unrelated_fields(self):
        self.assertEqual(parse_cursor(STATUS), Cursor("RUNNING", 42, 200, 190))
        self.assertIsNone(parse_cursor("state: RUNNING\n"))

    def test_card_lookup_is_exact(self):
        cards = (
            " 2 [Snow         ]: USB-Audio - Other\n"
            " 3 [Snowball     ]: USB-Audio - Blue Snowball\n"
        )
        self.assertEqual(card_number(cards, "Snowball"), 3)
        self.assertIsNone(card_number(cards, "snowball"))

    def test_pcm_path_tracks_dynamic_card_number(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = root / "cards"
            cards.write_text(" 7 [Snowball ]: USB-Audio - Blue\n")
            status = root / "proc/card7/pcm0c/sub0/status"
            status.parent.mkdir(parents=True)
            status.write_text(STATUS)
            self.assertEqual(pcm_path(cards, root / "proc", "Snowball"), status)

    def test_both_hardware_and_consumer_must_advance(self):
        before = Cursor("RUNNING", 42, 100, 90)
        self.assertEqual(
            assess_progress(before, Cursor("RUNNING", 42, 200, 190), 42),
            (True, "progressing"),
        )
        self.assertEqual(
            assess_progress(before, Cursor("RUNNING", 42, 100, 190), 42),
            (False, "alsa-hardware-stalled"),
        )
        self.assertEqual(
            assess_progress(before, Cursor("RUNNING", 42, 200, 90), 42),
            (False, "alsa-consumer-stalled"),
        )

    def test_wrong_owner_and_closed_pcm_fail(self):
        before = Cursor("RUNNING", 42, 100, 90)
        self.assertEqual(
            assess_progress(before, Cursor("RUNNING", 7, 200, 190), 42),
            (False, "alsa-owner-mismatch"),
        )
        self.assertEqual(
            assess_progress(before, Cursor("CLOSED", 42, 200, 190), 42),
            (False, "alsa-state:closed"),
        )
        self.assertEqual(
            assess_progress(None, None, 42),
            (False, "alsa-cursor-missing"),
        )

    def test_repeated_stall_terminates_only_recorded_worker(self):
        stalled = Cursor("RUNNING", 42, 100, 90)
        with patch(
            "sys.argv",
            [
                "alsa-watchdog",
                "--health",
                "/tmp/health",
                "--control-socket",
                "/tmp/control",
                "--startup-grace",
                "0",
                "--failures",
                "2",
            ],
        ), patch(
            "alsa_watchdog.read_snapshot",
            return_value={"pid": 42},
        ), patch(
            "alsa_watchdog.alive",
            return_value=True,
        ), patch(
            "alsa_watchdog.capture_active",
            return_value=True,
        ), patch(
            "alsa_watchdog.pcm_path",
            return_value=Path("/proc/fake"),
        ), patch(
            "alsa_watchdog.read_cursor",
            return_value=stalled,
        ), patch(
            "alsa_watchdog.time.sleep",
        ), patch(
            "alsa_watchdog.time.monotonic",
            side_effect=range(100),
        ), patch(
            "alsa_watchdog.terminate",
        ) as terminate:
            self.assertEqual(main(), 2)
        terminate.assert_called_once_with(42)


if __name__ == "__main__":
    unittest.main()
