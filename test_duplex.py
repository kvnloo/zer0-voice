import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from conversation import ListenConfig
from duplex import (
    AudioEvent,
    SentenceChunker,
    Speaker,
    UtteranceDetector,
    adaptive_threshold,
    append_metric,
    next_final,
    voice_control,
)


class DuplexTests(unittest.TestCase):
    def test_detector_reports_start_before_final_for_barge_in(self):
        config = ListenConfig(
            block_ms=10,
            threshold=0.1,
            start_blocks=2,
            silence_ms=20,
            pre_roll_ms=10,
        )
        detector = UtteranceDetector(config, min_speech_ms=20)
        loud = np.full(config.block_size, 0.5, dtype=np.float32)
        quiet = np.zeros(config.block_size, dtype=np.float32)
        self.assertEqual(detector.push(loud), [])
        self.assertEqual(detector.push(loud)[0].kind, "speech.started")
        self.assertEqual(detector.push(quiet), [])
        final = detector.push(quiet)
        self.assertEqual(final[0].kind, "speech.final")
        self.assertGreater(final[0].audio.size, 0)

    def test_detector_discards_short_transient_after_speech_start(self):
        config = ListenConfig(
            block_ms=10,
            threshold=0.1,
            start_blocks=2,
            silence_ms=20,
            pre_roll_ms=10,
        )
        detector = UtteranceDetector(config, min_speech_ms=60)
        loud = np.full(config.block_size, 0.5, dtype=np.float32)
        quiet = np.zeros(config.block_size, dtype=np.float32)
        detector.push(loud)
        self.assertEqual(detector.push(loud)[0].kind, "speech.started")
        self.assertEqual(detector.push(quiet), [])
        self.assertEqual(detector.push(quiet), [])

    def test_sentence_chunker_speaks_complete_thoughts_early(self):
        chunker = SentenceChunker()
        self.assertEqual(chunker.feed("First thought. Sec"), ["First thought."])
        self.assertEqual(chunker.feed("ond thought? Tail"), ["Second thought?"])
        self.assertEqual(chunker.flush(), "Tail")

    def test_adaptive_threshold_tracks_noise_with_safe_bounds(self):
        self.assertEqual(adaptive_threshold(np.array([])), 0.018)
        self.assertEqual(adaptive_threshold(np.full(10, 0.0001)), 0.004)
        self.assertAlmostEqual(adaptive_threshold(np.full(10, 0.003)), 0.0075)
        self.assertEqual(adaptive_threshold(np.full(10, 0.2)), 0.02)
        transient = np.array([0.003] * 9 + [0.4])
        self.assertAlmostEqual(adaptive_threshold(transient), 0.0075)

    def test_metric_ledger_is_append_only_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.jsonl"
            append_metric(path, {"schema": 1, "route": "pm", "total_seconds": 1.5})
            append_metric(path, {"schema": 1, "route": "zerOS", "total_seconds": 2.0})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["route"] for row in rows], ["pm", "zerOS"])

    def test_spoken_stop_is_exact_and_does_not_capture_task_language(self):
        for phrase in (
            "stop",
            "stop please",
            "can you stop please",
            "please stop voice mode",
            "quit listening",
        ):
            self.assertEqual(voice_control(phrase), "stop")
        for phrase in (
            "stop the server",
            "can you stop ollama",
            "how do I stop voice mode",
            "do not stop",
        ):
            self.assertIsNone(voice_control(phrase))

    @patch("duplex.subprocess.Popen")
    def test_pipewire_player_streams_wav_to_explicit_sink(self, popen):
        process = popen.return_value
        process.communicate.return_value = (b"", b"")
        process.returncode = 0
        speaker = Speaker(
            "http://kokoro",
            "af_heart",
            output="test.sink",
            latency="25ms",
        )
        speaker._pipewire_play(b"RIFFwav")
        popen.assert_called_once_with(
            [
                "pw-play",
                "--latency",
                "25ms",
                "--target",
                "test.sink",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        process.communicate.assert_called_once_with(b"RIFFwav")


class AsyncDuplexTests(unittest.IsolatedAsyncioTestCase):
    async def test_barge_in_interrupts_active_turn_before_final(self):
        class Mic:
            def __init__(self):
                self.events = [
                    AudioEvent("speech.started"),
                    AudioEvent("speech.final", np.ones(16, dtype=np.float32)),
                ]

            async def next(self):
                return self.events.pop(0)

        class Turn:
            def __init__(self):
                self.interrupted = asyncio.Event()

            async def interrupt(self):
                self.interrupted.set()

        turn = Turn()
        audio = await next_final(Mic(), turn, interrupt_on_start=True)
        self.assertTrue(turn.interrupted.is_set())
        self.assertEqual(audio.size, 16)

    async def test_default_barge_in_waits_for_final_to_avoid_echo_cutoff(self):
        class Mic:
            def __init__(self):
                self.events = [
                    AudioEvent("speech.started"),
                    AudioEvent("speech.final", np.ones(16, dtype=np.float32)),
                ]

            async def next(self):
                return self.events.pop(0)

        class Turn:
            def __init__(self):
                self.interrupted = False

            async def interrupt(self):
                self.interrupted = True

        turn = Turn()
        audio = await next_final(Mic(), turn)
        self.assertFalse(turn.interrupted)
        self.assertEqual(audio.size, 16)


if __name__ == "__main__":
    unittest.main()
