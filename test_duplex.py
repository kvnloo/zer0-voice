import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from conversation import ListenConfig
from duplex import (
    AudioEvent,
    DuplexTurn,
    SentenceChunker,
    Speaker,
    UtteranceDetector,
    adaptive_threshold,
    append_metric,
    next_final,
    publish_transcript,
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
        started = detector.push(loud)
        self.assertEqual(
            [event.kind for event in started],
            ["speech.started", "speech.confirmed"],
        )
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

    def test_detector_confirms_only_after_sustained_voice(self):
        config = ListenConfig(
            block_ms=10,
            threshold=0.1,
            start_blocks=2,
            silence_ms=20,
            pre_roll_ms=10,
        )
        detector = UtteranceDetector(config, min_speech_ms=60)
        loud = np.full(config.block_size, 0.5, dtype=np.float32)
        self.assertEqual(detector.push(loud), [])
        self.assertEqual(
            [event.kind for event in detector.push(loud)],
            ["speech.started"],
        )
        for _ in range(3):
            self.assertEqual(detector.push(loud), [])
        self.assertEqual(
            [event.kind for event in detector.push(loud)],
            ["speech.confirmed"],
        )

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

    @patch("duplex.urllib.request.urlopen")
    def test_transcript_event_preserves_text_and_thread(self, urlopen):
        urlopen.return_value.__enter__.return_value.status = 202
        publish_transcript("http://relay/v1/events", "thread-7", "create a task", 99)
        request = urlopen.call_args.args[0]
        event = json.loads(request.data)
        self.assertEqual(event["kind"], "voice.transcript.final")
        self.assertEqual(event["subject"], "voice:thread-7:99")
        self.assertEqual(event["payload"], {
            "text": "create a task",
            "thread": "thread-7",
        })

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

    @patch("duplex.urllib.request.urlopen")
    @patch("duplex.subprocess.Popen")
    def test_pipewire_tts_streams_raw_pcm_before_response_finishes(
        self,
        popen,
        urlopen,
    ):
        class Sink:
            def __init__(self):
                self.chunks = []

            def write(self, chunk):
                self.chunks.append(chunk)

            def flush(self):
                pass

            def close(self):
                pass

        class Response:
            def __init__(self):
                self.chunks = [b"first", b"second", b""]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self, _size):
                return self.chunks.pop(0)

        process = popen.return_value
        process.stdin = Sink()
        process.stderr.read.return_value = b""
        process.returncode = 0
        process.wait.return_value = 0
        urlopen.return_value = Response()
        speaker = Speaker("http://kokoro", "af_heart", output="room.sink")
        speaker._pipewire_stream("Hello.", 0.0)
        self.assertEqual(process.stdin.chunks, [b"first", b"second"])
        self.assertIsNotNone(speaker.first_synthesis_seconds)
        self.assertIsNotNone(speaker.first_play_at)
        command = popen.call_args.args[0]
        self.assertEqual(command[:8], [
            "pw-play",
            "--raw",
            "--format",
            "s16",
            "--rate",
            "24000",
            "--channels",
            "1",
        ])
        self.assertIn("room.sink", command)


class AsyncDuplexTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_turn_is_the_only_source_of_spoken_text(self):
        calls = []

        class Server:
            async def stream_turn(self, thread, text, *, effort):
                calls.append((thread, text, effort))
                yield SimpleNamespace(
                    kind="assistant.delta",
                    payload={"text": "Exact harness response."},
                    subject="turn:turn-1",
                )
                yield SimpleNamespace(
                    kind="assistant.completed",
                    payload={},
                    subject="turn:turn-1",
                )

        class CapturingSpeaker:
            def __init__(self):
                self.spoken = []

            async def say(self, text):
                self.spoken.append(text)

            def interrupt(self):
                raise AssertionError("turn should not be interrupted")

        speaker = CapturingSpeaker()
        turn = DuplexTurn(Server(), "current-harness", speaker)
        response = await turn.run("Exact spoken transcript.", "high")
        self.assertEqual(
            calls,
            [("current-harness", "Exact spoken transcript.", "high")],
        )
        self.assertEqual(response, "Exact harness response.")
        self.assertEqual(speaker.spoken, ["Exact harness response."])

    async def test_barge_in_interrupts_active_turn_before_final(self):
        class Mic:
            def __init__(self):
                self.events = [
                    AudioEvent("speech.started"),
                    AudioEvent("speech.confirmed"),
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
        audio = await next_final(Mic(), turn, interrupt_mode="immediate")
        self.assertTrue(turn.interrupted.is_set())
        self.assertEqual(audio.size, 16)

    async def test_sustained_barge_in_waits_for_confirmation(self):
        class Mic:
            def __init__(self):
                self.events = [
                    AudioEvent("speech.started"),
                    AudioEvent("speech.confirmed"),
                    AudioEvent("speech.final", np.ones(16, dtype=np.float32)),
                ]
                self.reads = 0

            async def next(self):
                self.reads += 1
                return self.events.pop(0)

        class Turn:
            def __init__(self, mic):
                self.mic = mic
                self.interrupted_at = None

            async def interrupt(self):
                self.interrupted_at = self.mic.reads

        mic = Mic()
        turn = Turn(mic)
        audio = await next_final(mic, turn, interrupt_mode="sustained")
        self.assertEqual(turn.interrupted_at, 2)
        self.assertEqual(audio.size, 16)

    async def test_default_barge_in_waits_for_final_to_avoid_echo_cutoff(self):
        class Mic:
            def __init__(self):
                self.events = [
                    AudioEvent("speech.started"),
                    AudioEvent("speech.confirmed"),
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
