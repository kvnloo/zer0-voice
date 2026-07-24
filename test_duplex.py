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
from control_plane import VoiceControl
from duplex import (
    AudioEvent,
    ConsecutiveFailureBudget,
    DuplexTurn,
    apply_live_reload,
    LiveContextMirror,
    Microphone,
    SentenceChunker,
    Speaker,
    UtteranceDetector,
    adaptive_threshold,
    append_debug,
    append_metric,
    collect_owned_transcript,
    mark_capture_active,
    next_final,
    next_controlled_final,
    race_response_and_capture,
    schedule_pm_decision,
    voice_pm_wiring,
    voice_control,
)
from floor import SubmissionDecision, TurnOwner
from modes import MicMode, VoiceModes
from providers import StreamEvent
from wiring import VoicePMWiring


class DuplexTests(unittest.TestCase):
    def test_apply_live_reload_updates_live_model_and_effort(self):
        class MockContext:
            def __init__(self, model):
                self.model = model

        control = VoiceControl()
        context = MockContext("baseline")
        model, effort = apply_live_reload(
            live_context=context,
            live_model="baseline",
            live_effort="low",
            control_state=control.state,
        )
        self.assertEqual(model, "baseline")
        self.assertEqual(effort, "low")

        asyncio.run(
            control.apply(
                {"live_model": "model-2", "live_effort": "critical"},
            )
        )
        model, effort = apply_live_reload(
            live_context=context,
            live_model=model,
            live_effort=effort,
            control_state=control.state,
        )
        self.assertEqual(context.model, "model-2")
        self.assertEqual(model, "model-2")
        self.assertEqual(effort, "critical")

    def test_continued_capture_leaves_bounded_transcribing_phase(self):
        class Health:
            def __init__(self):
                self.transitions = []

            def transition(self, phase, *, lane):
                self.transitions.append((phase, lane))

        health = Health()
        mark_capture_active(health)
        self.assertEqual(health.transitions, [("listening", "mic")])

    def test_capture_phase_marker_accepts_disabled_health(self):
        mark_capture_active(None)

    def test_live_lane_failure_budget_survives_transient_turn_errors(self):
        budget = ConsecutiveFailureBudget(limit=3)

        self.assertFalse(budget.failed())
        self.assertFalse(budget.failed())
        budget.recovered()
        self.assertFalse(budget.failed())
        self.assertFalse(budget.failed())
        self.assertTrue(budget.failed())

    def test_live_lane_failure_budget_rejects_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ConsecutiveFailureBudget(limit=0)

    def test_microphone_processes_blocks_off_callback_and_ignores_stale_generation(self):
        config = ListenConfig(threshold=0.01)
        mic = Microphone(config)
        mic.stream = object()
        mic.stream_generation = 2
        block = np.ones(config.block_size, dtype=np.float32)

        mic._process_block(block, "input_overflow", 1)
        self.assertTrue(mic.queue.empty())

        mic._process_block(block, "input_overflow", 2)
        warning = mic.queue.get_nowait()
        self.assertEqual(
            (warning.kind, warning.detail),
            ("audio.warning", "input_overflow"),
        )

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

    def test_detector_finalizes_held_push_to_talk_audio_on_release(self):
        config = ListenConfig(
            block_ms=10,
            threshold=0.1,
            start_blocks=2,
            silence_ms=500,
            pre_roll_ms=10,
        )
        detector = UtteranceDetector(config, min_speech_ms=20)
        loud = np.full(config.block_size, 0.5, dtype=np.float32)
        detector.push(loud)
        detector.push(loud)
        final = detector.finish()
        self.assertIsNotNone(final)
        self.assertEqual(final.kind, "speech.final")
        self.assertGreater(final.audio.size, 0)

    def test_semantic_endpoint_keeps_incomplete_partial_open(self):
        config = ListenConfig(
            block_ms=10,
            threshold=0.1,
            start_blocks=2,
            silence_ms=300,
            pre_roll_ms=10,
        )
        detector = UtteranceDetector(config, min_speech_ms=20)
        detector.update_transcript("I want to")
        self.assertGreater(detector.silence_needed_blocks(), config.silence_blocks)

    def test_semantic_endpoint_commits_complete_partial_faster(self):
        config = ListenConfig(
            block_ms=10,
            threshold=0.1,
            start_blocks=2,
            silence_ms=850,
            pre_roll_ms=10,
        )
        detector = UtteranceDetector(config, min_speech_ms=20)
        detector.update_transcript("make the main pane larger")
        self.assertLess(detector.silence_needed_blocks(), config.silence_blocks)

    def test_late_preview_cannot_retime_the_next_utterance(self):
        config = ListenConfig(
            block_ms=10,
            threshold=0.1,
            start_blocks=2,
            silence_ms=850,
            pre_roll_ms=10,
        )
        detector = UtteranceDetector(config, min_speech_ms=20)
        stale_generation = detector.generation
        detector.reset()

        self.assertFalse(
            detector.update_transcript("make the pane larger", stale_generation)
        )
        self.assertEqual(
            detector.silence_needed_blocks(),
            config.silence_blocks,
        )
        self.assertTrue(
            detector.update_transcript("make the pane larger", detector.generation)
        )
        self.assertLess(
            detector.silence_needed_blocks(),
            config.silence_blocks,
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

    def test_debug_ledger_records_live_transcript_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug.jsonl"
            append_debug(path, "speech.confirmed")
            append_debug(path, "asr.partial", text="hello world")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[0]["kind"], "speech.confirmed")
        self.assertEqual(rows[1]["text"], "hello world")

    def test_duplex_turn_publishes_cumulative_reply_deltas(self):
        class Provider:
            async def stream(self, _text, *, effort):
                self.effort = effort
                yield StreamEvent("delta", "1", "Hello ")
                yield StreamEvent("delta", "1", "there.")

            async def interrupt(self, _turn_id):
                return None

        class Speaker:
            async def say(self, _text):
                return None

            def interrupt(self):
                return None

        partials = []
        turn = DuplexTurn(Provider(), Speaker(), on_delta=partials.append)
        self.assertEqual(asyncio.run(turn.run("hi", "low")), "Hello there.")
        self.assertEqual(partials, ["Hello ", "Hello there."])

    def test_pm_publisher_hot_selects_without_replacing_thread_boundary(self):
        cache = {}
        first = voice_pm_wiring(
            cache,
            endpoint="http://relay/v1/voice/committed",
            lane="legacy",
            thread="thread-7",
            run_id="run-1",
            lifecycle=None,
        )
        second = voice_pm_wiring(
            cache,
            endpoint="http://relay/v1/voice/committed",
            lane="candidate",
            thread="thread-7",
            run_id="run-1",
            lifecycle=None,
        )
        other = voice_pm_wiring(
            cache,
            endpoint="http://relay/v1/voice/committed",
            lane="candidate",
            thread="thread-8",
            run_id="run-1",
            lifecycle=None,
        )
        self.assertIs(first, second)
        self.assertEqual(second.active, "candidate")
        self.assertIsNot(first, other)
        self.assertEqual(len(cache), 2)

    def test_production_duplex_commit_schedules_typed_pm_delivery(self):
        delivered = []

        async def candidate(turn):
            delivered.append(turn)
            return True

        wiring = VoicePMWiring(
            thread="thread-live",
            conversation="thread-live",
            run_id="voice-run-live",
            candidate=candidate,
            active="candidate",
            clock_ns=lambda: 99,
        )
        decision = SubmissionDecision(
            "submit",
            "Create the native PM issue.",
            "deadline",
        )

        async def exercise():
            with patch("duplex.voice_pm_wiring", return_value=wiring) as select:
                selected, lane = schedule_pm_decision(
                    {},
                    endpoint="http://127.0.0.1:8787/v1/voice/committed",
                    lane="candidate",
                    thread="thread-live",
                    run_id="voice-run-live",
                    lifecycle=None,
                    decision=decision,
                )
                self.assertIs(selected, wiring)
                self.assertEqual(lane, "candidate")
                select.assert_called_once_with(
                    {},
                    endpoint="http://127.0.0.1:8787/v1/voice/committed",
                    lane="candidate",
                    thread="thread-live",
                    run_id="voice-run-live",
                    lifecycle=None,
                )
            await wiring.drain()

        asyncio.run(exercise())
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].source_id, "voice-run-live-1")
        self.assertEqual(delivered[0].thread, "thread-live")
        self.assertEqual(delivered[0].text, "Create the native PM issue.")

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

    def test_pipewire_falls_back_to_local_playback_on_runtime_error(self):
        speaker = Speaker("http://kokoro", "af_heart")

        with patch.object(
            speaker, "_pipewire_stream", side_effect=RuntimeError("pipewire explode")
        ), patch.object(
            speaker,
            "_fallback_audio",
        ) as fallback:
            with patch("duplex.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args)):
                asyncio.run(speaker.say("backup"))
            fallback.assert_called_once()


class AsyncDuplexTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_transcript_keeps_audio_captured_during_asr(self):
        captured = asyncio.Queue()
        first_asr_started = asyncio.Event()
        release_first_asr = asyncio.Event()
        transcripts = []

        async def capture(on_event):
            audio = await captured.get()
            on_event(AudioEvent("speech.started"))
            on_event(AudioEvent("speech.final", audio))
            return audio

        async def transcribe_audio(audio):
            value = int(audio[0])
            transcripts.append(value)
            if value == 1:
                first_asr_started.set()
                await release_first_asr.wait()
                return "we need to"
            return "finish this send it"

        await captured.put(np.array([1], dtype=np.float32))
        task = asyncio.create_task(
            collect_owned_transcript(
                TurnOwner(),
                capture,
                transcribe_audio,
            )
        )
        await asyncio.wait_for(first_asr_started.wait(), timeout=0.1)
        # Fragment two endpoints while fragment one is still in ASR.
        await captured.put(np.array([2], dtype=np.float32))
        await asyncio.sleep(0)
        release_first_asr.set()
        owned = await asyncio.wait_for(task, timeout=0.1)
        self.assertEqual(transcripts, [1, 2])
        self.assertEqual(owned.decision.action, "submit")
        self.assertEqual(
            owned.decision.text,
            "we need to finish this send it",
        )
        owned.next_audio.cancel()
        await asyncio.gather(owned.next_audio, return_exceptions=True)

    async def test_owned_transcript_speech_start_beats_commit_deadline(self):
        second_started = asyncio.Event()
        finish_second = asyncio.Event()
        capture_count = 0

        async def capture(on_event):
            nonlocal capture_count
            capture_count += 1
            if capture_count == 1:
                on_event(AudioEvent("speech.started"))
                audio = np.array([1], dtype=np.float32)
                on_event(AudioEvent("speech.final", audio))
                return audio
            if capture_count == 2:
                on_event(AudioEvent("speech.started"))
                second_started.set()
                await finish_second.wait()
                audio = np.array([2], dtype=np.float32)
                on_event(AudioEvent("speech.final", audio))
                return audio
            await asyncio.Event().wait()

        async def transcribe_audio(audio):
            return "first thought." if int(audio[0]) == 1 else "second send it"

        task = asyncio.create_task(
            collect_owned_transcript(
                TurnOwner(settle_seconds=0.001),
                capture,
                transcribe_audio,
            )
        )
        await asyncio.wait_for(second_started.wait(), timeout=0.1)
        # No final endpoint exists yet, but active speech must suspend the
        # one-millisecond commit deadline.
        await asyncio.sleep(0.005)
        self.assertFalse(task.done())
        finish_second.set()
        owned = await asyncio.wait_for(task, timeout=0.1)
        self.assertEqual(
            owned.decision.text,
            "first thought. second send it",
        )
        owned.next_audio.cancel()
        await asyncio.gather(owned.next_audio, return_exceptions=True)

    async def test_owned_transcript_cancellation_closes_capture_and_state(self):
        capture_started = asyncio.Event()
        capture_closed = asyncio.Event()
        owner = TurnOwner()

        async def capture(_on_event):
            capture_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                capture_closed.set()

        async def transcribe_audio(_audio):
            return "we need to"

        task = asyncio.create_task(
            collect_owned_transcript(
                owner,
                capture,
                transcribe_audio,
                initial_audio=np.ones(1, dtype=np.float32),
            )
        )
        await asyncio.wait_for(capture_started.wait(), timeout=0.1)
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(capture_closed.is_set())
        self.assertEqual(owner.pending, "")
        self.assertIsNone(owner.deadline)

    async def test_live_context_reforks_authoritative_keyboard_history(self):
        class Server:
            def __init__(self):
                self.calls = []

            async def read_thread(self, _thread):
                return {
                    "turns": [
                        {"id": "done-1", "status": "completed"},
                        {"id": "active-2", "status": "inProgress"},
                    ]
                }

            async def fork_thread(self, thread, **kwargs):
                self.calls.append((thread, kwargs))
                return f"live-{len(self.calls)}"

        server = Server()
        mirror = LiveContextMirror(
            server,
            cwd=Path("/workspace/pm"),
            model=None,
            developer_instructions="speak briefly",
        )
        self.assertEqual(await mirror.refresh("harness"), "live-1")
        self.assertEqual(await mirror.refresh("harness"), "live-2")
        self.assertEqual(
            server.calls,
            [
                (
                    "harness",
                    {
                        "cwd": Path("/workspace/pm"),
                        "model": None,
                        "developer_instructions": "speak briefly",
                        "ephemeral": True,
                        "last_turn_id": "done-1",
                    },
                ),
                (
                    "harness",
                    {
                        "cwd": Path("/workspace/pm"),
                        "model": None,
                        "developer_instructions": "speak briefly",
                        "ephemeral": True,
                        "last_turn_id": "done-1",
                    },
                ),
            ],
        )

    async def test_completed_interruption_never_waits_for_stale_response(self):
        class Turn:
            def __init__(self):
                self.interrupted = 0

            async def interrupt(self):
                self.interrupted += 1

        turn = Turn()
        response = asyncio.create_task(asyncio.Event().wait())
        expected = np.ones(32, dtype=np.float32)

        async def captured():
            return expected

        listen = asyncio.create_task(captured())
        reply, audio = await asyncio.wait_for(
            race_response_and_capture(
                response,
                listen,
                turn,
                preempt_on_completed_capture=True,
            ),
            timeout=0.1,
        )
        self.assertIsNone(reply)
        np.testing.assert_array_equal(audio, expected)
        self.assertEqual(turn.interrupted, 1)
        self.assertTrue(response.cancelled())

    async def test_safe_capture_buffers_next_utterance_until_reply_finishes(self):
        class Turn:
            async def interrupt(self):
                raise AssertionError("safe capture must not interrupt the reply")

        release = asyncio.Event()
        expected = np.ones(32, dtype=np.float32)

        async def respond():
            await release.wait()
            return "complete audible reply"

        async def captured():
            return expected

        response = asyncio.create_task(respond())
        listen = asyncio.create_task(captured())
        race = asyncio.create_task(
            race_response_and_capture(response, listen, Turn())
        )
        await asyncio.sleep(0)
        self.assertFalse(race.done())
        release.set()
        reply, audio = await race
        self.assertEqual(reply, "complete audible reply")
        np.testing.assert_array_equal(audio, expected)

    async def test_completed_response_cancels_unused_listener(self):
        class Turn:
            async def interrupt(self):
                raise AssertionError("completed response must not be interrupted")

        async def captured():
            await asyncio.Event().wait()

        response = asyncio.create_task(asyncio.sleep(0, result="ready"))
        listen = asyncio.create_task(captured())
        reply, audio = await race_response_and_capture(response, listen, Turn())
        self.assertEqual(reply, "ready")
        self.assertIsNone(audio)
        self.assertTrue(listen.cancelled())

    async def test_continuation_cannot_cancel_reply_before_audio_starts(self):
        class Speaker:
            first_play_at = None

        class Turn:
            speaker = Speaker()

            async def interrupt(self):
                raise AssertionError("protected reply must not be interrupted")

        expected = np.ones(32, dtype=np.float32)

        async def captured():
            return expected

        response = asyncio.create_task(asyncio.sleep(0.01, result="audible reply"))
        listen = asyncio.create_task(captured())
        reply, audio = await race_response_and_capture(
            response,
            listen,
            Turn(),
            unspoken_reply_grace=0.1,
        )
        self.assertEqual(reply, "audible reply")
        np.testing.assert_array_equal(audio, expected)

    async def test_completed_utterance_interrupts_playback_that_already_started(self):
        class Speaker:
            first_play_at = 1.0

        class Turn:
            speaker = Speaker()

            def __init__(self):
                self.interrupted = 0

            async def interrupt(self):
                self.interrupted += 1

        turn = Turn()
        response = asyncio.create_task(asyncio.Event().wait())
        expected = np.ones(32, dtype=np.float32)

        async def captured():
            return expected

        listen = asyncio.create_task(captured())
        reply, audio = await race_response_and_capture(
            response,
            listen,
            turn,
            preempt_on_completed_capture=True,
        )
        self.assertIsNone(reply)
        np.testing.assert_array_equal(audio, expected)
        self.assertEqual(turn.interrupted, 1)

    async def test_cancelled_controlled_listener_cannot_consume_next_final(self):
        class Mic:
            def __init__(self):
                self.queue = asyncio.Queue()

            def start(self):
                pass

            def close(self):
                pass

            def pause(self, *, finalize=False):
                return None

            async def next(self):
                return await self.queue.get()

        class Indicator:
            def set(self, _state):
                pass

            def clear(self):
                pass

        mic = Mic()
        control = VoiceControl()
        abandoned = asyncio.create_task(
            next_controlled_final(mic, control, Indicator())
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        abandoned.cancel()
        await asyncio.gather(abandoned, return_exceptions=True)

        expected = np.ones(8, dtype=np.float32)
        await mic.queue.put(AudioEvent("speech.final", expected))
        replacement = asyncio.create_task(
            next_controlled_final(mic, control, Indicator())
        )
        actual = await asyncio.wait_for(replacement, timeout=0.1)
        np.testing.assert_array_equal(actual, expected)

    async def test_push_to_talk_opens_on_press_and_finalizes_on_release(self):
        class Mic:
            def __init__(self):
                self.started = 0
                self.closed = 0
                self.paused = []
                self.waiting = asyncio.Event()

            def start(self):
                self.started += 1

            def close(self):
                self.closed += 1

            def pause(self, *, finalize=False):
                self.paused.append(finalize)
                if finalize:
                    return AudioEvent(
                        "speech.final",
                        np.ones(16, dtype=np.float32),
                    )
                return None

            async def next(self):
                await self.waiting.wait()

        class Indicator:
            def __init__(self):
                self.states = []
                self.clears = 0

            def set(self, state):
                self.states.append(state)

            def clear(self):
                self.clears += 1

        control = VoiceControl(VoiceModes(mic=MicMode.PUSH_TO_TALK))
        mic, indicator = Mic(), Indicator()
        capture = asyncio.create_task(
            next_controlled_final(mic, control, indicator)
        )
        await asyncio.sleep(0)
        self.assertEqual(mic.started, 0)
        await control.apply({"push_held": True})
        await asyncio.sleep(0)
        self.assertEqual(mic.started, 1)
        await control.apply({"push_held": False})
        audio = await capture
        self.assertEqual(audio.size, 16)
        self.assertEqual(mic.paused, [True])
        self.assertGreaterEqual(indicator.clears, 1)

    async def test_muted_closes_capture_and_continuous_reopens_it(self):
        class Mic:
            def __init__(self):
                self.started = 0
                self.closed = 0
                self.queue = asyncio.Queue()

            def start(self):
                self.started += 1

            def close(self):
                self.closed += 1

            def pause(self, *, finalize=False):
                self.closed += 1
                return None

            async def next(self):
                return await self.queue.get()

        class Indicator:
            def __init__(self):
                self.clears = 0

            def set(self, _state):
                pass

            def clear(self):
                self.clears += 1

        control = VoiceControl()
        mic, indicator = Mic(), Indicator()
        capture = asyncio.create_task(
            next_controlled_final(mic, control, indicator)
        )
        await asyncio.sleep(0)
        self.assertEqual(mic.started, 1)
        await control.apply({"mic": "muted"})
        for _ in range(20):
            if mic.closed:
                break
            await asyncio.sleep(0)
        self.assertGreaterEqual(mic.closed, 1)
        self.assertGreaterEqual(indicator.clears, 1)
        await control.apply({"mic": "continuous"})
        for _ in range(20):
            if mic.started >= 2:
                break
            await asyncio.sleep(0)
        self.assertEqual(mic.started, 2)
        await mic.queue.put(
            AudioEvent("speech.final", np.ones(8, dtype=np.float32))
        )
        self.assertEqual((await capture).size, 8)

    async def test_codex_turn_is_the_only_source_of_spoken_text(self):
        calls = []

        class Provider:
            async def stream(self, text, *, context=(), effort=None):
                calls.append((text, context, effort))
                yield SimpleNamespace(
                    kind="delta",
                    text="Exact harness response.",
                    turn_id="turn-1",
                )
                yield SimpleNamespace(
                    kind="completed",
                    text="",
                    turn_id="turn-1",
                )

            async def interrupt(self, turn_id):
                raise AssertionError(f"turn {turn_id} should not be interrupted")

        class CapturingSpeaker:
            def __init__(self):
                self.spoken = []

            async def say(self, text):
                self.spoken.append(text)

            def interrupt(self):
                raise AssertionError("turn should not be interrupted")

        speaker = CapturingSpeaker()
        turn = DuplexTurn(Provider(), speaker)
        response = await turn.run("Exact spoken transcript.", "high")
        self.assertEqual(
            calls,
            [("Exact spoken transcript.", (), "high")],
        )
        self.assertEqual(response, "Exact harness response.")
        self.assertEqual(speaker.spoken, ["Exact harness response."])

    async def test_generation_failure_cancels_speech_without_leaking(self):
        speaking = asyncio.Event()

        class Provider:
            def __init__(self):
                self.interrupted = []

            async def stream(self, _text, *, context=(), effort=None):
                yield SimpleNamespace(
                    kind="delta",
                    text="Starting now. ",
                    turn_id="turn-generation-failed",
                )
                await speaking.wait()
                raise RuntimeError("generation failed")

            async def interrupt(self, turn_id):
                self.interrupted.append(turn_id)

        class BlockingSpeaker:
            def __init__(self):
                self.cancelled = False
                self.interruptions = 0

            async def say(self, _text):
                speaking.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

            def interrupt(self):
                self.interruptions += 1

        provider, speaker = Provider(), BlockingSpeaker()
        turn = DuplexTurn(provider, speaker)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await asyncio.wait_for(turn.run("hello", "low"), timeout=0.1)
        self.assertTrue(speaker.cancelled)
        self.assertEqual(speaker.interruptions, 1)
        self.assertEqual(provider.interrupted, ["turn-generation-failed"])

    async def test_speech_failure_cancels_generation_without_leaking(self):
        generator_closed = asyncio.Event()

        class Provider:
            def __init__(self):
                self.interrupted = []

            async def stream(self, _text, *, context=(), effort=None):
                try:
                    yield SimpleNamespace(
                        kind="delta",
                        text="Speak this. ",
                        turn_id="turn-speech-failed",
                    )
                    await asyncio.Event().wait()
                finally:
                    generator_closed.set()

            async def interrupt(self, turn_id):
                self.interrupted.append(turn_id)

        class FailingSpeaker:
            def __init__(self):
                self.interruptions = 0

            async def say(self, _text):
                raise RuntimeError("speech failed")

            def interrupt(self):
                self.interruptions += 1

        provider, speaker = Provider(), FailingSpeaker()
        turn = DuplexTurn(provider, speaker)
        with self.assertRaisesRegex(RuntimeError, "speech failed"):
            await asyncio.wait_for(turn.run("hello", "low"), timeout=0.1)
        self.assertTrue(generator_closed.is_set())
        self.assertEqual(speaker.interruptions, 1)
        self.assertEqual(provider.interrupted, ["turn-speech-failed"])

    async def test_local_acknowledgment_speaks_before_slow_reasoning_reply(self):
        release = asyncio.Event()

        class Provider:
            async def stream(self, _text, *, context=(), effort=None):
                await release.wait()
                yield SimpleNamespace(
                    kind="delta",
                    text="Reasoned response.",
                    turn_id="turn-1",
                )

        class CapturingSpeaker:
            def __init__(self):
                self.spoken = []
                self.acknowledged = asyncio.Event()

            async def say(self, text):
                self.spoken.append(text)
                if text == "I hear you.":
                    self.acknowledged.set()

            def interrupt(self):
                pass

        speaker = CapturingSpeaker()
        turn = DuplexTurn(
            Provider(),
            speaker,
            prefix_speech="I hear you.",
        )
        task = asyncio.create_task(turn.run("Keep working.", "high"))
        await asyncio.wait_for(speaker.acknowledged.wait(), timeout=0.1)
        self.assertEqual(speaker.spoken, ["I hear you."])
        release.set()
        self.assertEqual(await task, "Reasoned response.")
        self.assertEqual(speaker.spoken, ["I hear you.", "Reasoned response."])

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
                self.speaker = SimpleNamespace(first_play_at=1.0)

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
