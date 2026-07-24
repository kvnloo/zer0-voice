import asyncio
import json
import unittest

from shadow import (
    HashDiscardSink,
    MINIMUM_PROMOTION_CASES,
    ShadowCase,
    ShadowFragment,
    TranscriptLeakError,
    assert_transcript_free,
    dumps,
    run_shadow,
)
from release import verdict_from_canary


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeAsr:
    input_mode = "explicit-corpus"

    def __init__(self, clock, values):
        self.clock = clock
        self.values = dict(values)

    async def transcribe(self, audio):
        self.clock.advance(0.1)
        value = self.values[audio]
        if isinstance(value, Exception):
            raise value
        return value


class FakeModel:
    mode = "isolated-ephemeral"

    def __init__(self, clock):
        self.clock = clock
        self.prompts = []
        self.snapshots = []

    async def generate(self, prompt, *, completed_turn_snapshot):
        self.prompts.append(prompt)
        self.snapshots.append(completed_turn_snapshot)
        self.clock.advance(0.2)
        return f"answer {len(self.prompts)}"


class FakeTts:
    output_mode = "pcm-bytes"

    def __init__(self, clock, *, empty=False):
        self.clock = clock
        self.empty = empty

    async def synthesize(self, text):
        self.clock.advance(0.3)
        return b"" if self.empty else text.encode()


class FakeSink(HashDiscardSink):
    def __init__(self, clock):
        self.clock = clock

    async def consume(self, pcm):
        self.clock.advance(0.05)
        return await super().consume(pcm)


def corpus(size=MINIMUM_PROMOTION_CASES):
    return tuple(
        ShadowCase(
            f"case-{index}",
            (ShadowFragment(f"audio-{index}".encode(), 0.0),),
        )
        for index in range(size)
    )


def adapters(cases):
    clock = Clock()
    values = {
        case.fragments[0].audio: f"unique command number {index}."
        for index, case in enumerate(cases)
    }
    return (
        clock,
        FakeAsr(clock, values),
        FakeModel(clock),
        FakeTts(clock),
        FakeSink(clock),
    )


class ShadowTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_hermetic_cohort_promotes_with_deterministic_metrics(self):
        cases = corpus()
        clock, asr, model, tts, sink = adapters(cases)
        report = await run_shadow(
            cases,
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["promotion"]["eligible"])
        self.assertEqual(report["counts"]["completed"], 10)
        self.assertEqual(report["counts"]["empty_model_outputs"], 0)
        with self.assertRaisesRegex(
            ValueError,
            "physical continuous-voice",
        ):
            verdict_from_canary(report, "0" * 64)
        self.assertEqual(report["latencies"]["asr_seconds"]["p95"], 0.1)
        self.assertEqual(report["latencies"]["model_seconds"]["p95"], 0.2)
        self.assertEqual(
            report["cases"][0]["stage_sequence"],
            ["asr", "turn_owner", "model", "tts", "sink"],
        )
        self.assertRegex(report["cases"][0]["pcm"]["sha256"], r"^[a-f0-9]{64}$")

    async def test_nine_green_cases_hold_instead_of_promoting(self):
        cases = corpus(9)
        clock, asr, model, tts, sink = adapters(cases)
        report = await run_shadow(
            cases,
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        self.assertEqual(report["status"], "collecting")
        self.assertEqual(report["promotion"]["verdict"], "hold")
        self.assertFalse(report["promotion"]["eligible"])

    async def test_fragments_flow_through_one_turn_owner_commit(self):
        clock = Clock()
        case = ShadowCase(
            "fragmented",
            (
                ShadowFragment(b"one", 0.0),
                ShadowFragment(b"two", 0.2),
                ShadowFragment(b"three", 0.4),
            ),
        )
        asr = FakeAsr(
            clock,
            {
                b"one": "we need to",
                b"two": "test the shadow",
                b"three": "runner.",
            },
        )
        model, tts, sink = FakeModel(clock), FakeTts(clock), FakeSink(clock)
        report = await run_shadow(
            (case,),
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        self.assertEqual(report["counts"]["completed"], 1)
        self.assertEqual(model.prompts, ["we need to test the shadow runner."])
        self.assertNotIn(model.prompts[0], dumps(report))

    async def test_duplicate_finals_reject_entire_cohort(self):
        cases = corpus()
        clock, _, model, tts, sink = adapters(cases)
        asr = FakeAsr(
            clock,
            {case.fragments[0].audio: "same command." for case in cases},
        )
        report = await run_shadow(
            cases,
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["counts"]["duplicates"], 9)
        self.assertFalse(report["promotion"]["eligible"])

    async def test_adapter_error_is_sanitized_and_rejects(self):
        cases = corpus()
        clock, asr, model, tts, sink = adapters(cases)
        private = "private transcript from exception"
        asr.values[cases[0].fragments[0].audio] = RuntimeError(private)
        report = await run_shadow(
            cases,
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        rendered = dumps(report)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["counts"]["adapter_errors"], 1)
        self.assertNotIn(private, rendered)

    async def test_timeout_fails_closed_without_completing_stages(self):
        class HangingAsr:
            input_mode = "explicit-corpus"

            async def transcribe(self, _audio):
                await asyncio.Event().wait()

        cases = corpus()
        clock, _, model, tts, sink = adapters(cases)
        report = await run_shadow(
            cases,
            asr=HangingAsr(),
            model=model,
            tts=tts,
            sink=sink,
            timeout_seconds=0.001,
            clock=clock,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["counts"]["timeouts"], 10)
        self.assertEqual(report["cases"][0]["stage_sequence"], [])

    async def test_missing_tts_output_rejects_missing_stage(self):
        cases = corpus()
        clock, asr, model, _, sink = adapters(cases)
        report = await run_shadow(
            cases,
            asr=asr,
            model=model,
            tts=FakeTts(clock, empty=True),
            sink=sink,
            clock=clock,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cases"][0]["failure"]["stage"], "tts")
        self.assertNotIn("sink", report["cases"][0]["stage_sequence"])

    async def test_empty_model_response_is_a_hard_failure(self):
        class EmptyModel(FakeModel):
            async def generate(self, prompt, *, completed_turn_snapshot):
                await super().generate(
                    prompt,
                    completed_turn_snapshot=completed_turn_snapshot,
                )
                return "   "

        cases = corpus()
        clock, asr, _, tts, sink = adapters(cases)
        report = await run_shadow(
            cases,
            asr=asr,
            model=EmptyModel(clock),
            tts=tts,
            sink=sink,
            clock=clock,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cases"][0]["failure"], {
            "stage": "model",
            "code": "empty-model-output",
        })
        self.assertEqual(report["counts"]["empty_model_outputs"], 10)
        release_verdict = verdict_from_canary(report, "0" * 64)
        self.assertEqual(release_verdict["verdict"], "reject")
        self.assertEqual(release_verdict["empty_model_outputs"], 10)
        self.assertNotIn("tts", report["cases"][0]["stage_sequence"])

    async def test_synthetic_completed_turn_snapshot_reaches_only_ephemeral_model(self):
        snapshot = b"synthetic completed turn context"
        base = corpus(1)[0]
        case = ShadowCase(
            base.case_id,
            base.fragments,
            completed_turn_snapshot=snapshot,
        )
        clock, asr, model, tts, sink = adapters((case,))
        report = await run_shadow(
            (case,),
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        self.assertEqual(model.snapshots, [snapshot])
        self.assertEqual(report["counts"]["context_cases"], 1)
        self.assertNotIn(snapshot.decode(), dumps(report))

    async def test_non_ephemeral_model_contract_never_runs_candidate(self):
        class UnsafeModel(FakeModel):
            mode = "authoritative"

        cases = corpus()
        clock, asr, _, tts, sink = adapters(cases)
        model = UnsafeModel(clock)
        report = await run_shadow(
            cases,
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(model.prompts, [])
        self.assertEqual(
            report["cases"][0]["failure"]["code"],
            "non-ephemeral-model",
        )

    async def test_duplicate_case_ids_reject_before_adapters_run(self):
        original = corpus(1)[0]
        cases = (original, original)
        clock, asr, model, tts, sink = adapters(cases)
        report = await run_shadow(
            cases,
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["counts"]["duplicates"], 1)
        self.assertEqual(model.prompts, [])

    async def test_report_schema_rejects_text_fields(self):
        with self.assertRaises(TranscriptLeakError):
            assert_transcript_free({"counts": {}, "transcript": "secret"})
        with self.assertRaises(TranscriptLeakError):
            assert_transcript_free(
                {"status": "secret candidate response"},
                secrets=("secret candidate response",),
            )

    async def test_report_json_contains_no_candidate_transcripts_or_responses(self):
        cases = corpus()
        clock, asr, model, tts, sink = adapters(cases)
        report = await run_shadow(
            cases,
            asr=asr,
            model=model,
            tts=tts,
            sink=sink,
            clock=clock,
        )
        rendered = json.loads(dumps(report))
        serialized = json.dumps(rendered)
        for private in (*asr.values.values(), *model.prompts):
            self.assertNotIn(private, serialized)


if __name__ == "__main__":
    unittest.main()
