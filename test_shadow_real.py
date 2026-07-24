import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from shadow import MINIMUM_PROMOTION_CASES
from shadow_real import (
    EphemeralCodexModel,
    KokoroPcmSynthesizer,
    SYNTHETIC_TURNS,
    atomic_report,
    build_synthetic_corpus,
    parser,
)


class FakeServer:
    shared = False

    def __init__(self):
        self.starts = []

    async def start_thread(self, **kwargs):
        self.starts.append(kwargs)
        return f"ephemeral-{len(self.starts)}"


class UnsafeSharedServer(FakeServer):
    shared = True


class ShadowRealTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    async def test_corpus_is_explicit_in_memory_and_at_least_ten_cases(self):
        with (
            patch("shadow_real.kokoro_wav", return_value=b"wav"),
            patch(
                "shadow_real.decode_wav",
                return_value=(np.ones(8, dtype=np.float32), 24_000),
            ),
            patch("shadow_real.asyncio.to_thread", self._inline_to_thread),
        ):
            corpus = await build_synthetic_corpus(
                kokoro_url="http://isolated",
                voice="candidate",
                cases=MINIMUM_PROMOTION_CASES,
            )
        self.assertEqual(len(corpus), 10)
        self.assertEqual(len({case.case_id for case in corpus}), 10)
        self.assertTrue(all(case.fragments[0].audio == b"wav" for case in corpus))
        self.assertTrue(all(case.fragments[0].at_seconds == 0.0 for case in corpus))

    async def test_corpus_size_fails_closed_below_promotion_minimum(self):
        with self.assertRaisesRegex(ValueError, "cases must be"):
            await build_synthetic_corpus(
                kokoro_url="http://isolated",
                voice="candidate",
                cases=MINIMUM_PROMOTION_CASES - 1,
            )

    async def test_model_rejects_shared_production_capable_transport(self):
        with self.assertRaisesRegex(ValueError, "private app-server"):
            EphemeralCodexModel(
                UnsafeSharedServer(),
                ROOT_FOR_TEST,
                "gpt-5.6-luna",
                "low",
            )

    async def test_model_creates_fresh_ephemeral_thread_per_case(self):
        class Event:
            def __init__(self, kind, text=""):
                self.kind = kind
                self.text = text

        class Provider:
            def __init__(self, *_):
                pass

            async def stream(self, *_args, **_kwargs):
                yield Event("delta", "safe answer")
                yield Event("completed")

        server = FakeServer()
        adapter = EphemeralCodexModel(
            server,
            ROOT_FOR_TEST,
            "gpt-5.6-luna",
            "low",
        )
        with patch("shadow_real.CodexSubscription", Provider):
            first = await adapter.generate("one", completed_turn_snapshot=None)
            second = await adapter.generate(
                "two",
                completed_turn_snapshot=b"synthetic context",
            )
        self.assertEqual((first, second), ("safe answer", "safe answer"))
        self.assertEqual(len(server.starts), 2)
        self.assertTrue(all(item["ephemeral"] for item in server.starts))
        self.assertTrue(
            all(item["model"] == "gpt-5.6-luna" for item in server.starts)
        )

    async def test_kokoro_adapter_validates_pcm_without_playback(self):
        adapter = KokoroPcmSynthesizer("http://isolated", "candidate")
        with (
            patch("shadow_real.kokoro_wav", return_value=b"pcm-container"),
            patch(
                "shadow_real.decode_wav",
                return_value=(np.ones(8, dtype=np.float32), 24_000),
            ),
            patch("shadow_real.asyncio.to_thread", self._inline_to_thread),
        ):
            self.assertEqual(
                await adapter.synthesize("synthetic"),
                b"pcm-container",
            )

    def test_cli_exposes_no_production_resource_selectors(self):
        option_strings = {
            option
            for action in parser()._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--input",
            "--output",
            "--session",
            "--thread",
            "--shared",
            "--sink",
        ):
            self.assertNotIn(forbidden, option_strings)
        self.assertGreaterEqual(len(SYNTHETIC_TURNS), MINIMUM_PROMOTION_CASES)

    def test_report_is_atomic_and_release_consumable(self):
        report = {
            "schema": 1,
            "pipeline": "voice-shadow-v1",
            "status": "passed",
            "counts": {"empty_model_outputs": 0},
            "promotion": {
                "eligible": True,
                "verdict": "promote",
                "observed_completed_turns": 10,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            atomic_report(path, report)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded, report)
            self.assertEqual(
                verdict_from_canary(loaded, "a" * 64)["verdict"],
                "promote",
            )
            self.assertEqual(list(path.parent.glob(".cohort.json.*")), [])


from pathlib import Path
from release import verdict_from_canary

ROOT_FOR_TEST = Path("/synthetic-shadow")


if __name__ == "__main__":
    unittest.main()
