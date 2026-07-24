import io
import subprocess
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import conversation


class ConversationTests(unittest.TestCase):
    def test_energy_segmenter_keeps_preroll_and_trims_silence(self):
        config = conversation.ListenConfig(
            sample_rate=1000,
            block_ms=100,
            threshold=0.1,
            start_blocks=2,
            silence_ms=200,
            pre_roll_ms=200,
        )
        quiet = np.zeros(100, dtype=np.float32)
        loud = np.full(100, 0.5, dtype=np.float32)
        result = conversation.segment_blocks(
            [quiet, quiet, loud, loud, loud, quiet, quiet], config
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 300)
        self.assertEqual(float(result[-1]), 0.5)

    def test_decode_pcm_wav(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24_000)
            stream.writeframes(np.array([0, 32767], dtype="<i2").tobytes())
        samples, rate = conversation.decode_wav(output.getvalue())
        self.assertEqual(rate, 24_000)
        self.assertEqual(samples.shape, (2,))
        self.assertAlmostEqual(float(samples[1]), 32767 / 32768)

    def test_transcribe_drops_decoder_no_speech_prompt_leakage(self):
        class Model:
            def transcribe(self, _audio, **_kwargs):
                return (
                    iter(
                        (
                            SimpleNamespace(
                                text="JAX JAX JAX",
                                no_speech_prob=0.93,
                                avg_logprob=-1.8,
                            ),
                        )
                    ),
                    None,
                )

        self.assertEqual(
            conversation.transcribe(Model(), np.ones(8, dtype=np.float32)),
            "",
        )

    def test_transcribe_keeps_low_confidence_flag_when_speech_is_probable(self):
        class Model:
            def transcribe(self, _audio, **_kwargs):
                return (
                    iter(
                        (
                            SimpleNamespace(
                                text="Can you hear me?",
                                no_speech_prob=0.08,
                                avg_logprob=-1.2,
                            ),
                        )
                    ),
                    None,
                )

        self.assertEqual(
            conversation.transcribe(Model(), np.ones(8, dtype=np.float32)),
            "Can you hear me?",
        )

    def test_codex_turn_resumes_and_reads_output(self):
        def runner(command, **_kwargs):
            output = Path(command[command.index("-o") + 1])
            output.write_text("spoken answer", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        response = conversation.codex_turn(
            "hello", session="thread-1", cwd=Path("."), runner=runner
        )
        self.assertEqual(response, "spoken answer")

    @patch("conversation.urllib.request.urlopen")
    def test_kokoro_payload(self, urlopen):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"wav"
        urlopen.return_value = response
        self.assertEqual(conversation.kokoro_wav("hello"), b"wav")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.method, "POST")
        self.assertIn(b'"input": "hello"', request.data)


if __name__ == "__main__":
    unittest.main()
