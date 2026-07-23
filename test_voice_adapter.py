import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import voice_adapter


class Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return self.body


class VoiceAdapterTests(unittest.TestCase):
    @patch("voice_adapter.request_json", return_value={"status": "healthy"})
    def test_tts_health(self, _):
        self.assertTrue(voice_adapter.tts_health()["ok"])

    @patch("voice_adapter.urllib.request.urlopen", return_value=Response(b"R" * 64))
    def test_synthesize_writes_audio(self, _):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "speech.wav"
            result = voice_adapter.synthesize("hello", output)
            self.assertTrue(result["ok"])
            self.assertEqual(output.read_bytes(), b"R" * 64)

    @patch("voice_adapter.subprocess.run")
    def test_asr_health(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = "1.2.1\n"
        run.return_value.stderr = ""
        with tempfile.NamedTemporaryFile() as python:
            result = voice_adapter.asr_health(python.name)
        self.assertTrue(result["ok"])
        self.assertEqual(result["version"], "1.2.1")


if __name__ == "__main__":
    unittest.main()
