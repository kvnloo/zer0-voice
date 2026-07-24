import json
import tempfile
import unittest
from pathlib import Path

from lexicon import Lexicon, load_lexicon


class LexiconTests(unittest.TestCase):
    def test_hotwords_cover_project_and_ai_vocabulary(self):
        lexicon = Lexicon()
        hotwords = lexicon.hotwords
        for term in ("Codex", "Kokoro", "tldraw", "XYFlow", "PyTorch"):
            self.assertIn(term, hotwords)
        self.assertNotIn("JAX", lexicon.prompt)
        self.assertNotIn("Vocabulary:", lexicon.prompt)

    def test_corrections_are_phrase_bounded_and_conservative(self):
        lexicon = Lexicon()
        self.assertEqual(
            lexicon.correct("use Cocoro TTA in Teamux"),
            "use Kokoro TTS in tmux",
        )
        self.assertEqual(lexicon.correct("we should go now"), "we should go now")

    def test_repo_extension_merges_without_replacing_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexicon.json"
            path.write_text(json.dumps({"terms": ["NanoService"]}))
            lexicon = load_lexicon(path)
        self.assertIn("Codex", lexicon.terms)
        self.assertIn("NanoService", lexicon.terms)


if __name__ == "__main__":
    unittest.main()
