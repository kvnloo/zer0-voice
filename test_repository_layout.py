import tempfile
import unittest
from pathlib import Path

from repository_layout import logical_path, repository_root


class RepositoryLayoutTests(unittest.TestCase):
    def test_root_hoisted_mirror_resolves_voice_and_adapter_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / "release.py"
            anchor.write_text("")
            for name in ("candidate-service", "providers.py", "app_server.py", "events.py"):
                (root / name).write_text(name)

            self.assertEqual(repository_root(anchor), root)
            self.assertEqual(logical_path(root, "voice/candidate-service"), root / "candidate-service")
            self.assertEqual(logical_path(root, "adapters/llm/providers.py"), root / "providers.py")
            self.assertEqual(logical_path(root, "adapters/codex/app_server.py"), root / "app_server.py")
            self.assertEqual(logical_path(root, "contracts/events.py"), root / "events.py")

    def test_monorepo_and_staged_bundle_keep_canonical_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / "voice" / "release.py"
            anchor.parent.mkdir()
            anchor.write_text("")
            canonical = root / "voice" / "candidate-service"
            canonical.write_text("")

            self.assertEqual(repository_root(anchor), root)
            self.assertEqual(logical_path(root, "voice/candidate-service"), canonical)

    def test_missing_logical_input_is_not_silently_remapped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                logical_path(root, "adapters/llm/missing.py"),
                root / "adapters" / "llm" / "missing.py",
            )


if __name__ == "__main__":
    unittest.main()
