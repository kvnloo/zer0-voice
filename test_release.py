import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from release import (
    RUNTIME_FILES,
    ReleaseError,
    promote,
    read_pointer,
    resolve_production,
    rollback,
    stage,
    verdict_from_canary,
    verify_bundle,
)

ALLOWLIST = ("voice/runtime.py", "voice/runner")


def source(root: Path, marker: str) -> Path:
    (root / "voice").mkdir(parents=True)
    (root / "voice/runtime.py").write_text(f"VERSION = {marker!r}\n")
    runner = root / "voice/runner"
    runner.write_text("#!/bin/sh\nexec python runtime.py\n")
    runner.chmod(0o755)
    return root


def canary(digest: str, verdict: str = "promote", turns: int = 10):
    report = {
        "schema": 1,
        "status": "passed" if verdict == "promote" else "collecting",
        "promotion": {
            "eligible": verdict == "promote",
            "verdict": verdict,
            "observed_completed_turns": turns,
        },
        "counts": {
            "canonical_completions": turns,
            "empty_model_outputs": 0,
        },
    }
    return verdict_from_canary(report, digest)


class ReleaseTests(unittest.TestCase):
    def test_service_wrappers_use_dedicated_instant_lane_defaults(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "voice/candidate-service").read_text()
        self.assertIn("ZERO_VOICE_LIVE_MODEL:-gpt-5.6-luna", wrapper)
        self.assertIn("ZERO_VOICE_LIVE_EFFORT:-low", wrapper)
        self.assertIn("ZERO_VOICE_BARGE_IN:-final", wrapper)
        self.assertNotIn("ZERO_VOICE_BARGE_IN:-sustained", wrapper)
        manager = (root / "voice/runtime_manager.py").read_text()
        self.assertIn('"--startup-phrase"', manager)
        self.assertIn('""', manager)

    def test_candidate_supervisor_restarts_duplex_instead_of_silent_fallback(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "voice/candidate-service").read_text()
        self.assertIn('while :; do', wrapper)
        self.assertIn('"duplex canary exited status=$status', wrapper)
        self.assertIn('restarting in ${delay}s', wrapper)
        self.assertNotIn('exec "$bundle/voice/simple-daemon"', wrapper)

    def test_candidate_buffers_followup_speech_and_has_no_startup_chatter(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "voice/candidate-service").read_text()
        self.assertIn('ZERO_VOICE_BARGE_IN:-final', wrapper)
        self.assertIn('--startup-phrase ""', wrapper)
        self.assertIn('ZERO_VOICE_MIC_MODE:-continuous', wrapper)
        self.assertIn('ZERO_VOICE_CONTROL_SOCKET', wrapper)
        self.assertIn('ZERO_VOICE_STATE_DIR', wrapper)
        self.assertIn('ZERO_VOICE_RELEASE_STATE', wrapper)

    def test_production_buffers_followup_speech_and_has_no_startup_chatter(self):
        root = Path(__file__).resolve().parents[1]
        manager = (root / "voice/runtime_manager.py").read_text()
        self.assertIn('"--barge-in"', manager)
        self.assertIn('"--startup-phrase"', manager)

    def test_bundle_pins_adapter_recovery_code_and_its_regression_test(self):
        self.assertIn("voice/candidate-service", RUNTIME_FILES)
        self.assertIn("voice/release.py", RUNTIME_FILES)
        self.assertIn("voice/test_health.py", RUNTIME_FILES)
        self.assertIn("voice/handoff.py", RUNTIME_FILES)
        self.assertIn("voice/test_handoff.py", RUNTIME_FILES)
        self.assertIn("voice/runtime_manager.py", RUNTIME_FILES)
        self.assertIn("voice/test_runtime_manager.py", RUNTIME_FILES)
        self.assertIn("adapters/codex/app_server.py", RUNTIME_FILES)
        self.assertIn("adapters/codex/test_app_server.py", RUNTIME_FILES)
        self.assertIn("adapters/llm/providers.py", RUNTIME_FILES)
        self.assertIn("adapters/llm/test_providers.py", RUNTIME_FILES)
        self.assertIn("adapters/voice_pm/publisher.py", RUNTIME_FILES)
        self.assertIn("adapters/voice_pm/wiring.py", RUNTIME_FILES)
        self.assertIn("adapters/voice_pm/test_wiring.py", RUNTIME_FILES)
        self.assertIn("voice/simple.py", RUNTIME_FILES)
        self.assertIn("voice/test_simple.py", RUNTIME_FILES)
        self.assertIn("voice/workspace_router.py", RUNTIME_FILES)
        self.assertIn("voice/routes.json", RUNTIME_FILES)
        self.assertIn("voice/shadow_real.py", RUNTIME_FILES)
        self.assertIn("voice/test_shadow_real.py", RUNTIME_FILES)

    def staged(self, directory: str, marker: str):
        root = source(Path(directory) / f"source-{marker}", marker)
        state = Path(directory) / "state"
        result = stage(root, state, apply=True, allowlist=ALLOWLIST)
        return root, state, result["bundle_sha256"]

    def test_manifest_detects_tampering_and_unallowlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, digest = self.staged(directory, "one")
            bundle = state / "bundles" / digest
            runtime = bundle / "voice/runtime.py"
            runtime.chmod(0o644)
            runtime.write_text("VERSION = 'tampered'\n")
            with self.assertRaisesRegex(ReleaseError, "integrity"):
                verify_bundle(bundle)

    def test_manifest_rejects_extra_files_and_unsafe_source_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root, state, digest = self.staged(directory, "one")
            bundle = state / "bundles" / digest
            bundle.chmod(0o755)
            (bundle / "extra.py").write_text("unexpected\n")
            with self.assertRaisesRegex(ReleaseError, "allowlist"):
                verify_bundle(bundle)
            external = Path(directory) / "external"
            external.mkdir()
            (external / "runtime.py").write_text("external\n")
            unsafe = Path(directory) / "unsafe"
            unsafe.mkdir()
            (unsafe / "voice").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ReleaseError, "symlink"):
                stage(
                    unsafe,
                    Path(directory) / "unused",
                    allowlist=("voice/runtime.py",),
                )

    def test_hold_reject_and_short_canary_cannot_change_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, first = self.staged(directory, "one")
            promote(state, first, canary(first), apply=True)
            _, _, second = self.staged(directory, "two")
            before = (state / "production.json").read_bytes()
            for decision in ("hold", "reject"):
                result = promote(
                    state,
                    second,
                    canary(second, decision, 10),
                    apply=True,
                )
                self.assertFalse(result["applied"])
                self.assertEqual((state / "production.json").read_bytes(), before)
            with self.assertRaisesRegex(ReleaseError, "at least 10"):
                promote(state, second, canary(second, "promote", 9), apply=True)
            self.assertEqual((state / "production.json").read_bytes(), before)

    def test_promotion_is_dry_run_by_default_and_pointer_swap_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, digest = self.staged(directory, "one")
            dry_run = promote(state, digest, canary(digest))
            self.assertFalse(dry_run["applied"])
            self.assertIsNone(read_pointer(state))
            real_replace = os.replace
            replacements = []

            def observed_replace(source_path, destination_path):
                replacements.append(Path(destination_path))
                return real_replace(source_path, destination_path)

            with patch("release.os.replace", side_effect=observed_replace):
                result = promote(state, digest, canary(digest), apply=True)
            self.assertTrue(result["applied"])
            self.assertIn(state / "production.json", replacements)
            self.assertEqual(resolve_production(state).name, digest)

    def test_rollback_selects_prior_verified_bundle_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, first = self.staged(directory, "one")
            promote(state, first, canary(first), apply=True)
            _, _, second = self.staged(directory, "two")
            promote(state, second, canary(second), apply=True)
            preview = rollback(state)
            self.assertFalse(preview["applied"])
            self.assertEqual(resolve_production(state).name, second)
            applied = rollback(state, apply=True)
            self.assertTrue(applied["applied"])
            self.assertEqual(resolve_production(state).name, first)

    def test_dirty_worktree_cannot_change_pinned_restart_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, state, digest = self.staged(directory, "clean")
            promote(state, digest, canary(digest), apply=True)
            (root / "voice/runtime.py").write_text("VERSION = 'dirty'\n")
            pinned = resolve_production(state)
            self.assertIn(
                "clean",
                (pinned / "voice/runtime.py").read_text(),
            )
            self.assertNotIn(
                "dirty",
                (pinned / "voice/runtime.py").read_text(),
            )

    def test_canary_envelope_rejects_transcript_bearing_fields(self):
        report = {
            "promotion": {
                "verdict": "promote",
                "observed_completed_turns": 10,
            },
            "text": "private words",
        }
        with self.assertRaisesRegex(ReleaseError, "transcript-bearing"):
            verdict_from_canary(report, "0" * 64)

    def test_promotion_requires_explicit_zero_empty_model_outputs(self):
        report = {
            "promotion": {
                "verdict": "promote",
                "observed_completed_turns": 10,
            },
            "counts": {"empty_model_outputs": 1},
        }
        with self.assertRaisesRegex(ReleaseError, "zero empty"):
            verdict_from_canary(report, "0" * 64)

    def test_service_delegates_generation_execution_to_stable_manager(self):
        script = Path(__file__).with_name("service").read_text(encoding="utf-8")
        self.assertIn("resolve --path", script)
        self.assertIn('"$bundle_root/voice/runtime_manager.py"', script)
        self.assertNotIn('"$root/voice/runtime_manager.py"', script)
        self.assertNotIn('"$root/voice/duplex"', script)
        self.assertNotIn('"$root/voice/watchdog.py"', script)


if __name__ == "__main__":
    unittest.main()
