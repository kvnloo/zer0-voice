import json
import fcntl
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from repository_layout import logical_path, repository_root

from release import (
    LEGACY_RUNTIME_CONTRACT,
    MANAGED_RUNTIME_REQUIRED_FILES,
    RUNTIME_FILES,
    ReleaseError,
    atomic_json,
    promote,
    read_pointer,
    release_status,
    resolve_production,
    rollback,
    stage,
    verdict_from_canary,
    verify_bundle,
    verify_managed_production,
    write_release_record,
)

ALLOWLIST = (
    "voice/runtime.py",
    "voice/runner",
    *MANAGED_RUNTIME_REQUIRED_FILES,
)


def source(root: Path, marker: str) -> Path:
    (root / "voice").mkdir(parents=True)
    (root / "voice/runtime.py").write_text(f"VERSION = {marker!r}\n")
    runner = root / "voice/runner"
    runner.write_text("#!/bin/sh\nexec python runtime.py\n")
    runner.chmod(0o755)
    for relative in MANAGED_RUNTIME_REQUIRED_FILES:
        target = root / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# managed fixture {marker}: {relative}\n")
        if relative == "voice/duplex":
            target.chmod(0o755)
    return root


def canary(digest: str, verdict: str = "promote", turns: int = 10):
    report = {
        "schema": 1,
        "pipeline": "codex-continuous-pcm-v5",
        "bundle_sha256": digest,
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
        root = repository_root(Path(__file__))
        wrapper = logical_path(root, "voice/candidate-service").read_text()
        self.assertIn("ZERO_VOICE_LIVE_MODEL:-gpt-5.6-luna", wrapper)
        self.assertIn("ZERO_VOICE_LIVE_EFFORT:-low", wrapper)
        self.assertIn("ZERO_VOICE_BARGE_IN:-sustained", wrapper)
        self.assertNotIn("ZERO_VOICE_BARGE_IN:-final", wrapper)
        manager = logical_path(root, "voice/runtime_manager.py").read_text()
        self.assertIn('"--startup-phrase"', manager)
        self.assertIn('""', manager)

    def test_candidate_supervisor_restarts_duplex_instead_of_silent_fallback(self):
        root = repository_root(Path(__file__))
        wrapper = logical_path(root, "voice/candidate-service").read_text()
        self.assertIn('while :; do', wrapper)
        self.assertIn('"duplex canary exited status=$status', wrapper)
        self.assertIn('restarting in ${delay}s', wrapper)
        self.assertNotIn('exec "$bundle/voice/simple-daemon"', wrapper)

    def test_candidate_buffers_followup_speech_and_has_no_startup_chatter(self):
        root = repository_root(Path(__file__))
        wrapper = logical_path(root, "voice/candidate-service").read_text()
        self.assertIn('ZERO_VOICE_BARGE_IN:-sustained', wrapper)
        self.assertIn('--startup-phrase ""', wrapper)
        self.assertIn('ZERO_VOICE_MIC_MODE:-continuous', wrapper)
        self.assertIn('ZERO_VOICE_CONTROL_SOCKET', wrapper)
        self.assertIn('ZERO_VOICE_STATE_DIR', wrapper)
        self.assertIn('ZERO_VOICE_RELEASE_STATE', wrapper)
        self.assertIn('--release-bundle "$digest"', wrapper)

    def test_production_buffers_followup_speech_and_has_no_startup_chatter(self):
        root = repository_root(Path(__file__))
        manager = logical_path(root, "voice/runtime_manager.py").read_text()
        self.assertIn('"--barge-in"', manager)
        self.assertIn('"--startup-phrase"', manager)

    def test_service_has_side_effect_free_release_compatibility_preflight(self):
        root = repository_root(Path(__file__))
        wrapper = logical_path(root, "voice/service").read_text()
        parse_check = wrapper.index('if [ "${1:-}" = "--check" ]')
        resolve = wrapper.index('resolve --path')
        preflight_exit = wrapper.index(
            'if [ "${ZERO_VOICE_CHECK_ONLY:-0}" = 1 ]'
        )
        app_server = wrapper.index("codex app-server daemon start")
        kokoro_probe = wrapper.index('"$kokoro_url/health"')
        self.assertLess(parse_check, resolve)
        self.assertLess(resolve, preflight_exit)
        self.assertLess(preflight_exit, app_server)
        self.assertLess(preflight_exit, kokoro_probe)
        self.assertIn(
            "Voice release failed managed-production compatibility",
            wrapper,
        )

    def test_bundle_pins_adapter_recovery_code_and_its_regression_test(self):
        self.assertIn("voice/candidate-service", RUNTIME_FILES)
        self.assertIn("voice/test_duplex.py", RUNTIME_FILES)
        self.assertIn("voice/release.py", RUNTIME_FILES)
        self.assertIn("voice/repository_layout.py", RUNTIME_FILES)
        self.assertIn("voice/test_health.py", RUNTIME_FILES)
        self.assertIn("voice/turn_contract.py", RUNTIME_FILES)
        self.assertIn("voice/test_turn_contract.py", RUNTIME_FILES)
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

    def pin_legacy(self, root: Path, state: Path) -> str:
        result = stage(
            root,
            state,
            apply=True,
            allowlist=ALLOWLIST,
            runtime_contract=LEGACY_RUNTIME_CONTRACT,
        )
        digest = result["bundle_sha256"]
        release_verdict = canary(digest)
        record = write_release_record(state, digest, release_verdict)
        atomic_json(
            state / "production.json",
            {
                "schema": 1,
                "bundle_sha256": digest,
                "previous_bundle_sha256": None,
                "release_sha256": record["release_sha256"],
                "generation": 1,
            },
        )
        return digest

    def test_legacy_integrity_is_distinct_from_managed_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = source(Path(directory) / "legacy", "legacy")
            state = Path(directory) / "state"
            digest = self.pin_legacy(root, state)
            bundle = state / "bundles" / digest
            self.assertEqual(verify_bundle(bundle)["bundle_sha256"], digest)
            with self.assertRaisesRegex(
                ReleaseError,
                "lacks managed-production profile managed-runtime-v1",
            ):
                verify_managed_production(bundle)
            with self.assertRaisesRegex(
                ReleaseError,
                "lacks managed-production profile managed-runtime-v1",
            ):
                resolve_production(state)
            status = release_status(state)
            self.assertEqual(status["pointer"]["generation"], 1)
            self.assertEqual(status["pointer"]["bundle_sha256"], digest)
            self.assertFalse(status["managed_compatible"])
            self.assertRegex(
                status["managed_compatibility_error"],
                "lacks managed-production profile",
            )

    def test_legacy_cli_status_succeeds_without_pointer_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            state = base / "releases"
            digest = self.pin_legacy(source(base / "legacy", "legacy"), state)
            pointer = state / "production.json"
            before = pointer.read_bytes()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("release.py")),
                    "--state",
                    str(state),
                    "status",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["pointer"]["generation"], 1)
            self.assertEqual(payload["pointer"]["bundle_sha256"], digest)
            self.assertFalse(payload["managed_compatible"])
            self.assertEqual(pointer.read_bytes(), before)

    def test_status_rejects_invalid_legacy_integrity_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            state = base / "releases"
            digest = self.pin_legacy(source(base / "legacy", "legacy"), state)
            pointer = state / "production.json"
            before = pointer.read_bytes()
            runtime = state / "bundles" / digest / "voice/runtime.py"
            runtime.chmod(0o644)
            runtime.write_text("tampered\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("release.py")),
                    "--state",
                    str(state),
                    "status",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(completed.stdout, "")
            self.assertRegex(completed.stderr, "integrity mismatch")
            self.assertEqual(pointer.read_bytes(), before)

    def test_service_check_is_side_effect_free_and_ignores_runtime_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, _, digest = self.staged(directory, "managed-check")
            self.assertTrue(root.is_dir())
            # self.staged uses base/state; give the service that release state.
            staged_state = base / "state"
            promote(staged_state, digest, canary(digest), apply=True)
            managed_state = base / "managed-runtime"
            managed_state.mkdir()
            lock_path = managed_state / "service-v2.lock"
            sentinel = base / "dependency-side-effect"
            app_server_sentinel = base / "app-server-side-effect"
            fake_bin = base / "bin"
            fake_bin.mkdir()
            fake_codex = fake_bin / "codex"
            fake_codex.write_text(
                "#!/bin/sh\n"
                f"touch {app_server_sentinel}\n"
                "exit 99\n"
            )
            fake_codex.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "ZERO_VOICE_STATE_DIR": str(managed_state),
                "ZERO_VOICE_RELEASE_STATE": str(staged_state),
                "ZERO_KOKORO_LAUNCHER": str(sentinel),
            }
            service = Path(__file__).with_name("service")
            with lock_path.open("w") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                completed = subprocess.run(
                    [str(service), "--check", "thread"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertIn(digest, completed.stderr)
            self.assertEqual(list(managed_state.iterdir()), [lock_path])
            self.assertFalse(sentinel.exists())
            self.assertFalse(app_server_sentinel.exists())

    def test_service_normal_startup_uses_canonical_root_without_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, release_state, digest = self.staged(directory, "normal-startup")
            promote(release_state, digest, canary(digest), apply=True)
            runtime_state = base / "runtime"
            fake_bin = base / "bin"
            fake_bin.mkdir()
            calls = base / "calls"
            real_python = sys.executable

            (fake_bin / "codex").write_text(
                f"#!/bin/sh\nprintf 'codex %s\\n' \"$*\" >> {calls}\n"
            )
            (fake_bin / "cargo").write_text(
                f"#!/bin/sh\nprintf 'cargo %s\\n' \"$*\" >> {calls}\n"
            )
            (fake_bin / "curl").write_text(
                "#!/bin/sh\n"
                f"printf 'curl %s\\n' \"$*\" >> {calls}\n"
                "case \"$*\" in\n"
                "  *8880/health*) exit 0 ;;\n"
                f"  *8787/healthz*) test -e {base / 'relay-probed'} && exit 0; "
                f"touch {base / 'relay-probed'}; exit 1 ;;\n"
                "esac\n"
                "exit 1\n"
            )
            (fake_bin / "python").write_text(
                "#!/bin/sh\n"
                f"case \"$1\" in\n  */integration/relay.py|*/voice/runtime_manager.py) "
                f"printf 'python %s\\n' \"$*\" >> {calls}; exit 0 ;;\nesac\n"
                f"exec {real_python} \"$@\"\n"
            )
            for executable in fake_bin.iterdir():
                executable.chmod(0o755)

            completed = subprocess.run(
                [str(Path(__file__).with_name("service")), "thread"],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "ZERO_VOICE_LOCKED": "1",
                    "ZERO_VOICE_STATE_DIR": str(runtime_state),
                    "ZERO_VOICE_RELEASE_STATE": str(release_state),
                },
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            invocation_log = calls.read_text()
            root = repository_root(Path(__file__))
            self.assertIn(
                f"cargo build --release --manifest-path {root}/core/Cargo.toml --bin zer0d",
                invocation_log,
            )
            self.assertIn(
                f"python {root}/integration/relay.py --events {runtime_state}/events.jsonl "
                f"--zer0d {root}/core/target/release/zer0d",
                invocation_log,
            )
            self.assertIn(f"--root {root}", invocation_log)
            self.assertIn(f"--metrics {root}/bench/voice-history.jsonl", invocation_log)

    def test_incompatible_service_check_creates_no_state_or_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            release_state = base / "releases"
            pointer_digest = self.pin_legacy(
                source(base / "legacy", "legacy"),
                release_state,
            )
            runtime_state = base / "must-not-exist"
            pointer = release_state / "production.json"
            before = pointer.read_bytes()
            completed = subprocess.run(
                [str(Path(__file__).with_name("service")), "--check", "thread"],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "ZERO_VOICE_STATE_DIR": str(runtime_state),
                    "ZERO_VOICE_RELEASE_STATE": str(release_state),
                },
            )
            self.assertEqual(completed.returncode, 78)
            self.assertEqual(completed.stdout, "")
            self.assertIn("managed-production compatibility", completed.stderr)
            self.assertIn(pointer_digest, str(release_state / "bundles" / pointer_digest))
            self.assertFalse(runtime_state.exists())
            self.assertEqual(pointer.read_bytes(), before)

    def test_managed_profile_rejects_missing_transitive_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = source(Path(directory) / "source", "missing")
            state = Path(directory) / "state"
            allowlist = tuple(
                relative
                for relative in ALLOWLIST
                if relative != "voice/health.py"
            )
            result = stage(root, state, apply=True, allowlist=allowlist)
            bundle = state / "bundles" / result["bundle_sha256"]
            verify_bundle(bundle)
            with self.assertRaisesRegex(
                ReleaseError,
                "missing required file: voice/health.py",
            ):
                verify_managed_production(bundle)

    def test_migration_from_verified_legacy_to_managed_uses_generation_cas(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            state = base / "state"
            legacy = self.pin_legacy(source(base / "legacy", "legacy"), state)
            _, _, managed = self.staged(directory, "managed")
            result = promote(
                state,
                managed,
                canary(managed),
                apply=True,
                expected_generation=1,
            )
            self.assertEqual(result["before"], legacy)
            self.assertEqual(read_pointer(state)["bundle_sha256"], managed)
            self.assertEqual(read_pointer(state)["generation"], 2)
            self.assertEqual(resolve_production(state).name, managed)

    def test_rollback_refuses_legacy_incompatible_target_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            state = base / "state"
            self.pin_legacy(source(base / "legacy", "legacy"), state)
            _, _, managed = self.staged(directory, "managed")
            promote(
                state,
                managed,
                canary(managed),
                apply=True,
                expected_generation=1,
            )
            pointer = state / "production.json"
            before = pointer.read_bytes()
            with self.assertRaisesRegex(
                ReleaseError,
                "lacks managed-production profile managed-runtime-v1",
            ):
                rollback(state, apply=True, expected_generation=2)
            self.assertEqual(pointer.read_bytes(), before)

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

    def test_contended_promotions_have_one_winner_per_expected_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, first = self.staged(directory, "one")
            promote(state, first, canary(first), apply=True)
            _, _, second = self.staged(directory, "two")
            _, _, third = self.staged(directory, "three")
            barrier = threading.Barrier(2)

            def contender(digest):
                barrier.wait()
                try:
                    return promote(
                        state,
                        digest,
                        canary(digest),
                        apply=True,
                        expected_generation=1,
                    )
                except ReleaseError as error:
                    return error

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(contender, (second, third)))
            winners = [item for item in outcomes if isinstance(item, dict)]
            losers = [item for item in outcomes if isinstance(item, ReleaseError)]
            self.assertEqual((len(winners), len(losers)), (1, 1))
            pointer = read_pointer(state)
            self.assertEqual(pointer["generation"], 2)
            self.assertEqual(pointer["previous_bundle_sha256"], first)
            self.assertRegex(str(losers[0]), "generation changed")

    def test_contended_promote_and_rollback_cannot_lose_pointer_update(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, first = self.staged(directory, "one")
            promote(state, first, canary(first), apply=True)
            _, _, second = self.staged(directory, "two")
            promote(state, second, canary(second), apply=True)
            _, _, third = self.staged(directory, "three")
            barrier = threading.Barrier(2)

            def promote_third():
                barrier.wait()
                return promote(
                    state,
                    third,
                    canary(third),
                    apply=True,
                    expected_generation=2,
                )

            def roll_back():
                barrier.wait()
                return rollback(
                    state,
                    apply=True,
                    expected_generation=2,
                )

            def outcome(operation):
                try:
                    return operation()
                except ReleaseError as error:
                    return error

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(outcome, promote_third),
                    pool.submit(outcome, roll_back),
                ]
                outcomes = [future.result() for future in futures]
            self.assertEqual(
                sum(isinstance(item, dict) for item in outcomes),
                1,
            )
            self.assertEqual(
                sum(isinstance(item, ReleaseError) for item in outcomes),
                1,
            )
            pointer = read_pointer(state)
            self.assertEqual(pointer["generation"], 3)
            self.assertEqual(pointer["previous_bundle_sha256"], second)

    def test_independent_cli_promotions_have_one_cas_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, first = self.staged(directory, "one")
            promote(state, first, canary(first), apply=True)
            _, _, second = self.staged(directory, "two")
            _, _, third = self.staged(directory, "three")
            verdicts = []
            for index, digest in enumerate((second, third)):
                path = Path(directory) / f"verdict-{index}.json"
                path.write_text(json.dumps(canary(digest)))
                verdicts.append(path)
            script = Path(__file__).with_name("release.py")
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        str(script),
                        "--state",
                        str(state),
                        "promote",
                        digest,
                        "--verdict",
                        str(verdict),
                        "--apply",
                        "--expected-generation",
                        "1",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for digest, verdict in zip(
                    (second, third), verdicts, strict=True
                )
            ]
            outcomes = [process.communicate(timeout=10) for process in processes]
            self.assertEqual(
                sorted(process.returncode for process in processes),
                [0, 1],
            )
            self.assertTrue(
                any("generation changed" in stderr for _, stderr in outcomes)
            )
            pointer = read_pointer(state)
            self.assertEqual(pointer["generation"], 2)
            self.assertEqual(pointer["previous_bundle_sha256"], first)

    def test_cli_apply_requires_expected_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, digest = self.staged(directory, "one")
            verdict = Path(directory) / "verdict.json"
            verdict.write_text(json.dumps(canary(digest)))
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("release.py")),
                    "--state",
                    str(state),
                    "promote",
                    digest,
                    "--verdict",
                    str(verdict),
                    "--apply",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("--expected-generation", result.stderr)
            self.assertIsNone(read_pointer(state))

    def test_interrupted_pointer_replace_keeps_prior_release_resolvable(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, first = self.staged(directory, "one")
            promote(state, first, canary(first), apply=True)
            _, _, second = self.staged(directory, "two")
            real_atomic = __import__("release").atomic_json

            def interrupted(path, value):
                if Path(path).name == "production.json":
                    raise OSError("injected pointer interruption")
                return real_atomic(path, value)

            with patch("release.atomic_json", side_effect=interrupted):
                with self.assertRaisesRegex(OSError, "injected"):
                    promote(
                        state,
                        second,
                        canary(second),
                        apply=True,
                        expected_generation=1,
                    )
            pointer = read_pointer(state)
            self.assertEqual(pointer["bundle_sha256"], first)
            self.assertEqual(pointer["generation"], 1)
            self.assertEqual(resolve_production(state).name, first)
            applied = promote(
                state,
                second,
                canary(second),
                apply=True,
                expected_generation=1,
            )
            self.assertTrue(applied["applied"])
            self.assertEqual(resolve_production(state).name, second)

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

    def test_shadow_only_report_cannot_authorize_production(self):
        report = {
            "schema": 1,
            "pipeline": "voice-shadow-v1",
            "status": "passed",
            "promotion": {
                "eligible": True,
                "verdict": "promote",
                "observed_completed_turns": 10,
            },
            "counts": {"empty_model_outputs": 0},
        }
        with self.assertRaisesRegex(ReleaseError, "physical continuous-voice"):
            verdict_from_canary(report, "0" * 64)

    def test_physical_report_from_another_bundle_cannot_authorize(self):
        report = {
            "schema": 1,
            "pipeline": "codex-continuous-pcm-v5",
            "bundle_sha256": "1" * 64,
            "status": "passed",
            "promotion": {
                "eligible": True,
                "verdict": "promote",
                "observed_completed_turns": 10,
            },
            "counts": {"empty_model_outputs": 0},
        }
        with self.assertRaisesRegex(ReleaseError, "another bundle"):
            verdict_from_canary(report, "2" * 64)

    def test_service_delegates_generation_execution_to_stable_manager(self):
        script = Path(__file__).with_name("service").read_text(encoding="utf-8")
        self.assertIn("resolve --path", script)
        self.assertIn('"$bundle_root/voice/runtime_manager.py"', script)
        self.assertNotIn('"$root/voice/runtime_manager.py"', script)
        self.assertNotIn('"$root/voice/duplex"', script)
        self.assertNotIn('"$root/voice/watchdog.py"', script)

    def test_codex_wrapper_has_no_competing_voice_supervisor(self):
        script = Path(__file__).with_name("codex-harness").read_text(
            encoding="utf-8"
        )
        self.assertIn('"$root/voice/service" "$thread"', script)
        self.assertNotIn("supervise_voice", script)
        self.assertNotIn('"$root/voice/duplex"', script)

    def test_production_service_enables_fail_closed_focus_routing(self):
        script = Path(__file__).with_name("service").read_text(encoding="utf-8")
        self.assertIn("workspace_routing=(--workspace-routing)", script)
        self.assertIn('"${workspace_routing[@]}"', script)


if __name__ == "__main__":
    unittest.main()
