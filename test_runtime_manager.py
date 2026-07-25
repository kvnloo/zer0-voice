import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from handoff import Candidate, HandoffError
from runtime_manager import ControlProxy, Generation, RuntimeManager, parser


class Process:
    returncode = None

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


def args(root: Path):
    return argparse.Namespace(
        thread="thread-1",
        root=root,
        state=root / "state",
        release_state=root / "releases",
        control_socket=root / "stable.sock",
        kokoro_url="http://kokoro",
        kokoro_launcher=root / "kokoro",
        dependency_timeout=0.1,
        dependency_interval=0.01,
        relay_url="http://relay",
        input="mic",
        output="sink",
        mic_mode="continuous",
        notification_mode="conversational",
        live_model="live",
        live_effort="low",
        live_timeout=35,
        barge_in="final",
        keyboard_indicator="none",
        readiness_timeout=0.1,
        probation=0.01,
        poll_interval=0.001,
        metrics=root / "metrics.jsonl",
        debug_events=root / "debug.jsonl",
        active_generation=root / "active-generation.json",
    )


class RuntimeManagerTests(unittest.IsolatedAsyncioTestCase):
    def test_cli_defaults_match_production_conversation_contract(self):
        parsed = parser().parse_args(["thread-1"])
        self.assertEqual(parsed.live_model, "gpt-5.6-luna")
        self.assertEqual(parsed.live_effort, "low")
        self.assertEqual(parsed.barge_in, "sustained")
        self.assertEqual(parsed.mic_mode, "continuous")

    def test_generation_command_has_private_control_and_same_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = RuntimeManager(args(root))
            command = manager.command(
                root / "bundle",
                root / "generation",
                "muted",
            )
        self.assertEqual(command[command.index("--session") + 1], "thread-1")
        self.assertEqual(command[command.index("--mic-mode") + 1], "muted")
        self.assertEqual(
            command[command.index("--control-socket") + 1],
            str(root / "generation/control.sock"),
        )

    def test_active_manifest_names_exact_health_and_global_debug_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = RuntimeManager(args(root))
            generation = Generation(
                "1" * 64,
                root / "bundle",
                Candidate(
                    root / "state/generations/00000001/control.sock",
                    root / "state/generations/00000001/health.json",
                ),
                Process(),
            )
            manager.publish_active(generation)
            manifest = json.loads(manager.args.active_generation.read_text())
        self.assertEqual(manifest["digest"], "1" * 64)
        self.assertEqual(manifest["generation"], "00000001")
        self.assertEqual(manifest["health"], str(generation.candidate.health))
        self.assertEqual(manifest["debug"], str(manager.args.debug_events))

    def test_dependency_health_requires_explicit_healthy_payload(self):
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        with patch(
            "runtime_manager.urllib.request.urlopen",
            return_value=response,
        ):
            response.read.return_value = b'{"status":"degraded"}'
            self.assertFalse(RuntimeManager.url_ready("http://kokoro/health"))
            response.read.return_value = b'{"status":"healthy"}'
            self.assertTrue(RuntimeManager.url_ready("http://kokoro/health"))

    async def test_dependency_is_repaired_without_replacing_voice_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = args(root)
            configuration.kokoro_launcher.write_text("#!/bin/sh\n")
            configuration.kokoro_launcher.chmod(0o700)
            manager = RuntimeManager(configuration)
            repair = Process()
            repair.returncode = 0
            with patch(
                "runtime_manager.asyncio.to_thread",
                new=AsyncMock(side_effect=[False, True]),
            ), patch(
                "runtime_manager.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=repair),
            ) as spawn:
                await manager.ensure_kokoro()
        spawn.assert_awaited_once_with(str(configuration.kokoro_launcher), "start")

    async def test_dependency_repair_fails_closed_without_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = RuntimeManager(args(Path(directory)))
            with patch(
                "runtime_manager.asyncio.to_thread",
                new=AsyncMock(return_value=False),
            ):
                with self.assertRaisesRegex(RuntimeError, "launcher is missing"):
                    await manager.ensure_kokoro()

    async def test_stable_proxy_follows_active_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "one.sock"
            second_path = root / "two.sock"
            first = Generation(
                "1" * 64,
                root,
                __import__("handoff").Candidate(first_path, root / "one.json"),
                Process(),
            )
            second = Generation(
                "2" * 64,
                root,
                __import__("handoff").Candidate(second_path, root / "two.json"),
                Process(),
            )
            current = first
            proxy = ControlProxy(root / "stable.sock", lambda: current)

            class Reader:
                async def readline(self):
                    return b'{"mic":"muted"}\n'

            class Writer:
                def __init__(self):
                    self.value = b""

                def write(self, value):
                    self.value += value

                async def drain(self):
                    pass

                def close(self):
                    pass

                async def wait_closed(self):
                    pass

            forwarded = AsyncMock(
                side_effect=[
                    {"ok": True, "label": "one"},
                    {"ok": True, "label": "two"},
                ]
            )
            with patch("runtime_manager.request", forwarded):
                writer = Writer()
                await proxy._handle(Reader(), writer)
                self.assertEqual(json.loads(writer.value)["label"], "one")
                current = second
                writer = Writer()
                await proxy._handle(Reader(), writer)
                self.assertEqual(json.loads(writer.value)["label"], "two")
            self.assertEqual(forwarded.await_args_list[0].args[0], first_path)
            self.assertEqual(forwarded.await_args_list[1].args[0], second_path)

    async def test_probation_rejects_dead_or_unhealthy_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = RuntimeManager(args(root))
            health = root / "health.json"
            health.write_text("{}")
            generation = Generation(
                "1" * 64,
                root,
                __import__("handoff").Candidate(root / "sock", health),
                Process(),
            )
            self.assertFalse(await manager.probation(generation))

    async def test_hot_swap_keeps_old_until_candidate_probation_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = RuntimeManager(args(root))
            old = Generation(
                "1" * 64,
                root / "old",
                Candidate(root / "old.sock", root / "old.json"),
                Process(),
            )
            new = Generation(
                "2" * 64,
                root / "new",
                Candidate(root / "new.sock", root / "new.json"),
                Process(),
            )
            manager.active = old
            with patch.object(
                manager,
                "launch",
                new=AsyncMock(return_value=new),
            ), patch(
                "runtime_manager.handoff",
                new=AsyncMock(return_value={"status": "candidate-active"}),
            ) as switch, patch.object(
                manager,
                "probation",
                new=AsyncMock(return_value=True),
            ), patch.object(
                manager,
                "terminate",
                new=AsyncMock(),
            ) as terminate:
                self.assertTrue(await manager.hot_swap(new.digest, new.bundle))
            self.assertIs(manager.active, new)
            self.assertIsNone(manager.rollback)
            self.assertEqual(
                json.loads(manager.args.active_generation.read_text())["digest"],
                new.digest,
            )
            switch.assert_awaited_once()
            terminate.assert_awaited_once_with(old)

    async def test_failed_probation_restores_old_before_killing_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = RuntimeManager(args(root))
            old = Generation(
                "1" * 64,
                root / "old",
                Candidate(root / "old.sock", root / "old.json"),
                Process(),
            )
            new = Generation(
                "2" * 64,
                root / "new",
                Candidate(root / "new.sock", root / "new.json"),
                Process(),
            )
            manager.active = old
            controls = AsyncMock(
                side_effect=[
                    {"ok": True, "capture_active": False},
                    {"ok": True, "capture_active": True},
                ]
            )
            with patch.object(
                manager,
                "launch",
                new=AsyncMock(return_value=new),
            ), patch(
                "runtime_manager.handoff",
                new=AsyncMock(return_value={"status": "candidate-active"}),
            ), patch.object(
                manager,
                "probation",
                new=AsyncMock(return_value=False),
            ), patch(
                "runtime_manager.request",
                controls,
            ), patch.object(
                manager,
                "terminate",
                new=AsyncMock(),
            ) as terminate:
                self.assertFalse(await manager.hot_swap(new.digest, new.bundle))
            self.assertIs(manager.active, old)
            self.assertEqual(
                json.loads(manager.args.active_generation.read_text())["digest"],
                old.digest,
            )
            self.assertEqual(manager.rejected_digest, new.digest)
            self.assertIsNone(manager.rollback)
            self.assertEqual(
                [call.args for call in controls.await_args_list],
                [
                    (new.candidate.control, {"mic": "muted"}),
                    (old.candidate.control, {"mic": "continuous"}),
                ],
            )
            terminate.assert_awaited_once_with(new)

    async def test_rejected_pointer_is_quarantined_without_losing_active(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = RuntimeManager(args(root))
            active = Generation(
                "1" * 64,
                root / "active",
                Candidate(root / "active.sock", root / "active.json"),
                Process(),
            )
            manager.active = active
            manager.hot_swap = AsyncMock(side_effect=HandoffError("bad candidate"))
            with patch(
                "runtime_manager.read_pointer",
                return_value={"bundle_sha256": "2" * 64},
            ), patch(
                "runtime_manager.resolve_production",
                return_value=root / "candidate",
            ):
                with self.assertRaisesRegex(HandoffError, "bad candidate"):
                    await manager.reconcile_release()
                self.assertIsNone(await manager.reconcile_release())
        self.assertIs(manager.active, active)
        self.assertEqual(manager.rejected_digest, "2" * 64)
        manager.hot_swap.assert_awaited_once()
