import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "contracts"))
sys.path.insert(0, str(ROOT / "adapters" / "codex"))

from app_server import AppServerError, CodexAppServer, UnixWebSocket


class FakeWriter:
    def __init__(self):
        self.data = bytearray()

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        return None


class FakeProcess:
    def __init__(self):
        self.stdin = FakeWriter()


class AppServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_client_uses_managed_daemon_websocket(self):
        version = AsyncMock()
        version.returncode = 0
        version.communicate.return_value = (
            b'{"socketPath":"/tmp/codex.sock"}',
            b"",
        )

        async def request(method, _params):
            if method == "initialize":
                return {}
            raise AssertionError(method)

        server = CodexAppServer(cwd=Path("."), shared=True)
        server.request = request
        with patch(
            "app_server.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=version),
        ) as spawn, patch.object(
            UnixWebSocket, "connect", new=AsyncMock()
        ), patch.object(
            UnixWebSocket, "send", new=AsyncMock()
        ) as send:
            await server.start()
        spawn.assert_awaited_once_with(
            "codex",
            "app-server",
            "daemon",
            "version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.assertEqual(server.websocket.path, Path("/tmp/codex.sock"))
        self.assertIn('"method":"initialized"', send.await_args.args[0])
        for task in (server.reader, server.stderr):
            if task:
                task.cancel()

    async def test_request_is_jsonl_and_correlated(self):
        server = CodexAppServer(cwd=Path("."))
        server.process = FakeProcess()
        task = asyncio.create_task(server.request("thread/start", {"cwd": "/tmp"}))
        await asyncio.sleep(0)
        sent = json.loads(bytes(server.process.stdin.data))
        self.assertEqual(sent["method"], "thread/start")
        server.pending[sent["id"]].set_result(
            {"id": sent["id"], "result": {"thread": {"id": "thread-1"}}}
        )
        self.assertEqual((await task)["thread"]["id"], "thread-1")

    async def test_protocol_diagnostics_record_methods_but_not_payloads(self):
        server = CodexAppServer(cwd=Path("."))

        class Reader:
            lines = [
                json.dumps(
                    {
                        "method": "item/agentMessage/delta",
                        "params": {
                            "threadId": "thread-1",
                            "delta": "private response text",
                        },
                    }
                ).encode()
                + b"\n",
                b"",
            ]

            async def readline(self):
                return self.lines.pop(0)

        server.process = type("Process", (), {"stdout": Reader()})()
        await server._read()
        self.assertEqual(
            list(server.protocol_methods),
            ["item/agentMessage/delta"],
        )
        self.assertNotIn("private response text", repr(server.protocol_methods))

    async def test_eof_fails_pending_rpc_immediately(self):
        server = CodexAppServer(cwd=Path("."))

        class Reader:
            async def readline(self):
                return b""

        server.process = type(
            "Process",
            (),
            {"stdin": FakeWriter(), "stdout": Reader()},
        )()
        request = asyncio.create_task(server.request("thread/read", {}))
        await asyncio.sleep(0)
        await server._read()
        with self.assertRaisesRegex(AppServerError, "connection closed"):
            await asyncio.wait_for(request, timeout=0.1)
        self.assertEqual(server.pending, {})

    async def test_eof_fails_active_stream_immediately(self):
        server = CodexAppServer(cwd=Path("."))

        async def read_thread(_thread):
            return {"id": "thread-1", "turns": []}

        async def request(method, _params):
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            raise AssertionError(method)

        server.read_thread = read_thread
        server.request = request
        stream = server.stream_turn("thread-1", "hello")
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        self.assertEqual(len(server.notifications["thread-1"]), 1)

        class Reader:
            async def readline(self):
                return b""

        server.process = type("Process", (), {"stdout": Reader()})()
        await server._read()
        with self.assertRaisesRegex(AppServerError, "connection closed"):
            await asyncio.wait_for(waiting, timeout=0.1)
        self.assertEqual(server.notifications["thread-1"], [])

    async def test_stream_turn_emits_shared_delta_and_completion(self):
        server = CodexAppServer(cwd=Path("."))

        async def request(method, _params):
            if method == "thread/read":
                return {"thread": {"id": "thread-1", "turns": []}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            raise AssertionError(method)

        server.request = request
        stream = server.stream_turn("thread-1", "hello")
        delta_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        server.notifications["thread-1"][0].put_nowait(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "delta": "hi",
                },
            }
        )
        delta = await delta_task
        self.assertEqual((delta.kind, delta.payload["text"]), ("assistant.delta", "hi"))

        completed_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        server.notifications["thread-1"][0].put_nowait(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {"id": "turn-1", "status": "completed"},
                },
            }
        )
        completed = await completed_task
        self.assertEqual(completed.kind, "assistant.completed")
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

    async def test_stream_turn_starts_unmaterialized_thread(self):
        server = CodexAppServer(cwd=Path("."))
        requests = []

        async def read_thread(_thread):
            raise AppServerError("thread/read: thread is not materialized yet")

        server.read_thread = read_thread

        async def start(method, params):
            requests.append((method, params))
            if method == "turn/start":
                return {"turn": {"id": "turn-first"}}
            raise AssertionError(method)

        server.request = start
        stream = server.stream_turn("thread-new", "first message")
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        server.notifications["thread-new"][0].put_nowait(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-new",
                    "turnId": "turn-first",
                    "item": {
                        "id": "reply-first",
                        "type": "agentMessage",
                        "text": "ready",
                    },
                },
            }
        )
        delta = await task
        self.assertEqual(delta.payload["text"], "ready")
        completed_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        server.notifications["thread-new"][0].put_nowait(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-new",
                    "turn": {"id": "turn-first", "status": "completed"},
                },
            }
        )
        completed = await completed_task
        self.assertEqual(completed.kind, "assistant.completed")

    async def test_stream_turn_starts_ephemeral_thread_without_turn_read(self):
        server = CodexAppServer(cwd=Path("."))
        requests = []

        async def read_thread(_thread):
            raise AppServerError(
                "thread/read: ephemeral threads do not support includeTurns"
            )

        server.read_thread = read_thread

        async def request(method, params):
            requests.append((method, params))
            if method == "turn/start":
                return {"turn": {"id": "turn-live"}}
            raise AssertionError(method)

        server.request = request
        stream = server.stream_turn("thread-live", "speak now", effort="low")
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        server.notifications["thread-live"][0].put_nowait(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-live",
                    "turn": {"id": "turn-live", "status": "completed"},
                },
            }
        )
        with self.assertRaisesRegex(
            AppServerError, "completed without assistant text"
        ):
            await task
        self.assertEqual(
            requests,
            [
                (
                    "turn/start",
                    {
                        "threadId": "thread-live",
                        "input": [{"type": "text", "text": "speak now"}],
                        "effort": "low",
                    },
                )
            ],
        )

    async def test_stream_turn_recovers_coalesced_completed_item_text(self):
        server = CodexAppServer(cwd=Path("."))

        async def request(method, _params):
            if method == "thread/read":
                return {"thread": {"id": "thread-1", "turns": []}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            raise AssertionError(method)

        server.request = request
        stream = server.stream_turn("thread-1", "hello")
        delta_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        server.notifications["thread-1"][0].put_nowait(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {
                        "id": "reply-1",
                        "type": "agentMessage",
                        "text": "coalesced reply",
                    },
                },
            }
        )
        delta = await delta_task
        self.assertEqual(delta.payload["text"], "coalesced reply")
        self.assertTrue(delta.payload["recovered"])
        completed = await anext(stream)
        self.assertEqual(completed.kind, "assistant.completed")

    async def test_start_thread_marks_live_lane_ephemeral(self):
        server = CodexAppServer(cwd=Path("/workspace/product"))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            return {"thread": {"id": "thread-live"}}

        server.request = request
        thread = await server.start_thread(
            developer_instructions="voice lane",
            ephemeral=True,
        )
        self.assertEqual(thread, "thread-live")
        self.assertEqual(requests[0][0], "thread/start")
        self.assertTrue(requests[0][1]["ephemeral"])

    async def test_start_thread_pins_execution_sandbox_and_reviewer(self):
        server = CodexAppServer(cwd=Path("/workspace/product"))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            return {"thread": {"id": "thread-executor"}}

        server.request = request
        thread = await server.start_thread(
            developer_instructions="execute one approved issue",
            sandbox="workspace-write",
            approval_policy="on-request",
            approvals_reviewer="auto_review",
            service_name="zer0-pm-executor",
        )
        self.assertEqual(thread, "thread-executor")
        self.assertEqual(
            requests[0],
            (
                "thread/start",
                {
                    "cwd": "/workspace/product",
                    "developerInstructions": "execute one approved issue",
                    "sandbox": "workspace-write",
                    "approvalPolicy": "on-request",
                    "approvalsReviewer": "auto_review",
                    "serviceName": "zer0-pm-executor",
                },
            ),
        )

    async def test_stream_turn_passes_structured_output_schema(self):
        server = CodexAppServer(cwd=Path("."))
        requests = []

        async def read_thread(_thread):
            return {"id": "thread-1", "turns": []}

        async def request(method, params):
            requests.append((method, params))
            return {"turn": {"id": "turn-1"}}

        server.read_thread = read_thread
        server.request = request
        schema = {
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "string"}},
        }
        stream = server.stream_turn(
            "thread-1",
            "execute",
            effort="high",
            output_schema=schema,
        )
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        self.assertEqual(requests[0][1]["outputSchema"], schema)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)

    async def test_execution_stream_waits_for_turn_and_returns_last_agent_message(self):
        server = CodexAppServer(cwd=Path("."))

        async def read_thread(_thread):
            return {"id": "thread-1", "turns": []}

        async def request(method, _params):
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            raise AssertionError(method)

        server.read_thread = read_thread
        server.request = request
        stream = server.stream_turn(
            "thread-1",
            "execute",
            wait_for_turn_completed=True,
        )
        result_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        queue = server.notifications["thread-1"][0]
        queue.put_nowait(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {
                        "id": "preamble",
                        "type": "agentMessage",
                        "text": "I will inspect it.",
                    },
                },
            }
        )
        await asyncio.sleep(0)
        self.assertFalse(result_task.done())
        queue.put_nowait(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {
                        "id": "final",
                        "type": "agentMessage",
                        "text": '{"status":"completed"}',
                    },
                },
            }
        )
        queue.put_nowait(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {"id": "turn-1", "status": "completed"},
                },
            }
        )
        result = await result_task
        self.assertEqual(result.payload["text"], '{"status":"completed"}')
        self.assertTrue(result.payload["final"])
        completed = await anext(stream)
        self.assertEqual(completed.kind, "assistant.completed")

    async def test_fork_thread_inherits_authoritative_history_ephemerally(self):
        server = CodexAppServer(cwd=Path("/workspace/product"))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            return {"thread": {"id": "thread-live-fork"}}

        server.request = request
        thread = await server.fork_thread(
            "thread-authoritative",
            model="voice-model",
            developer_instructions="voice lane",
        )
        self.assertEqual(thread, "thread-live-fork")
        self.assertEqual(
            requests,
            [
                (
                    "thread/fork",
                    {
                        "threadId": "thread-authoritative",
                        "cwd": "/workspace/product",
                        "ephemeral": True,
                        "model": "voice-model",
                        "developerInstructions": "voice lane",
                    },
                )
            ],
        )

    async def test_fork_thread_retries_without_transient_rollout_boundary(self):
        server = CodexAppServer(cwd=Path("/workspace/product"))
        requests = []

        async def request(method, params):
            requests.append((method, dict(params)))
            if len(requests) == 1:
                raise AppServerError(
                    "thread/fork: lastTurnId 'rollout-42' is not a persisted "
                    "canonical turn in the source thread"
                )
            return {"thread": {"id": "thread-canonical-fork"}}

        server.request = request
        thread = await server.fork_thread(
            "thread-authoritative",
            last_turn_id="rollout-42",
        )
        self.assertEqual(thread, "thread-canonical-fork")
        self.assertEqual(requests[0][1]["lastTurnId"], "rollout-42")
        self.assertNotIn("lastTurnId", requests[1][1])

    async def test_stream_turn_steers_busy_thread_and_finishes_agent_item(self):
        server = CodexAppServer(cwd=Path("."))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            if method == "thread/read":
                return {
                    "thread": {
                        "id": "thread-1",
                        "turns": [
                            {"id": "turn-active", "status": "inProgress"}
                        ],
                    }
                }
            if method == "turn/steer":
                return {}
            raise AssertionError(method)

        server.request = request
        stream = server.stream_turn("thread-1", "new voice input")
        delta_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        client_id = requests[-1][1]["clientUserMessageId"]
        queue = server.notifications["thread-1"][0]
        queue.put_nowait(
            {
                "method": "item/started",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-active",
                    "item": {
                        "id": "user-voice",
                        "type": "userMessage",
                        "clientId": client_id,
                    },
                },
            }
        )
        queue.put_nowait(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-active",
                    "itemId": "agent-response",
                    "delta": "heard",
                },
            }
        )
        delta = await delta_task
        self.assertEqual(delta.payload["text"], "heard")

        completed_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        queue.put_nowait(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-active",
                    "completedAtMs": 1,
                    "item": {
                        "id": "agent-response",
                        "type": "agentMessage",
                        "text": "heard",
                    },
                },
            }
        )
        completed = await completed_task
        self.assertEqual(completed.kind, "assistant.completed")
        self.assertTrue(completed.payload["steered"])
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertEqual(requests[1][0], "turn/steer")
        self.assertEqual(
            requests[1][1]["expectedTurnId"],
            "turn-active",
        )

    async def test_stream_turn_survives_steer_race_and_steers_successor(self):
        server = CodexAppServer(cwd=Path("."))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            if method == "thread/read":
                turn = "turn-a" if len(requests) == 1 else "turn-b"
                return {
                    "thread": {
                        "id": "thread-1",
                        "turns": [{"id": turn, "status": "inProgress"}],
                    }
                }
            if method == "turn/steer":
                if params["expectedTurnId"] == "turn-a":
                    raise AppServerError(
                        "turn/steer: {'code': -32600, 'message': 'expected "
                        "active turn id `turn-a` but found `turn-b`'}"
                    )
                return {}
            raise AssertionError(method)

        server.request = request
        stream = server.stream_turn("thread-1", "new voice input")
        delta_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        self.assertEqual(
            [method for method, _ in requests],
            ["thread/read", "turn/steer", "thread/read", "turn/steer"],
        )
        self.assertEqual(requests[-1][1]["expectedTurnId"], "turn-b")
        client_id = requests[-1][1]["clientUserMessageId"]
        queue = server.notifications["thread-1"][0]
        queue.put_nowait(
            {
                "method": "item/started",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-b",
                    "item": {
                        "id": "user-voice",
                        "type": "userMessage",
                        "clientId": client_id,
                    },
                },
            }
        )
        queue.put_nowait(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-b",
                    "itemId": "agent-response",
                    "delta": "still here",
                },
            }
        )
        delta = await delta_task
        self.assertEqual(delta.payload["text"], "still here")
        completed_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        queue.put_nowait(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-b",
                    "completedAtMs": 1,
                    "item": {
                        "id": "agent-response",
                        "type": "agentMessage",
                        "text": "still here",
                    },
                },
            }
        )
        completed = await completed_task
        self.assertEqual(completed.kind, "assistant.completed")
        self.assertTrue(completed.payload["steered"])
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

    async def test_submit_turn_starts_fresh_after_race_when_thread_went_idle(self):
        server = CodexAppServer(cwd=Path("."))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            if method == "thread/read":
                turns = (
                    [{"id": "turn-a", "status": "inProgress"}]
                    if len(requests) == 1
                    else [{"id": "turn-a", "status": "completed"}]
                )
                return {"thread": {"id": "thread-1", "turns": turns}}
            if method == "turn/steer":
                raise AppServerError(
                    "turn/steer: {'code': -32600, 'message': 'expected "
                    "active turn id `turn-a` but found `turn-b`'}"
                )
            if method == "turn/start":
                return {"turn": {"id": "turn-new"}}
            raise AssertionError(method)

        server.request = request
        turn_id = await server.submit_turn("thread-1", "follow up")
        self.assertEqual(turn_id, "turn-new")
        self.assertEqual(
            [method for method, _ in requests],
            ["thread/read", "turn/steer", "thread/read", "turn/start"],
        )

    async def test_interrupt_tolerates_already_finished_turn(self):
        server = CodexAppServer(cwd=Path("."))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            raise AppServerError(
                "turn/interrupt: {'code': -32600, 'message': "
                "'no active turn to interrupt'}"
            )

        server.request = request
        await server.interrupt("thread-1", "turn-a")
        self.assertEqual(requests[0][0], "turn/interrupt")

        async def hard_failure(method, params):
            raise AppServerError("turn/interrupt: connection lost")

        server.request = hard_failure
        with self.assertRaisesRegex(AppServerError, "connection lost"):
            await server.interrupt("thread-1", "turn-a")

    async def test_resume_thread_uses_native_thread_id_and_cwd(self):
        server = CodexAppServer(cwd=Path("/workspace/product"))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            return {"thread": {"id": "thread-existing"}}

        server.request = request
        result = await server.resume_thread(
            "thread-existing",
            developer_instructions="voice lane",
        )
        self.assertEqual(result, "thread-existing")
        self.assertEqual(
            requests,
            [
                (
                    "thread/resume",
                    {
                        "threadId": "thread-existing",
                        "cwd": "/workspace/product",
                        "developerInstructions": "voice lane",
                    },
                )
            ],
        )

    async def test_list_threads_filters_to_interactive_cli_project(self):
        server = CodexAppServer(cwd=Path("/workspace/default"))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            return {"data": [{"id": "thread-newest"}], "nextCursor": None}

        server.request = request
        threads = await server.list_threads(Path("/workspace/project"), limit=7)
        self.assertEqual(threads, [{"id": "thread-newest"}])
        self.assertEqual(
            requests,
            [
                (
                    "thread/list",
                    {
                        "cwd": "/workspace/project",
                        "limit": 7,
                        "sortKey": "recency_at",
                        "sortDirection": "desc",
                        "sourceKinds": ["cli"],
                    },
                )
            ],
        )

    async def test_read_thread_requests_turns(self):
        server = CodexAppServer(cwd=Path("/workspace/default"))
        requests = []

        async def request(method, params):
            requests.append((method, params))
            return {"thread": {"id": "thread-1", "turns": []}}

        server.request = request
        thread = await server.read_thread("thread-1")
        self.assertEqual(thread["id"], "thread-1")
        self.assertEqual(
            requests,
            [
                (
                    "thread/read",
                    {"threadId": "thread-1", "includeTurns": True},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
