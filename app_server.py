"""Small async client for the installed Codex app-server JSONL protocol."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import struct
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from events import Event


class AppServerError(RuntimeError):
    pass

STEER_RACE_MARKER = "expected active turn id"


def _active_turn(thread: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            turn
            for turn in reversed(thread.get("turns", []))
            if turn.get("status") == "inProgress"
        ),
        None,
    )


class UnixWebSocket:
    """Minimal RFC 6455 text client for Codex's local Unix-socket transport."""

    GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, path: Path):
        self.path = path
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        self.reader, self.writer = await asyncio.open_unix_connection(self.path)
        key = base64.b64encode(os.urandom(16))
        request = (
            b"GET / HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Version: 13\r\n"
            b"Sec-WebSocket-Key: " + key + b"\r\n\r\n"
        )
        self.writer.write(request)
        await self.writer.drain()
        response = await self.reader.readuntil(b"\r\n\r\n")
        if not response.startswith(b"HTTP/1.1 101 "):
            raise AppServerError(
                "app-server WebSocket upgrade failed: "
                + response.split(b"\r\n", 1)[0].decode(errors="replace")
            )
        headers = {}
        for line in response.split(b"\r\n")[1:]:
            if b":" in line:
                name, value = line.split(b":", 1)
                headers[name.strip().lower()] = value.strip()
        expected = base64.b64encode(hashlib.sha1(key + self.GUID).digest())
        if headers.get(b"sec-websocket-accept") != expected:
            raise AppServerError("app-server WebSocket accept key mismatch")

    async def send(self, text: str, *, opcode: int = 1) -> None:
        if not self.writer:
            raise AppServerError("app-server WebSocket is not connected")
        payload = text.encode()
        mask = os.urandom(4)
        size = len(payload)
        header = bytearray([0x80 | opcode])
        if size < 126:
            header.append(0x80 | size)
        elif size < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", size))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", size))
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.writer.write(bytes(header) + mask + masked)
        await self.writer.drain()

    async def receive(self) -> str | None:
        if not self.reader:
            return None
        while True:
            try:
                first, second = await self.reader.readexactly(2)
            except asyncio.IncompleteReadError:
                return None
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            size = second & 0x7F
            if size == 126:
                size = struct.unpack("!H", await self.reader.readexactly(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", await self.reader.readexactly(8))[0]
            mask = await self.reader.readexactly(4) if masked else b""
            payload = await self.reader.readexactly(size)
            if masked:
                payload = bytes(
                    byte ^ mask[index % 4] for index, byte in enumerate(payload)
                )
            if opcode == 8:
                return None
            if opcode == 9:
                await self.send(payload.decode(errors="replace"), opcode=10)
                continue
            if opcode == 1:
                return payload.decode()

    async def close(self) -> None:
        if self.writer:
            try:
                await self.send("", opcode=8)
            except (ConnectionError, AppServerError):
                pass
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except ConnectionError:
                pass
        self.reader = self.writer = None


class CodexAppServer:
    def __init__(
        self,
        *,
        cwd: Path,
        shared: bool = False,
        startup_timeout: float = 8.0,
    ):
        self.cwd = cwd
        self.shared = shared
        self.startup_timeout = startup_timeout
        self.process: asyncio.subprocess.Process | None = None
        self.reader: asyncio.Task | None = None
        self.stderr: asyncio.Task | None = None
        self.pending: dict[int, asyncio.Future] = {}
        self.notifications: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self.request_id = 0
        self.sequence = 0
        self.stderr_lines: list[str] = []
        self.protocol_methods: deque[str] = deque(maxlen=200)
        self.websocket: UnixWebSocket | None = None

    def _fail_waiters(self, error: AppServerError) -> None:
        """Wake every RPC and stream consumer when the transport disappears."""
        for future in tuple(self.pending.values()):
            if not future.done():
                future.set_exception(error)
        self.pending.clear()
        for queues in tuple(self.notifications.values()):
            for queue in tuple(queues):
                # A disconnected stream cannot complete successfully. Drop any
                # stale deltas so consumers observe the failure immediately.
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                queue.put_nowait(error)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def start(self) -> None:
        if self.process is not None:
            return
        if self.shared:
            version = await asyncio.create_subprocess_exec(
                "codex",
                "app-server",
                "daemon",
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await version.communicate()
            if version.returncode:
                raise AppServerError(
                    "could not locate managed app-server socket: "
                    + stderr.decode(errors="replace").strip()
                )
            socket_path = json.loads(stdout)["socketPath"]
            self.websocket = UnixWebSocket(Path(socket_path))
            await self.websocket.connect()
            self.reader = asyncio.create_task(self._read())
        else:
            command = ("codex", "app-server", "--listen", "stdio://")
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.reader = asyncio.create_task(self._read())
            self.stderr = asyncio.create_task(self._read_stderr())
        try:
            await asyncio.wait_for(
                self.request(
                    "initialize",
                    {
                        "clientInfo": {
                            "name": "zer0_company_os",
                            "title": "Zer0 Company OS",
                            "version": "0.1.0",
                        }
                    },
                ),
                timeout=self.startup_timeout,
            )
        except TimeoutError as exc:
            detail = "; ".join(self.stderr_lines[-3:])
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            elif self.process and self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=1)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            for task in (self.reader, self.stderr):
                if task:
                    task.cancel()
            self.process = None
            raise AppServerError(
                "app-server initialization timed out"
                + (f": {detail}" if detail else "")
            ) from exc
        await self.send({"method": "initialized", "params": {}})

    async def close(self) -> None:
        self._fail_waiters(AppServerError("app-server client closed"))
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            for task in (self.reader, self.stderr):
                if task:
                    task.cancel()
            return
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
            try:
                await asyncio.wait_for(
                    self.process.stdin.wait_closed(),
                    timeout=1,
                )
            except (TimeoutError, BrokenPipeError):
                pass
        try:
            await asyncio.wait_for(self.process.wait(), timeout=3)
        except asyncio.TimeoutError:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=1)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        for task in (self.reader, self.stderr):
            if task:
                task.cancel()
        self.process = None

    async def send(self, message: dict[str, Any]) -> None:
        if self.websocket:
            await self.websocket.send(json.dumps(message, separators=(",", ":")))
            return
        if not self.process or not self.process.stdin:
            raise AppServerError("app-server is not running")
        self.process.stdin.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
        await self.process.stdin.drain()

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.request_id += 1
        request_id = self.request_id
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self.send({"method": method, "id": request_id, "params": params})
            response = await future
        finally:
            self.pending.pop(request_id, None)
        if "error" in response:
            raise AppServerError(f"{method}: {response['error']}")
        return response["result"]

    async def start_thread(
        self,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        developer_instructions: str | None = None,
        ephemeral: bool = False,
        sandbox: str | None = None,
        approval_policy: str | None = None,
        approvals_reviewer: str | None = None,
        service_name: str | None = None,
    ) -> str:
        params: dict[str, Any] = {"cwd": str(cwd or self.cwd)}
        if model:
            params["model"] = model
        if developer_instructions:
            params["developerInstructions"] = developer_instructions
        if ephemeral:
            params["ephemeral"] = True
        if sandbox:
            params["sandbox"] = sandbox
        if approval_policy:
            params["approvalPolicy"] = approval_policy
        if approvals_reviewer:
            params["approvalsReviewer"] = approvals_reviewer
        if service_name:
            params["serviceName"] = service_name
        result = await self.request("thread/start", params)
        return result["thread"]["id"]

    async def fork_thread(
        self,
        thread_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        developer_instructions: str | None = None,
        ephemeral: bool = True,
        last_turn_id: str | None = None,
    ) -> str:
        """Branch an existing conversation so a realtime lane inherits history."""
        params: dict[str, Any] = {
            "threadId": thread_id,
            "cwd": str(cwd or self.cwd),
            "ephemeral": ephemeral,
        }
        if model:
            params["model"] = model
        if developer_instructions:
            params["developerInstructions"] = developer_instructions
        if last_turn_id:
            params["lastTurnId"] = last_turn_id
        try:
            result = await self.request("thread/fork", params)
        except AppServerError as error:
            # Managed app-server can expose transient rollout ids from
            # thread/read while thread/fork accepts only persisted canonical
            # turn ids. In that exact case, let the server choose the latest
            # persisted boundary rather than making voice unavailable.
            if not (
                last_turn_id
                and "lastTurnId" in str(error)
                and "not a persisted canonical turn" in str(error)
            ):
                raise
            params.pop("lastTurnId")
            result = await self.request("thread/fork", params)
        return result["thread"]["id"]

    async def submit_turn(
        self,
        thread_id: str,
        text: str,
        *,
        effort: str | None = None,
    ) -> str:
        """Submit/steer user input without waiting for an assistant response."""
        try:
            thread = await self.read_thread(thread_id)
        except AppServerError as error:
            if not any(
                marker in str(error)
                for marker in ("not materialized yet", "no rollout found")
            ):
                raise
            thread = {"turns": []}
        active = _active_turn(thread)
        if active:
            try:
                await self.steer(
                    thread_id,
                    active["id"],
                    text,
                    client_user_message_id=f"voice-{uuid.uuid4()}",
                )
                return active["id"]
            except AppServerError as error:
                if STEER_RACE_MARKER not in str(error):
                    raise
                # The turn we read completed before turn/steer landed. Steer
                # its successor once, or fall through to a fresh turn.
                thread = await self.read_thread(thread_id)
                active = _active_turn(thread)
                if active:
                    await self.steer(
                        thread_id,
                        active["id"],
                        text,
                        client_user_message_id=f"voice-{uuid.uuid4()}",
                    )
                    return active["id"]
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": text}],
        }
        if effort:
            params["effort"] = effort
        result = await self.request("turn/start", params)
        return result["turn"]["id"]

    async def resume_thread(
        self,
        thread_id: str,
        *,
        cwd: Path | None = None,
        developer_instructions: str | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "cwd": str(cwd or self.cwd),
        }
        if developer_instructions:
            params["developerInstructions"] = developer_instructions
        result = await self.request("thread/resume", params)
        return result["thread"]["id"]

    async def list_threads(self, cwd: Path, *, limit: int = 20) -> list[dict[str, Any]]:
        result = await self.request(
            "thread/list",
            {
                "cwd": str(cwd),
                "limit": limit,
                "sortKey": "recency_at",
                "sortDirection": "desc",
                "sourceKinds": ["cli"],
            },
        )
        return result.get("data", [])

    async def read_thread(self, thread_id: str) -> dict[str, Any]:
        result = await self.request(
            "thread/read",
            {"threadId": thread_id, "includeTurns": True},
        )
        return result["thread"]

    async def stream_turn(
        self,
        thread_id: str,
        text: str,
        *,
        effort: str | None = None,
        output_schema: dict[str, Any] | None = None,
        wait_for_turn_completed: bool = False,
    ) -> AsyncIterator[Event]:
        queue: asyncio.Queue = asyncio.Queue()
        self.notifications[thread_id].append(queue)
        try:
            try:
                thread = await self.read_thread(thread_id)
            except AppServerError as error:
                if not any(
                    marker in str(error)
                    for marker in (
                        "not materialized yet",
                        "no rollout found",
                        "ephemeral threads do not support includeTurns",
                    )
                ):
                    raise
                thread = {"id": thread_id, "turns": []}
            active = _active_turn(thread)
            steering = active is not None
            client_message_id = f"voice-{uuid.uuid4()}" if steering else None
            if steering:
                turn_id = active["id"]
                try:
                    await self.steer(
                        thread_id,
                        turn_id,
                        text,
                        client_user_message_id=client_message_id,
                    )
                except AppServerError as error:
                    if STEER_RACE_MARKER not in str(error):
                        raise
                    # The turn we read completed before turn/steer landed.
                    # Re-resolve once: steer the successor, else start fresh
                    # instead of killing the live lane.
                    thread = await self.read_thread(thread_id)
                    active = _active_turn(thread)
                    if active is not None:
                        turn_id = active["id"]
                        await self.steer(
                            thread_id,
                            turn_id,
                            text,
                            client_user_message_id=client_message_id,
                        )
                    else:
                        steering = False
                        client_message_id = None
            if not steering:
                params: dict[str, Any] = {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": text}],
                }
                if effort:
                    params["effort"] = effort
                if output_schema:
                    params["outputSchema"] = output_schema
                result = await self.request("turn/start", params)
                turn_id = result["turn"]["id"]
            steering_seen = not steering
            response_item: str | None = None
            output_chars = 0
            item_text: dict[str, list[str]] = defaultdict(list)
            last_completed_text = ""
            while True:
                message = await queue.get()
                if isinstance(message, AppServerError):
                    raise message
                method = message.get("method", "")
                params = message.get("params", {})
                event_turn = params.get("turnId") or params.get("turn", {}).get("id")
                if event_turn != turn_id:
                    continue
                item = params.get("item", {})
                if (
                    steering
                    and method in ("item/started", "item/completed")
                    and item.get("type") == "userMessage"
                    and item.get("clientId") == client_message_id
                ):
                    steering_seen = True
                    continue
                if method == "item/agentMessage/delta":
                    if not steering_seen:
                        continue
                    item_id = params.get("itemId")
                    delta = str(params.get("delta", ""))
                    if wait_for_turn_completed:
                        if item_id and delta:
                            item_text[item_id].append(delta)
                        continue
                    if response_item is None:
                        response_item = item_id
                    if item_id != response_item:
                        continue
                    if not delta:
                        continue
                    output_chars += len(delta)
                    self.sequence += 1
                    yield Event(
                        source="codex",
                        kind="assistant.delta",
                        subject=f"turn:{turn_id}",
                        payload={"text": delta, "thread_id": thread_id},
                        seq=self.sequence,
                    )
                elif (
                    method == "item/completed"
                    and item.get("type") == "agentMessage"
                ):
                    item_id = item.get("id")
                    completed_text = str(item.get("text", ""))
                    if wait_for_turn_completed:
                        last_completed_text = completed_text or "".join(
                            item_text.get(item_id, ())
                        )
                        continue
                    if response_item is None:
                        response_item = item_id
                    if item_id != response_item or not steering_seen:
                        continue
                    # Some app-server transports coalesce the assistant reply
                    # into item/completed and send no delta notifications.
                    # Recover that text once; never report an empty completion.
                    if output_chars == 0 and completed_text:
                        output_chars = len(completed_text)
                        self.sequence += 1
                        yield Event(
                            source="codex",
                            kind="assistant.delta",
                            subject=f"turn:{turn_id}",
                            payload={
                                "text": completed_text,
                                "thread_id": thread_id,
                                "recovered": True,
                            },
                            seq=self.sequence,
                        )
                    if output_chars == 0:
                        raise AppServerError(
                            f"turn {turn_id} completed without assistant text"
                        )
                    self.sequence += 1
                    yield Event(
                        source="codex",
                        kind="assistant.completed",
                        subject=f"turn:{turn_id}",
                        payload={
                            "thread_id": thread_id,
                            "status": "completed",
                            "steered": steering,
                        },
                        seq=self.sequence,
                    )
                    return
                elif method == "turn/completed":
                    status = params["turn"].get("status", "completed")
                    if status != "completed":
                        raise AppServerError(
                            f"turn {turn_id} finished with status {status}"
                        )
                    if wait_for_turn_completed and last_completed_text:
                        output_chars = len(last_completed_text)
                        self.sequence += 1
                        yield Event(
                            source="codex",
                            kind="assistant.delta",
                            subject=f"turn:{turn_id}",
                            payload={
                                "text": last_completed_text,
                                "thread_id": thread_id,
                                "final": True,
                            },
                            seq=self.sequence,
                        )
                    if output_chars == 0:
                        raise AppServerError(
                            f"turn {turn_id} completed without assistant text"
                        )
                    self.sequence += 1
                    yield Event(
                        source="codex",
                        kind="assistant.completed",
                        subject=f"turn:{turn_id}",
                        payload={
                            "thread_id": thread_id,
                            "status": status,
                        },
                        seq=self.sequence,
                    )
                    return
        finally:
            self.notifications[thread_id].remove(queue)

    async def steer(
        self,
        thread_id: str,
        turn_id: str,
        text: str,
        *,
        client_user_message_id: str | None = None,
    ) -> None:
        params = {
            "threadId": thread_id,
            "expectedTurnId": turn_id,
            "input": [{"type": "text", "text": text}],
        }
        if client_user_message_id:
            params["clientUserMessageId"] = client_user_message_id
        await self.request("turn/steer", params)

    async def interrupt(self, thread_id: str, turn_id: str) -> None:
        try:
            await self.request(
                "turn/interrupt",
                {"threadId": thread_id, "turnId": turn_id},
            )
        except AppServerError as error:
            # Interrupting a turn that already finished reached the desired
            # state; killing the live lane over it created error churn.
            if "no active turn" not in str(error):
                raise

    async def _read(self) -> None:
        failure = AppServerError("app-server connection closed")
        try:
            while True:
                if self.websocket:
                    raw = await self.websocket.receive()
                    if raw is None:
                        return
                    message = json.loads(raw)
                else:
                    assert self.process and self.process.stdout
                    line = await self.process.stdout.readline()
                    if not line:
                        return
                    message = json.loads(line)
                request_id = message.get("id")
                if method := message.get("method"):
                    self.protocol_methods.append(str(method))
                if request_id is not None and ("result" in message or "error" in message):
                    future = self.pending.pop(request_id, None)
                    if future and not future.done():
                        future.set_result(message)
                    continue
                thread_id = message.get("params", {}).get("threadId")
                if thread_id:
                    for queue in tuple(self.notifications.get(thread_id, ())):
                        queue.put_nowait(message)
        except asyncio.CancelledError:
            failure = AppServerError("app-server reader cancelled")
            raise
        except Exception as error:
            failure = AppServerError(
                f"app-server reader failed ({type(error).__name__})"
            )
        finally:
            self._fail_waiters(failure)

    async def _read_stderr(self) -> None:
        assert self.process and self.process.stderr
        while line := await self.process.stderr.readline():
            self.stderr_lines.append(line.decode(errors="replace").rstrip())
            self.stderr_lines[:] = self.stderr_lines[-50:]
