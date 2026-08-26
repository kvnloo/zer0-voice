import asyncio
import json
import unittest
from unittest.mock import patch

from providers import CodexSubscription, EmptyCompletionError, OpenAICompatible


class Response:
    status = 200

    def __init__(self, lines):
        self.lines, self.closed = lines, False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


class Server:
    async def stream_turn(self, thread, text, effort=None):
        self.started = (thread, text, effort)
        yield type(
            "Event",
            (),
            {
                "subject": "turn:t1",
                "kind": "assistant.delta",
                "payload": {"text": "hello"},
            },
        )
        yield type(
            "Event",
            (),
            {
                "subject": "turn:t1",
                "kind": "assistant.completed",
                "payload": {},
            },
        )

    async def interrupt(self, thread, turn):
        self.interrupted = (thread, turn)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_subscription_normalizes_stream_and_interrupt(self):
        server = Server()
        provider = CodexSubscription(server, "thread-1")
        output = [
            event
            async for event in provider.stream(
                "question", context=("context",), effort="high"
            )
        ]
        self.assertEqual([event.kind for event in output], ["delta", "completed"])
        self.assertEqual(output[0].text, "hello")
        self.assertEqual(server.started, ("thread-1", "context\nquestion", "high"))
        await provider.interrupt("t1")
        self.assertEqual(server.interrupted, ("thread-1", "t1"))

    async def test_codex_subscription_rejects_empty_transport_completion(self):
        class EmptyServer(Server):
            async def stream_turn(self, thread, text, effort=None):
                del thread, text, effort
                yield type(
                    "Event",
                    (),
                    {
                        "subject": "turn:empty",
                        "kind": "assistant.completed",
                        "payload": {},
                    },
                )

        provider = CodexSubscription(EmptyServer(), "thread-1")
        with self.assertRaisesRegex(
            EmptyCompletionError, "completed without assistant text"
        ):
            _ = [event async for event in provider.stream("question")]

    @patch("providers.urllib.request.urlopen")
    async def test_openai_compatible_streams_sse(self, urlopen):
        chunks = [
            {"choices": [{"delta": {"content": "fast "}}]},
            {"choices": [{"delta": {"content": "reply"}}]},
        ]
        urlopen.return_value = Response(
            [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
            + [b"data: [DONE]\n"]
        )
        provider = OpenAICompatible("http://local", "model", api_key="secret")
        output = [event async for event in provider.stream("hello")]
        self.assertEqual(
            [(event.kind, event.text) for event in output],
            [("delta", "fast "), ("delta", "reply"), ("completed", "")],
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://local/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer secret")


if __name__ == "__main__":
    unittest.main()
