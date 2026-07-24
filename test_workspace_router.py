import asyncio
import unittest
from pathlib import Path

from duplex import HarnessRouter, codex_thread_context
from workspace_router import Resolution, Route, resolve_context


ROUTES = {
    "pm": Path("/workspace/zer0/products/pm"),
    "zerOS": Path("/workspace/zer0/products/zerOS"),
}


def context(*windows, focus=None):
    value = {
        "tmux": [{"session": "0", "windows": list(windows)}],
    }
    if focus:
        value["tmux_focus"] = {"pane_id": focus}
    return value


def window(index, project, pane, *, active_pane=True, active_window=False):
    return {
        "index": str(index),
        "active": active_window,
        "panes": [
            {
                "id": pane,
                "index": "1",
                "active": active_pane,
                "dead": False,
                "command": "codex",
                "project": project,
            }
        ],
    }


class WorkspaceRouterTests(unittest.TestCase):
    def test_unique_active_harness_routes_without_guessing(self):
        result = resolve_context(context(window(1, "pm", "%1")), ROUTES)
        self.assertEqual(result.reason, "unique_active_harness")
        self.assertEqual(result.route.project, "pm")

    def test_focused_pane_wins_across_multiple_active_windows(self):
        result = resolve_context(
            context(
                window(1, "pm", "%1"),
                window(2, "zerOS", "%2"),
                focus="%2",
            ),
            ROUTES,
        )
        self.assertEqual(result.reason, "focused_pane")
        self.assertEqual(result.route.project, "zerOS")

    def test_active_window_field_is_supported_when_sensor_exposes_it(self):
        result = resolve_context(
            context(
                window(1, "pm", "%1"),
                window(2, "zerOS", "%2", active_window=True),
            ),
            ROUTES,
        )
        self.assertEqual(result.reason, "active_window")
        self.assertEqual(result.route.pane_id, "%2")

    def test_ambiguous_windows_fail_closed(self):
        result = resolve_context(
            context(window(1, "pm", "%1"), window(2, "zerOS", "%2")),
            ROUTES,
        )
        self.assertIsNone(result.route)
        self.assertEqual(result.reason, "ambiguous_active_windows")
        self.assertEqual({item.project for item in result.candidates}, {"pm", "zerOS"})

    def test_codex_thread_context_keeps_recent_conversation_only(self):
        thread = {
            "turns": [
                {
                    "items": [
                        {
                            "type": "userMessage",
                            "content": [{"type": "text", "text": "old question"}],
                        },
                        {"type": "commandExecution", "command": "secret raw command"},
                        {"type": "agentMessage", "text": "old answer"},
                    ]
                },
                {
                    "items": [
                        {
                            "type": "userMessage",
                            "content": [{"type": "text", "text": "new question"}],
                        },
                        {"type": "agentMessage", "text": "new answer"},
                    ]
                },
            ]
        }
        result = codex_thread_context(thread, messages=2)
        self.assertEqual(
            result,
            ("user: new question", "assistant: new answer"),
        )
        self.assertNotIn("secret raw command", " ".join(result))


class HarnessRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_focus_switch_gets_distinct_cached_harness_threads(self):
        routes = [
            Route("pm", ROUTES["pm"], "%1", "0", "1"),
            Route("zerOS", ROUTES["zerOS"], "%2", "0", "2"),
            Route("pm", ROUTES["pm"], "%1", "0", "1"),
        ]

        class Workspace:
            def resolve(self):
                route = routes.pop(0)
                return Resolution(route, "focused_pane", (route,))

        class Server:
            def __init__(self):
                self.resumed = []

            async def list_threads(self, cwd):
                return [{"id": f"native-{cwd.name}"}]

            async def resume_thread(self, thread, **kwargs):
                self.resumed.append((thread, kwargs["cwd"]))
                return thread

            async def read_thread(self, _thread):
                return {"turns": []}

            async def start_thread(self, **_kwargs):
                raise AssertionError("existing harness thread should be resumed")

        server = Server()
        router = HarnessRouter(
            server,
            Workspace(),
            "fallback",
            instructions="voice",
            timeout=1,
        )
        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        from unittest.mock import patch

        with patch("duplex.asyncio.to_thread", new=inline):
            first = await router.resolve()
            second = await router.resolve()
            third = await router.resolve()
        self.assertEqual((first.key, second.key, third.key), ("pm", "zerOS", "pm"))
        self.assertEqual(third.thread, first.thread)
        self.assertEqual(
            server.resumed,
            [
                ("native-pm", ROUTES["pm"]),
                ("native-zerOS", ROUTES["zerOS"]),
            ],
        )

    async def test_spoken_pin_routes_until_follow_focus(self):
        pm = Route("pm", ROUTES["pm"], "%1", "0", "1")

        class Workspace:
            routes = ROUTES

            def resolve(self):
                return Resolution(pm, "focused_pane", (pm,))

        class Server:
            async def list_threads(self, cwd):
                return [{"id": f"native-{cwd.name}"}]

            async def resume_thread(self, thread, **_kwargs):
                return thread

            async def read_thread(self, _thread):
                return {"turns": []}

            async def start_thread(self, **_kwargs):
                raise AssertionError

        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        router = HarnessRouter(
            Server(),
            Workspace(),
            "fallback",
            instructions="voice",
            timeout=1,
        )
        from unittest.mock import patch

        with patch("duplex.asyncio.to_thread", new=inline):
            pinned = await router.resolve("switch to zerOS")
            still_pinned = await router.resolve("what are we building")
            focused = await router.resolve("follow focus")
        self.assertEqual(pinned.key, "zerOS")
        self.assertEqual(pinned.reason, "spoken_pin")
        self.assertEqual(still_pinned.key, "zerOS")
        self.assertEqual(focused.key, "pm")

    async def test_slow_thread_context_never_blocks_live_route(self):
        pm = Route("pm", ROUTES["pm"], "%1", "0", "1")

        class Workspace:
            routes = ROUTES

            def resolve(self):
                return Resolution(pm, "focused_pane", (pm,))

        class Server:
            async def list_threads(self, _cwd):
                return [{"id": "native-pm"}]

            async def resume_thread(self, thread, **_kwargs):
                return thread

            async def read_thread(self, _thread):
                await asyncio.sleep(1)
                return {"turns": []}

            async def start_thread(self, **_kwargs):
                raise AssertionError

        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        router = HarnessRouter(
            Server(),
            Workspace(),
            "fallback",
            instructions="voice",
            timeout=0.01,
        )
        from unittest.mock import patch

        with patch("duplex.asyncio.to_thread", new=inline):
            started = asyncio.get_running_loop().time()
            binding = await router.resolve("hello")
            elapsed = asyncio.get_running_loop().time() - started
        self.assertEqual(binding.key, "pm")
        self.assertEqual(binding.context, ())
        self.assertLess(elapsed, 0.1)


if __name__ == "__main__":
    unittest.main()
