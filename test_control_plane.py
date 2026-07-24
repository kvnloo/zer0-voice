import asyncio
import tempfile
import unittest
from pathlib import Path

from control_plane import ControlServer, VoiceControl, request
from modes import MicMode, NotificationMode, VoiceModes


class VoiceControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_modes_and_push_state_are_independent(self):
        control = VoiceControl(
            VoiceModes(
                mic=MicMode.PUSH_TO_TALK,
                notifications=NotificationMode.UPDATES,
            )
        )
        self.assertFalse(control.state.capture_active)

        pressed = await control.apply({"push_held": True})
        self.assertTrue(pressed.capture_active)
        self.assertEqual(pressed.modes.notifications, NotificationMode.UPDATES)

        muted = await control.apply({"mic": "muted"})
        self.assertFalse(muted.capture_active)
        self.assertTrue(muted.push_held)

    async def test_wait_after_wakes_on_real_change_only(self):
        control = VoiceControl()
        waiter = asyncio.create_task(control.wait_after(0))
        await control.apply({"mic": "continuous"})
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        await control.apply({"mic": "muted"})
        self.assertEqual((await waiter).revision, 1)

    async def test_unix_socket_round_trip_and_private_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.sock"
            control = VoiceControl()
            server = ControlServer(path, control)
            try:
                await server.start()
            except PermissionError:
                self.skipTest("unix sockets unavailable in this environment")
            try:
                result = await request(
                    path,
                    {
                        "mic": "push-to-talk",
                        "notifications": "critical",
                        "push_held": True,
                    },
                )
                self.assertEqual(result["mic"], "push-to-talk")
                self.assertEqual(result["notifications"], "critical")
                self.assertTrue(result["capture_active"])
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            finally:
                await server.close()
            self.assertFalse(path.exists())

    async def test_reload_commands_update_live_model_and_effort(self):
        control = VoiceControl()
        first = await control.apply({"live_model": "model-a", "live_effort": "medium"})
        self.assertEqual(first.live_model, "model-a")
        self.assertEqual(first.live_effort, "medium")
        second = await control.apply({"live_model": None, "live_effort": "high"})
        self.assertIsNone(second.live_model)
        self.assertEqual(second.live_effort, "high")

    async def test_control_events_reflect_live_lane_fields(self):
        control = VoiceControl()
        await control.apply({"live_model": "model-b", "live_effort": "low"})
        state = await control.wait_after(0)
        self.assertEqual(state.live_model, "model-b")
        self.assertEqual(state.live_effort, "low")


if __name__ == "__main__":
    unittest.main()
