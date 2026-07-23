import asyncio
import time
import unittest

from fleet import (
    Horizon,
    LaneResult,
    ReasoningFleet,
    Severity,
    default_lane_specs,
)


class Lane:
    def __init__(self, name, horizon, delay, severity=Severity.CONTEXT):
        self.name = name
        self.horizon = horizon
        self.delay = delay
        self.severity = severity
        self.calls = 0

    async def think(self, _text, _context):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return LaneResult(self.name, self.horizon, self.name, self.severity)


class FleetTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_lanes_finish_in_parallel_and_completion_order(self):
        instant = Lane("instant", Horizon.INSTANT, 0.01, Severity.STEER)
        medium = Lane("medium", Horizon.SHORT, 0.02)
        high = Lane("high", Horizon.MID, 0.03)
        pro = Lane("pro", Horizon.META, 0.04)
        fleet = ReasoningFleet(
            default_lane_specs(
                instant=instant, medium=medium, high=high, pro=pro
            )
        )

        started = time.monotonic()
        updates = [
            update
            async for update in fleet.stream(
                "turn", (), turn_number=24, material_change=False
            )
        ]
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.07)
        self.assertEqual(
            [update.result.lane for update in updates],
            ["instant", "medium", "high", "pro"],
        )
        self.assertEqual(updates[0].result.severity, Severity.STEER)

    async def test_expensive_strategy_lanes_use_cadence(self):
        lanes = {
            "instant": Lane("instant", Horizon.INSTANT, 0),
            "medium": Lane("medium", Horizon.SHORT, 0),
            "high": Lane("high", Horizon.MID, 0),
            "pro": Lane("pro", Horizon.META, 0),
        }
        fleet = ReasoningFleet(default_lane_specs(**lanes))
        updates = [
            update
            async for update in fleet.stream("turn", (), turn_number=1)
        ]
        self.assertEqual(
            {update.result.lane for update in updates}, {"instant", "medium"}
        )

        updates = [
            update
            async for update in fleet.stream(
                "important", (), turn_number=1, material_change=True
            )
        ]
        self.assertEqual(len(updates), 4)


if __name__ == "__main__":
    unittest.main()
