"""Deadline-aware parallel intelligence lanes for continuous conversation."""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass
from typing import Protocol


class Horizon(str, enum.Enum):
    INSTANT = "instant"
    SHORT = "short"
    MID = "mid"
    META = "meta"


class Severity(int, enum.Enum):
    CONTEXT = 0
    STEER = 1
    INTERRUPT = 2


@dataclass(frozen=True)
class LaneResult:
    lane: str
    horizon: Horizon
    text: str
    severity: Severity = Severity.CONTEXT
    latency_ms: int = 0


class IntelligenceLane(Protocol):
    async def think(self, text: str, context: tuple[str, ...]) -> LaneResult: ...


@dataclass(frozen=True)
class LaneSpec:
    name: str
    horizon: Horizon
    lane: IntelligenceLane
    cadence: int = 1
    soft_deadline_ms: int | None = None

    def active(self, turn_number: int, material_change: bool) -> bool:
        return self.cadence <= 1 or material_change or turn_number % self.cadence == 0


@dataclass(frozen=True)
class FleetUpdate:
    result: LaneResult
    late: bool


class ReasoningFleet:
    """Run 1..N intelligence lanes concurrently and stream results by completion."""

    def __init__(self, specs: list[LaneSpec]):
        if not specs:
            raise ValueError("reasoning fleet needs at least one lane")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("lane names must be unique")
        self.specs = tuple(specs)

    async def stream(
        self,
        text: str,
        context: tuple[str, ...],
        *,
        turn_number: int,
        material_change: bool = False,
    ):
        active = [
            spec
            for spec in self.specs
            if spec.active(turn_number, material_change)
        ]

        async def run(spec: LaneSpec) -> FleetUpdate:
            started = time.monotonic()
            result = await spec.lane.think(text, context)
            elapsed = round((time.monotonic() - started) * 1000)
            if result.lane != spec.name or result.horizon != spec.horizon:
                result = LaneResult(
                    lane=spec.name,
                    horizon=spec.horizon,
                    text=result.text,
                    severity=result.severity,
                    latency_ms=elapsed,
                )
            elif not result.latency_ms:
                result = LaneResult(
                    lane=result.lane,
                    horizon=result.horizon,
                    text=result.text,
                    severity=result.severity,
                    latency_ms=elapsed,
                )
            late = (
                spec.soft_deadline_ms is not None
                and elapsed > spec.soft_deadline_ms
            )
            return FleetUpdate(result, late)

        tasks = [asyncio.create_task(run(spec)) for spec in active]
        for task in asyncio.as_completed(tasks):
            yield await task


def default_lane_specs(
    *,
    instant: IntelligenceLane,
    medium: IntelligenceLane,
    high: IntelligenceLane,
    pro: IntelligenceLane,
) -> list[LaneSpec]:
    """Default four reasoning lanes; the fifth lane is the live voice generator."""
    return [
        LaneSpec("instant", Horizon.INSTANT, instant, cadence=1, soft_deadline_ms=350),
        LaneSpec("medium", Horizon.SHORT, medium, cadence=1, soft_deadline_ms=2_000),
        LaneSpec("high", Horizon.MID, high, cadence=3, soft_deadline_ms=8_000),
        LaneSpec("pro", Horizon.META, pro, cadence=8, soft_deadline_ms=30_000),
    ]
